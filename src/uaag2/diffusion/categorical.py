import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import sort_edge_index
import sys
sys.path.append('.')
from uaag2.diffusion.continuous import get_beta_schedule

DEFAULT_BETAS = get_beta_schedule(kind="cosine", num_diffusion_timesteps=500)
DEFAULT_ALPHAS = 1.0 - DEFAULT_BETAS
ALPHAS_BAR = torch.cumprod(DEFAULT_ALPHAS, dim=0)


def get_one_step_transition(alpha_t: float, terminal_distribution: torch.Tensor):
    stay_prob = torch.eye(len(terminal_distribution)) * alpha_t
    diffuse_prob = (1.0 - alpha_t) * (
        torch.ones(1, len(terminal_distribution)) * (terminal_distribution.unsqueeze(0))
    )
    Q_t = stay_prob + diffuse_prob
    return Q_t


class CategoricalDiffusionKernel(torch.nn.Module):
    def __init__(
        self,
        terminal_distribution: torch.Tensor,
        alphas: torch.Tensor = DEFAULT_ALPHAS,
        num_bond_types: int = 5,
        num_atom_types: int = 16,
        num_charge_types: int = 6,
        num_is_aromatic: int = 2,
        num_hybridization: int = 8,
        num_degree: int = 5,
    ):
        super().__init__()

        self.num_bond_types = num_bond_types
        self.num_atom_types = num_atom_types
        self.num_charge_types = num_charge_types
        self.num_is_aromatic = num_is_aromatic
        self.num_hybridization = num_hybridization
        self.num_degree = num_degree
        self.num_classes = len(terminal_distribution)
        assert (terminal_distribution.sum() - 1.0).abs() < 1e-4

        self.register_buffer("eye", torch.eye(self.num_classes))
        self.register_buffer("terminal_distribution", terminal_distribution)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))
        self.register_buffer("one_minus_alphas_bar", 1.0 - self.alphas_bar)
        Qt = [
            get_one_step_transition(
                alpha_t=a.item(), terminal_distribution=terminal_distribution
            )
            for a in alphas
        ]
        self.register_buffer("Qt", torch.stack(Qt, dim=0))
        Qt_prev = torch.eye(self.num_classes)
        Qt_bar = []
        for i in range(len(alphas)):
            Qtb = Qt_prev @ Qt[i]
            Qt_bar.append(Qtb)
            Qt_prev = Qtb

        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)
        self.register_buffer("Qt_bar", Qt_bar)
        self.register_buffer("Qt_bar_prev", Qt_bar_prev)
        # print("Current Script and line number: ", __file__, "line number: ", 71)
        # from IPython import embed; embed()
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the forward categorical posterior q(xt | x0) ~ Cat(xt, p = x0_j . Qt_bar_ji)
        Args:
            x0 (torch.Tensor): _description_ one-hot vectors of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)

        Returns:
            _type_: _description_
        """

        # Qt_bar (k0, k_t)
        # mark line 1
        # print("Current Script and line number: ", __file__, "line number: ", 85)
        # from IPython import embed; embed()
        t = t.long().clamp(min=0, max=self.Qt_bar.size(0) - 1)
        x0 = x0.float()

        probs = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar[t]])
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=0.0)

        row_sums = probs.sum(-1, keepdim=True)
        invalid_rows = (~torch.isfinite(row_sums)) | (row_sums <= 0)
        if invalid_rows.any():
            probs[invalid_rows.squeeze(-1)] = self.terminal_distribution.unsqueeze(0)
            row_sums = probs.sum(-1, keepdim=True)

        probs = probs / row_sums.clamp(min=1e-8)

        return probs

    def reverse_posterior_for_every_x0(self, xt: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
        but for every possible value of x0
        Args:
            xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)
        Returns:
            _type_: _description_
        """

        # xt: (n, k_t)

        # x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype).unsqueeze(0)
        # x0 = x0.repeat((xt.size(0), 1, 1))
        # (n, k, k)

        Qt_T = self.Qt[t]  # (n, k_t-1, k_t)
        assert Qt_T.ndim == 3
        Qt_T = Qt_T.permute(0, 2, 1)
        # (n, k_t, k_t-1)

        a = torch.einsum("nj, nji -> ni", [xt, Qt_T])
        # (n, k_t-1)

        a = a.unsqueeze(1)
        # (n, 1, k_t-1)

        # b = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar_prev[t]])
        b = self.Qt_bar_prev[t]
        # (n, k_0, k_t-1)

        p0 = a * b
        # (n, k_0, k_t-1)

        # p1 = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar[t]])
        p1 = self.Qt_bar[t]
        # (n, k_0, k_t)

        ## xt_ = xt.unsqueeze(1)
        # (n, 1, k_t)

        ## p1 = (p1 * xt_).sum(-1, keepdims=True)

        p1 = torch.einsum("nij, nj -> ni", [p1, xt])
        # (n, k_0)

        p1 = p1.unsqueeze(-1)
        # (n, k_0, 1)

        probs = p0 / (p1.clamp(min=1e-5))
        # (n, k_0, k_t-1)

        # check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
        # assert check

        return probs

    def reverse_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
        Args:
            x0 (torch.Tensor): _description_ one specific one-hot vector of shape (n, k)
            xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)
        Returns:
            _type_: _description_
        """
        a = torch.einsum("nj, nji -> ni", [xt, self.Qt[t].transpose(-2, -1)])
        b = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar_prev[t]])
        p0 = a * b
        # (n, k)
        p1 = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar[t]])
        p1 = (p1 * xt).sum(-1, keepdims=True)
        # (n, 1)

        probs = p0 / p1
        check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
        assert check

        return probs

    def sample_reverse_categorical(
        self,
        xt: Tensor,
        x0: Tensor,
        t: Tensor,
        num_classes: int,
        eps: float = 1.0e-5,
        local_rank: int = 0,
    ):
        reverse = self.reverse_posterior_for_every_x0(xt=xt, t=t)
        # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
        # (N, a_0, a_t-1)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        # (N, a_t-1)
        probs = unweighted_probs / (unweighted_probs.sum(-1, keepdims=True) + eps)
        probs[torch.isnan(probs)] = 1e-5
        x_tm1 = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=num_classes,
        ).float()

        return x_tm1

    def sample_reverse_edges_categorical(
        self,
        edge_attr_global: Tensor,
        edges_pred: Tensor,
        t: Tensor,
        mask: Tensor,
        mask_i: Tensor,
        batch: Tensor,
        edge_index_global: Tensor,
        num_classes: int,
    ):  
        
        # x0 = edges_pred[mask]
        # xt = edge_attr_global[mask]
        x0 = edges_pred
        xt = edge_attr_global
        
        # t = t[batch[mask_i]]
        t = t[batch]
        reverse = self.reverse_posterior_for_every_x0(xt=xt, t=t)
        reverse = self.reverse_posterior_for_every_x0(xt=edge_attr_global, t=t[batch])
        # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
        # (N, a_0, a_t-1)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        # (N, a_t-1)
        probs = unweighted_probs / unweighted_probs.sum(-1, keepdims=True)
        # convert the nan values to 1e-5
        probs[torch.isnan(probs)] = 1e-5
        

        edges_triu = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=num_classes,
        ).float()

        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )

        return edge_attr_global, edge_index_global, mask, mask_i

    def reverse_posterior_jump(self, xt, t, s):
        """q(x_s | x_t, x0) for all x0, for arbitrary jump from t to s (s < t).

        Uses the closed-form DDPM-absorbing posterior adapted for non-unit jumps.
        Q_{s->t}_{ij} = alpha_{s->t} * delta_{ij} + (1 - alpha_{s->t}) * m_j
        where alpha_{s->t} = alphas_bar[t] / alphas_bar[s].

        Args:
            xt: (n, k) one-hot noisy vectors at time t
            t: (n,) current timestep, valid index into Qt_bar (0..T)
            s: (n,) target timestep, valid index into Qt_bar (0..T), s < t
        Returns:
            (n, k0, k) posterior tensor q(x_s=k | x_t, x0=k0)
        """
        t = t.long().clamp(min=0, max=self.Qt_bar.size(0) - 1)
        s = s.long().clamp(min=0, max=self.Qt_bar.size(0) - 1)

        # alpha_{s->t} = alphabar_t / alphabar_s
        abar_t = self.alphas_bar[t]  # (n,)
        abar_s = self.alphas_bar[s]  # (n,)
        alpha_st = (abar_t / abar_s.clamp_min(1e-8)).clamp(max=1.0)  # (n,)

        # (Q_{s->t}^T @ x_t)_k = alpha_st * x_t_k + (1 - alpha_st) * m_k
        alpha_col = alpha_st.unsqueeze(-1)  # (n, 1)
        a = alpha_col * xt + (1.0 - alpha_col) * self.terminal_distribution.unsqueeze(0)
        # a: (n, k) — likelihood of x_t given x_s for each class
        a = a.unsqueeze(1)  # (n, 1, k)

        # Qt_bar[s] gives q(x_s | x0) for each x0 class: shape (n, k0, k)
        b = self.Qt_bar[s]  # (n, k0, k)

        p0 = a * b  # (n, k0, k)

        # denominator: q(x_t | x0) @ x_t for each x0 class
        p1 = self.Qt_bar[t]  # (n, k0, k)
        p1 = torch.einsum("nij, nj -> ni", [p1, xt])  # (n, k0)
        p1 = p1.unsqueeze(-1)  # (n, k0, 1)

        probs = p0 / p1.clamp_min(1e-5)  # (n, k0, k)
        return probs

    def sample_reverse_categorical_jump(self, xt, x0, t, s, num_classes, eps=1e-5):
        """Reverse categorical step from t to s (arbitrary jump, s < t).

        Args:
            xt: (n, k) current one-hot noisy vectors at time t
            x0: (n, k) x0 prediction (soft probabilities from model)
            t: (n,) current timestep index
            s: (n,) target timestep index (s < t)
            num_classes: number of classes k
            eps: small value for numerical stability
        Returns:
            x_s: (n, k) one-hot sample at time s
        """
        reverse = self.reverse_posterior_jump(xt=xt, t=t, s=s)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        probs = unweighted_probs / (unweighted_probs.sum(-1, keepdims=True) + eps)
        probs[torch.isnan(probs)] = 1e-5
        x_s = F.one_hot(probs.multinomial(1).squeeze(), num_classes=num_classes).float()
        return x_s

    # ------------------------------------------------------------------
    # CTMC tau-leaping for empirical-marginal D3PM
    # ------------------------------------------------------------------

    def _ctmc_expected_transitions(
        self,
        xt: Tensor,
        x0_pred: Tensor,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Expected number of CTMC transitions from current class to each class.

        For empirical-marginal D3PM the reverse CTMC rate simplifies to:
          E[N(j* → k)] = (ᾱ_s - ᾱ_t) / (1 - ᾱ_t) * p_θ(x0=k | x_t)   for k ≠ j*

        Args:
            xt:      (n, K) one-hot current state at time t
            x0_pred: (n, K) soft x0 prediction (probabilities)
            t:       (n,)  current discrete timestep index
            s:       (n,)  target discrete timestep index (s < t)
        Returns:
            rates: (n, K) expected number of transitions to each class
        """
        t = t.long().clamp(0, self.alphas_bar.size(0) - 1)
        s = s.long().clamp(0, self.alphas_bar.size(0) - 1)

        abar_t = self.alphas_bar[t]           # (n,)
        abar_s = self.alphas_bar[s]           # (n,)
        # ᾱ_s > ᾱ_t because s has less noise
        delta_abar = (abar_s - abar_t).clamp_min(0.0)    # (n,)  ≥ 0
        one_minus_abar_t = (1.0 - abar_t).clamp_min(1e-8)   # (n,)

        # scalar rate per step for each sample
        rate = delta_abar / one_minus_abar_t   # (n,)

        # expected transitions to each class: rate * p_θ(x0=k)
        # zero out the current class (no self-transitions in rate)
        current = xt.argmax(-1, keepdim=True)  # (n, 1)
        rates = rate.unsqueeze(-1) * x0_pred   # (n, K)
        rates.scatter_(-1, current, 0.0)       # zero diagonal

        return rates

    def sample_ctmc_tauleaping(
        self,
        xt: Tensor,
        x0_pred: Tensor,
        t: Tensor,
        s: Tensor,
        num_classes: int,
        eps: float = 1e-5,
    ) -> Tensor:
        """CTMC tau-leaping reverse step from t to s for empirical-marginal D3PM.

        Uses Poisson sampling of the expected number of transitions and picks
        the class with the most transitions. Falls back to the D3PM posterior
        jump when the total expected rate is >= 1 (large step / high SNR regime)
        where the Poisson approximation is less reliable.

        Args:
            xt:          (n, K) one-hot current state
            x0_pred:     (n, K) soft x0 prediction from model
            t:           (n,)  current timestep index
            s:           (n,)  target timestep index (s < t)
            num_classes: K
        Returns:
            x_s: (n, K) one-hot sample at time s
        """
        rates = self._ctmc_expected_transitions(xt, x0_pred, t, s)  # (n, K)
        total_rate = rates.sum(-1)  # (n,)

        # For large steps (rate ≥ 1): Poisson approximation is inaccurate;
        # fall back to the exact posterior jump formula.
        large_step_mask = total_rate >= 1.0   # (n,) bool

        # ---- Poisson tau-leaping (small-rate regime) ----
        # Sample number of transitions to each class
        with torch.no_grad():
            transitions = torch.poisson(rates)  # (n, K)  ← integer counts

        # New class = argmax of transitions (or stay if no transitions)
        current = xt.argmax(-1)                          # (n,)
        any_transition = transitions.sum(-1) > 0          # (n,)
        tauleap_class = torch.where(
            any_transition,
            transitions.argmax(-1),
            current,
        )

        # ---- Exact posterior jump (large-rate regime) ----
        reverse = self.reverse_posterior_jump(xt=xt, t=t, s=s)
        unweighted = (reverse * x0_pred.unsqueeze(-1)).sum(1)
        unweighted[unweighted.sum(-1) == 0] = eps
        probs = unweighted / (unweighted.sum(-1, keepdim=True) + eps)
        probs[torch.isnan(probs)] = eps
        posterior_class = probs.multinomial(1).squeeze(-1)   # (n,)

        # Select based on step size
        new_class = torch.where(large_step_mask, posterior_class, tauleap_class)

        x_s = F.one_hot(new_class, num_classes=num_classes).float()
        return x_s

    def sample_ctmc_tauleaping_rk2(
        self,
        xt: Tensor,
        x0_pred_t: Tensor,
        x0_pred_mid: Tensor,
        t: Tensor,
        s: Tensor,
        num_classes: int,
        eps: float = 1e-5,
        theta: float = 0.5,
    ) -> Tensor:
        """RK2 (trapezoidal) CTMC step using two x0 predictions.

        Stage 2 of DiscreteFastSolver RK2: combines the rate at (x_t, t)
        with the rate at (x_mid, t_mid) for a higher-order update.

        Args:
            xt:          (n, K) one-hot state at t
            x0_pred_t:   (n, K) x0 prediction from GNN at (x_t,   t)
            x0_pred_mid: (n, K) x0 prediction from GNN at (x_mid, t_mid)
            t, s:        current / target timestep tensors
            theta:       weighting parameter (0.5 = trapezoidal rule)
        """
        rates_t   = self._ctmc_expected_transitions(xt, x0_pred_t,   t, s)
        rates_mid = self._ctmc_expected_transitions(xt, x0_pred_mid, t, s)
        rates_combined = (1.0 - 0.5 / theta) * rates_t + (0.5 / theta) * rates_mid

        rates_combined = rates_combined.clamp_min(0.0)
        total_rate = rates_combined.sum(-1)
        large_step_mask = total_rate >= 1.0

        with torch.no_grad():
            transitions = torch.poisson(rates_combined)

        current = xt.argmax(-1)
        any_transition = transitions.sum(-1) > 0
        tauleap_class = torch.where(any_transition, transitions.argmax(-1), current)

        # Posterior fallback for large steps
        x0_blended = (1.0 - 0.5 / theta) * x0_pred_t + (0.5 / theta) * x0_pred_mid
        x0_blended = x0_blended.clamp_min(0).div(x0_blended.clamp_min(0).sum(-1, keepdim=True).clamp_min(1e-8))
        reverse = self.reverse_posterior_jump(xt=xt, t=t, s=s)
        unweighted = (reverse * x0_blended.unsqueeze(-1)).sum(1)
        unweighted[unweighted.sum(-1) == 0] = eps
        probs = unweighted / (unweighted.sum(-1, keepdim=True) + eps)
        probs[torch.isnan(probs)] = eps
        posterior_class = probs.multinomial(1).squeeze(-1)

        new_class = torch.where(large_step_mask, posterior_class, tauleap_class)
        return F.one_hot(new_class, num_classes=num_classes).float()

    def sample_reverse_edges_categorical_jump(
        self,
        edge_attr_global,
        edges_pred,
        t,
        s,
        mask,
        mask_i,
        batch,
        edge_index_global,
        num_classes,
    ):
        """Reverse edge categorical step from t to s (arbitrary jump, s < t)."""
        x0 = edges_pred
        xt = edge_attr_global
        t_edge = t[batch]
        s_edge = s[batch]
        reverse = self.reverse_posterior_jump(xt=xt, t=t_edge, s=s_edge)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        probs = unweighted_probs / unweighted_probs.sum(-1, keepdims=True)
        probs[torch.isnan(probs)] = 1e-5

        edges_triu = F.one_hot(
            probs.multinomial(1).squeeze(),
            num_classes=num_classes,
        ).float()

        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )
        return edge_attr_global, edge_index_global, mask, mask_i

    def sample_edges_categorical(
        self,
        t,
        edge_index_global,
        edge_attr_global,
        data_batch,
        return_one_hot=True,
    ):
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        edge_attr_triu = edge_attr_global[mask]

        edge_attr_triu_ohe = F.one_hot(
            edge_attr_triu.long(), num_classes=self.num_bond_types
        ).float()
        t_edge = t[data_batch[mask_i]].long().clamp(min=0, max=self.Qt_bar.size(0) - 1)
        probs = self.marginal_prob(edge_attr_triu_ohe, t=t_edge)
        edges_t_given_0 = probs.multinomial(
            1,
        ).squeeze()
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global_perturbed = torch.stack([j, i], dim=0)
        edge_attr_global_perturbed = torch.concat(
            [edges_t_given_0, edges_t_given_0], dim=0
        )
        edge_index_global_perturbed, edge_attr_global_perturbed = sort_edge_index(
            edge_index=edge_index_global_perturbed,
            edge_attr=edge_attr_global_perturbed,
            sort_by_row=False,
        )

        edge_index_global_perturbed_2, edge_attr_global_original = sort_edge_index(
            edge_index=edge_index_global_perturbed,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )
        
        # check edge_index_global_perturbed_2 == edge_index_global_perturbed
        assert torch.all(edge_index_global_perturbed_2 == edge_index_global_perturbed)
        
        edge_attr_global_perturbed = (
            F.one_hot(
                edge_attr_global_perturbed, num_classes=self.num_bond_types
            ).float()
            if return_one_hot
            else edge_attr_global_perturbed
        )
        # from IPython import embed; embed()
        edge_attr_global_original = F.one_hot(
                edge_attr_global_original.long(), num_classes=self.num_bond_types
        ).float()
        
        return edge_attr_global_perturbed, edge_attr_global_original

    def sample_categorical(
        self, t, x0, data_batch, dataset_info, num_classes=16, type="atoms"
    ):
        assert type in ["atoms", "charges", "ring", "aromatic", "hybridization", "degree"]

        # if type == "charges":
        #     x0 = dataset_info.one_hot_charges(x0)
        # else:
            # print("Current Script and line number: ", __file__, "line number: ", 297)   
            # from IPython import embed; embed()
        labels = x0.squeeze().long().clamp(min=0, max=num_classes - 1)
        x0 = F.one_hot(labels, num_classes=num_classes).float()
        t_index = t[data_batch].long().clamp(min=0, max=self.Qt_bar.size(0) - 1)
        probs = self.marginal_prob(x0.float(), t_index)
        x0_perturbed = probs.multinomial(
            1,
        ).squeeze()
        x0_perturbed = F.one_hot(x0_perturbed, num_classes=num_classes).float()

        return x0, x0_perturbed


def _some_debugging():
    num_classes = 5
    uniform_distribution = (
        torch.ones(
            num_classes,
        )
        / num_classes
    )
    absorbing_distribution = torch.zeros(
        num_classes,
    )
    absorbing_distribution[0] = 1.0
    absorbing_distribution = torch.tensor(
        [9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02]
    )

    atoms_drugs = [
        4.4119e-01,
        1.0254e-06,
        4.0564e-01,
        6.4677e-02,
        6.6144e-02,
        4.8741e-03,
        0.0000e00,
        9.1150e-07,
        1.0847e-04,
        1.2260e-02,
        4.0306e-03,
        0.0000e00,
        1.0503e-03,
        1.9806e-05,
        0.0000e00,
        7.5958e-08,
    ]

    edges_drugs = [9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02]

    atoms_qm9 = [0.5122, 0.3526, 0.0562, 0.0777, 0.0013]
    edges_qm9 = [0.8818, 0.1104, 0.0060, 0.0018, 0.0000]

    C0 = CategoricalDiffusionKernel(
        terminal_distribution=uniform_distribution, alphas=DEFAULT_ALPHAS
    )

    C1 = CategoricalDiffusionKernel(
        terminal_distribution=torch.tensor(edges_drugs), alphas=DEFAULT_ALPHAS
    )

    t = 290
    a = C0.Qt_bar[t]
    alphas_bar_t = C0.alphas_bar[t].unsqueeze(-1)
    b = alphas_bar_t * C0.eye + (
        (1.0 - alphas_bar_t) * torch.ones_like(C0.terminal_distribution)
    ).unsqueeze(-1) * C0.terminal_distribution.unsqueeze(0)

    print(a - b)

    alphas_t = C0.alphas[t].unsqueeze(-1)
    a = C0.Qt[t]
    b = alphas_t * C0.eye + (
        (1.0 - alphas_t) * torch.ones_like(C0.terminal_distribution)
    ).unsqueeze(-1) * C0.terminal_distribution.unsqueeze(0)

    print(a - b)
    Qt = C1.Qt[t]
    return None
