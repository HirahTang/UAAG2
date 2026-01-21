from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import tempfile
from argparse import Namespace
from torch_geometric.loader import DataLoader

# Add project root to sys.path to ensure imports work
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.uaag2.equivariant_diffusion import Trainer
from src.uaag2.datasets.uaag_dataset import Dataset_Info, UAAG2Dataset_sampling

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Global variables for model and config
MODEL = None
DATASET_INFO = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants (should match training/eval or be loaded from config)
CHECKPOINT_PATH = "models/good_model/last.ckpt"
DATA_INFO_PATH = "data/statistic.pkl"
DEFAULT_HPARAMS = {
    "batch_size": 1,
    "num_workers": 0,
    "virtual_node_size": 15,
    "num_samples": 3,  # Generate 5 samples
    "split_index": 0,
    "save_dir": "temp_generations",
    # Add other necessary hparams defaults or load from checkpoint
}


def load_data_info():
    global DATASET_INFO
    if os.path.exists(DATA_INFO_PATH):
        # Create a minimal namespace for Dataset_Info initialization
        hparams = Namespace(
            remove_hs=False,
            dataset_root="----",
            load_ckpt=None,
            load_ckpt_from_pretrained=None,
            dataset="drugs",
            use_adaptive_loader=True,
            select_train_subset=False,
            train_size=0.99,
            val_size=0.01,
            test_size=100,
            dropout_prob=0.3,
            batch_size=32,
            inference_batch_size=32,
            gamma=0.975,
            grad_clip_val=10.0,
            lr_scheduler="reduce_on_plateau",
            optimizer="adam",
            lr=5e-4,
            lr_min=5e-5,
            lr_step_size=10000,
            lr_frequency=5,
            lr_patience=20,
            lr_cooldown=5,
            lr_factor=0.75,
            sdim=256,
            vdim=64,
            latent_dim=None,
            rbf_dim=32,
            edim=32,
            edge_mp=False,
            vector_aggr="mean",
            num_layers=7,
            fully_connected=True,
            local_global_model=False,
            local_edge_attrs=False,
            use_cross_product=False,
            cutoff_local=7.0,
            cutoff_global=10.0,
            energy_training=False,
            property_training=False,
            regression_property="polarizability",
            energy_loss="l2",
            use_pos_norm=False,
            additional_feats=True,
            use_qm_props=False,
            build_mol_with_addfeats=False,
            continuous=False,
            noise_scheduler="cosine",
            eps_min=1e-3,
            beta_min=1e-4,
            beta_max=2e-2,
            timesteps=500,
            max_time=None,
            lc_coords=3.0,
            lc_atoms=0.4,
            lc_bonds=2.0,
            lc_charges=1.0,
            lc_mulliken=1.5,
            lc_wbo=2.0,
            pocket_noise_std=0.1,
            use_ligand_dataset_sizes=False,
            loss_weighting="snr_t",
            snr_clamp_min=0.05,
            snr_clamp_max=1.50,
            ligand_pocket_interaction=False,
            diffusion_pretraining=False,
            continuous_param="data",
            atoms_categorical=True,
            bonds_categorical=True,
            atom_type_masking=True,
            use_absorbing_state=False,
            num_bond_classes=5,
            num_charge_classes=6,
            bond_guidance_model=False,
            bond_prediction=False,
            bond_model_guidance=False,
            energy_model_guidance=False,
            polarizabilty_model_guidance=False,
            ckpt_bond_model=None,
            ckpt_energy_model=None,
            ckpt_polarizabilty_model=None,
            guidance_scale=1.0e-4,
            context_mapping=False,
            num_context_features=0,
            properties_list=[],
            property_prediction=False,
            prior_beta=1.0,
            sdim_latent=256,
            vdim_latent=64,
            edim_latent=32,
            num_layers_latent=7,
            latent_layers=7,
            latentmodel="diffusion",
            latent_detach=False,
            id="0",
            gpus=1,
            num_epochs=300,
            eval_freq=1.0,
            test_interval=5,
            no_h=False,
            precision=32,
            detect_anomaly=False,
            num_workers=4,
            max_num_conformers=5,
            accum_batch=1,
            max_num_neighbors=128,
            ema_decay=0.9999,
            weight_decay=0.9999,
            seed=42,
            backprop_local=False,
            num_test_graphs=10000,
            calculate_energy=False,
            save_xyz=False,
            variational_sampling=False,
            benchmark_path=None,
            num_samples=500,
            virtual_node=1,
            virtual_node_size=15,
            split_index=0,
            save_dir="ProteinGymSampling",
        )
        DATASET_INFO = Dataset_Info(hparams, DATA_INFO_PATH)
        return hparams
    else:
        raise FileNotFoundError(f"{DATA_INFO_PATH} not found")


@app.on_event("startup")
async def startup_event():
    global MODEL
    # Allow safe globals for torch.load
    # check if torch.serialization has add_safe_globals
    if hasattr(torch.serialization, "add_safe_globals"):
        # We might need to add specific classes if they are in the pt file
        # But here we load the checkpoint
        pass

    try:
        hparams = load_data_info()
        print(f"Loading model from {CHECKPOINT_PATH}...")

        print(f"Loading model from {CHECKPOINT_PATH}...")

        # Load hparams from checkpoint if possible
        # Trainer.load_from_checkpoint merges passed hparams with checkpoint hparams

        MODEL = Trainer.load_from_checkpoint(
            CHECKPOINT_PATH,
            hparams=hparams,
            dataset_info=DATASET_INFO,
            strict=False,  # To avoid errors with missing keys if any
        ).to(DEVICE)
        MODEL.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # We don't raise here to allow app to start, but generation will fail
        pass


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Load the data
        try:
            # Handle the torch load issue
            data_list = torch.load(temp_path, weights_only=False)
        except Exception as e:
            # Fallback or error handling
            raise HTTPException(status_code=400, detail=f"Invalid .pt file: {str(e)}")

        if isinstance(data_list, list):
            if len(data_list) == 0:
                raise HTTPException(status_code=400, detail="Empty data list")
            graph = data_list[0]  # Take first
        else:
            graph = data_list  # Assume it's a single Data object

        # Create dataset and loader
        hparams = Namespace(**DEFAULT_HPARAMS)
        hparams.save_dir = tempfile.mkdtemp()
        hparams.id = "api_gen"

        dataset_save_path = os.path.join(hparams.save_dir, "samples")

        dataset = UAAG2Dataset_sampling(
            graph,
            hparams,
            dataset_save_path,
            DATASET_INFO,
            sample_size=hparams.virtual_node_size,
            sample_length=hparams.num_samples,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )

        # Run generation
        # generate_ligand saves to os.path.join(self.save_dir, save_path, f"iter_{i}")

        # We will pass 'gen_output' as save_path
        MODEL.hparams.save_dir = hparams.save_dir
        MODEL.hparams.id = hparams.id
        # Update model save_dir manually because it was set during init
        MODEL.save_dir = os.path.join(hparams.save_dir, f"run{hparams.id}")

        MODEL.generate_ligand(dataloader, save_path="gen_output", verbose=True, device=DEVICE)

        # Find the generated file
        # Path: {hparams.save_dir}/run{hparams.id}/gen_output/iter_0/batch_0/final/ligand.mol

        base_output_dir = os.path.join(MODEL.save_dir, "gen_output")

        # Collect results
        results = []

        # Iterate over iterations (samples)
        for i in range(hparams.num_samples):
            mol_path = os.path.join(base_output_dir, f"iter_{i}", "batch_0", "final", "ligand.mol")
            xyz_path = os.path.join(base_output_dir, f"iter_{i}", "batch_0", "final", "ligand.xyz")

            if os.path.exists(mol_path):
                with open(mol_path, "r") as f:
                    mol_content = f.read()
                results.append({"format": "mol", "content": mol_content, "id": i})
            elif os.path.exists(xyz_path):
                with open(xyz_path, "r") as f:
                    xyz_content = f.read()
                results.append({"format": "xyz", "content": xyz_content, "id": i})

        if not results:
            # Check for logs or deeper
            raise HTTPException(status_code=500, detail="Generation failed to produce output files")

        return JSONResponse(content={"results": results})

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(temp_path)
        # shutil.rmtree(hparams.save_dir) # Clean up? Maybe keep for debug for now


# Mount static files
app.mount("/", StaticFiles(directory="src/uaag2/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
