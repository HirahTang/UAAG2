from typing import Optional

import torch
from torch.utils.data import Subset
from torch_geometric.data.lightning import LightningDataset


class Mixin:
    def __getitem__(self, idx):
        return self.dataloaders["train"][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ["train", "val", "test"]:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    unique, counts = torch.unique(data.batch, return_counts=True)
                    for count in counts:
                        all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for batch in self.dataloaders["train"]:
            for data in batch:
                num_classes = data.x.shape[1]
                break
            break

        counts = torch.zeros(num_classes)

        for split in ["train", "val", "test"]:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = 5

        d = torch.zeros(num_classes)

        for split in ["train", "val", "test"]:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue
                    unique, counts = torch.unique(data.batch, return_counts=True)

                    all_pairs = 0
                    for count in counts:
                        all_pairs += count * (count - 1)

                    num_edges = data.edge_index.shape[1]
                    num_non_edges = all_pairs - num_edges
                    edge_types = data.edge_attr.sum(dim=0)
                    assert num_non_edges >= 0
                    d[0] += num_non_edges
                    d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)  # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for split in ["train", "val", "test"]:
            for i, batch in enumerate(self.dataloaders[split]):
                for data in batch:
                    if data is None:
                        continue

                    n = data.x.shape[0]

                    for atom in range(n):
                        edges = data.edge_attr[data.edge_index[0] == atom]
                        edges_total = edges.sum(dim=0)
                        valency = (edges_total * multiplier).sum()
                        valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDataModule(Mixin, LightningDataset):
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        super().__init__(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            pin_memory=getattr(cfg.dataset, "pin_memory", False),
        )
        self.cfg = cfg
