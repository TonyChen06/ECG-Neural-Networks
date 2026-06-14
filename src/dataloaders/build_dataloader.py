import argparse

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.gpu_setup import get_world_size, get_rank

from dataloaders.dataset_mixer import DatasetMixer


class BuildDataLoader:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.args = args
        self.dataset_mixer = DatasetMixer(self.args)
        self._dbeta_collator = self._maybe_build_dbeta_collator()

    def _maybe_build_dbeta_collator(self):
        if getattr(self.args, "neural_network", None) != "dbeta":
            return None
        from dataloaders.dbeta_collator import DBETACollator
        retriever = None
        if getattr(self.args, "dbeta_n3s_index", None):
            from dataloaders.n3s import N3SRetriever
            retriever = N3SRetriever(self.args.dbeta_n3s_index)
        return DBETACollator(
            tokenizer_name=getattr(self.args, "dbeta_text_model", "google/flan-t5-base"),
            max_text_size=getattr(self.args, "dbeta_max_text_len", 256),
            mlm_prob=getattr(self.args, "dbeta_mlm_prob", 0.25),
            neg_ratio=getattr(self.args, "dbeta_neg_ratio", 0.5),
            n3s_retriever=retriever,
            seed=self.args.seed,
        )

    def build_dataloader(
        self,
    ):
        torch_dataset = self.dataset_mixer.build_torch_dataset()
        torch_data_loader = self.build_torch_dataloader(torch_dataset)
        return torch_data_loader

    def build_torch_dataloader(self, torch_dataset):
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        if "train" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=(sampler is None),
                num_workers=self.args.num_workers,
                sampler=sampler,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
                persistent_workers=(self.args.num_workers > 0),
                prefetch_factor=4 if self.args.num_workers > 0 else None,
            )
        elif "eval" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=1,  # batched inference/eval not implemented
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
            )
        return torch_data_loader

    def get_torch_dataloader_sampler(
        self,
        torch_dataset,
    ):
        if self.args.distributed:
            sampler = DistributedSampler(torch_dataset, num_replicas=get_world_size(), 
                                         rank=get_rank(), seed=self.args.seed, shuffle=True)
        else:
            sampler = None
        return sampler

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        if self._dbeta_collator is not None:
            return self._dbeta_collator(batch)
        return torch.utils.data.dataloader.default_collate(batch)