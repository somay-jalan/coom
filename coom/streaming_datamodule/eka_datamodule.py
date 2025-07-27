import torch
import numpy as np
import lightning.pytorch as pl
import streaming
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Union
from nemo.lightning.pytorch.plugins import MegatronDataSampler
import os
from nemo.lightning.data import setup_microbatch_calculator
import torch.distributed as dist


class StreamingCollator:
    """
    Collator that packs multiple documents into a single sequence and creates all necessary tensors.
    
    Output keys: ['tokens', 'labels', 'loss_mask', 'position_ids']
    """
    def __init__(self, seq_length: int, eod_token_id: int = 50256):
        self.seq_length = seq_length
        self.eod_token_id = eod_token_id
        # We need seq_length + 1 tokens to create the labels via shifting
        self.chunk_size = seq_length + 1

    def __call__(self, samples: List[Dict[str, any]]) -> Optional[Dict[str, torch.Tensor]]:
        # Concatenate all tokens from the samples in the batch
        all_tokens = [sample['tokens'] for sample in samples]
        concatenated_tokens = np.concatenate(all_tokens)
        
        # The number of sequences we can form
        num_sequences = len(concatenated_tokens) // self.chunk_size
        
        # Truncate to a multiple of chunk_size
        trunc_len = num_sequences * self.chunk_size
        buffer = torch.from_numpy(concatenated_tokens[:trunc_len]).long()
        
        # Reshape into a batch of sequences
        buffer = buffer.view(num_sequences, self.chunk_size)
        
        # 1. Create tokens (inputs) and labels
        tokens = buffer[:, :-1].contiguous()
        labels = buffer[:, 1:].contiguous()
        
        position_ids = torch.arange(self.seq_length, dtype=torch.long).unsqueeze(0).expand_as(tokens)
        
        loss_mask = torch.ones_like(labels, dtype=torch.float)

        eod_indices = (labels == self.eod_token_id).nonzero(as_tuple=True)

        for row, col in zip(*eod_indices):
            loss_mask[row, :col + 1] = 0.0
            
        return {
            "tokens": tokens, 
            "labels": labels, 
            "loss_mask": loss_mask,
            "position_ids": position_ids
        }


class NonIterableStreamingDataset(torch.utils.data.Dataset):
    """
    Wrapper to convert iterable StreamingDataset to non-iterable (map-style) dataset.
    When __getitem__ is called, it returns the next item from the iterator.
    """
    def __init__(self, streaming_dataset, estimated_length: int = 10000):
        self.streaming_dataset = streaming_dataset
        self.iterator = None
        self._length = estimated_length
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # Ignore the idx and just return next item from iterator
        if self.iterator is None:
            self.iterator = iter(self.streaming_dataset)
        
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator for new epoch
            self.iterator = iter(self.streaming_dataset)
            return next(self.iterator)
    
    def reset_iterator(self):
        """Reset the iterator - useful for new epochs"""
        self.iterator = None


class StreamingPreTrainingDataModule(pl.LightningDataModule):
    """
    DataModule for GPT pre-training using StreamingDataset converted to map-style dataset.
    """
    def __init__(self, 
                 dataset_path: str, 
                 seq_length: int, 
                 micro_batch_size: int, 
                 global_batch_size: int, 
                 num_workers: int, 
                 remote_path: str | None = None, 
                 eod_token_id: int = 50256, 
                 seed: int = 1234, 
                 rampup_batch_size: Optional[List[int]] = None,
                 estimated_dataset_length: int = 10000):
        super().__init__()
        self.remote_path = remote_path
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.estimated_dataset_length = estimated_dataset_length
        
        self._streaming_state = None
        self._train_ds = None
        self._validation_ds = None
        self.collator = StreamingCollator(seq_length=self.seq_length, eod_token_id=eod_token_id)
        self.rampup_batch_size = rampup_batch_size
        
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
        )

    def setup(self, stage: str = ""):
        streaming_batch_size = self.micro_batch_size
        from torch.distributed import is_initialized, get_rank

        rank = get_rank() if is_initialized() else 0
        
        # Create StreamingDatasets
        train_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/train',
            local=f"{self.dataset_path}/train_rank{rank}",
            shuffle=True,
            shuffle_seed=self.seed,
            batch_size=streaming_batch_size,
        )
        
        val_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/val',
            local=f"{self.dataset_path}/val_rank{rank}",
            shuffle=False,
            batch_size=streaming_batch_size,
        )
        
        # Convert to non-iterable datasets
        self._train_ds = NonIterableStreamingDataset(
            train_streaming_ds, 
            estimated_length=self.estimated_dataset_length
        )
        self._validation_ds = NonIterableStreamingDataset(
            val_streaming_ds, 
            estimated_length=self.estimated_dataset_length // 10  # Smaller validation set
        )
        
        if self._streaming_state:
            train_streaming_ds.load_state_dict(self._streaming_state)

    def train_dataloader(self) -> DataLoader:
        print("Creating train DataLoader")
        
        dataloader = DataLoader(
            dataset=self._train_ds,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Train DataLoader created: {dataloader}")
        
        # Apply MegatronDataSampler if needed
        dataloader = self.data_sampler.transform_dataloader(dataloader)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        print("Creating validation DataLoader")
        
        dataloader = DataLoader(
            dataset=self._validation_ds,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Validation DataLoader created: {dataloader}")
        
        dataloader = self.data_sampler.transform_dataloader(dataloader)
        return dataloader

    def state_dict(self):
        return self._train_ds.streaming_dataset.state_dict() if self._train_ds else {}
    
    def load_state_dict(self, state_dict):
        self._streaming_state = state_dict