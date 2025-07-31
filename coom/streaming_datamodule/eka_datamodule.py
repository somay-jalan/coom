import torch
import numpy as np
import lightning.pytorch as pl
import streaming
from streaming import MDSWriter, StreamingDataset, StreamingDataLoader
from torch.utils.data import DataLoader, get_worker_info
from typing import Optional, Dict, Any, List, Union
from nemo.lightning.pytorch.plugins import MegatronDataSampler
import os
from nemo.lightning.data import setup_microbatch_calculator
import torch.distributed as dist
import random

class StreamingCollator:

    def __init__(self, seq_length: int, micro_batch_size: int, eod_token_id: int = 50256):
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.eod_token_id = eod_token_id
        self.chunk_size = seq_length + 1

    def __call__(self, samples: List[Dict[str, any]]) -> Optional[Dict[str, torch.Tensor]]:
        all_tokens = [sample['tokens'] for sample in samples]
        concatenated_tokens = np.concatenate(all_tokens)
        
        required_tokens = self.micro_batch_size * self.chunk_size
        if len(concatenated_tokens) < required_tokens:
            pad_len = required_tokens - len(concatenated_tokens)
            concatenated_tokens = np.concatenate([
                concatenated_tokens, 
                np.full(pad_len, self.eod_token_id, dtype=np.int32)
            ])

        buffer = torch.from_numpy(concatenated_tokens[:required_tokens]).long()
        buffer = buffer.view(self.micro_batch_size, self.chunk_size)

        tokens = buffer[:, :-1].contiguous()
        labels = buffer[:, 1:].contiguous()

        position_ids = torch.arange(self.seq_length, dtype=torch.long).unsqueeze(0).repeat(self.micro_batch_size, 1)

        loss_mask = torch.ones_like(labels, dtype=torch.float)

        eod_indices = (labels == self.eod_token_id).nonzero(as_tuple=True)
        for row, col in zip(*eod_indices):
            loss_mask[row, :col + 1] = 0.0
            
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

class NonIterableStreamingDataset(torch.utils.data.Dataset):
    """
    Wrapper to convert iterable StreamingDataset to non-iterable (map-style) dataset.
    This version properly handles multi-worker scenarios via worker_init_fn.
    """
    def __init__(self, streaming_dataset, estimated_length: int = 10000):
        self.streaming_dataset = streaming_dataset
        self._estimated_length = estimated_length
        self.iterator = None
        self._worker_iterator = None
    
    def __len__(self):
        return self._estimated_length
    
    def __getitem__(self, idx):
        # Lazily init or reset iterator
        if self._worker_iterator is None:
            self._worker_iterator = iter(self.streaming_dataset)
        try:
            return next(self._worker_iterator)
        except StopIteration:
            self._worker_iterator = iter(self.streaming_dataset)
            return next(self._worker_iterator)
    
    def reset_iterator(self):
        self._worker_iterator = None

def _streaming_worker_init_fn(worker_id: int):
    """
    Worker init function for StreamingDataset in a map-style wrapper.
    """
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    if hasattr(dataset, "reset_iterator"):
        dataset.reset_iterator()

class StreamingPreTrainingDataModule(pl.LightningDataModule):
    """
    DataModule for GPT pre-training using StreamingDataset.
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
                 estimated_dataset_length: int = 100,):
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
        self.collator = StreamingCollator(
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            eod_token_id=eod_token_id,
        )
        self.rampup_batch_size = rampup_batch_size
        
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
        )

    def setup(self, stage: str = ""):
        print("inside setup")
        # Clean shared memory only on rank 0 to avoid race conditions
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     streaming.base.util.clean_stale_shared_memory()
            
            # # Also clean up old cache directories to prevent conflicts
            # import glob
            # import shutil
            
            # # Remove old rank-specific directories
            # for pattern in [f"{self.dataset_path}/train_rank_*", f"{self.dataset_path}/val_rank_*"]:
            #     for old_dir in glob.glob(pattern):
            #         try:
            #             if os.path.exists(old_dir):
            #                 shutil.rmtree(old_dir)
            #                 print(f"Cleaned up old directory: {old_dir}")
            #         except Exception as e:
            #             print(f"Warning: Could not clean {old_dir}: {e}")
        
        # # Synchronize all processes after cleanup
        # if dist.is_initialized():
        #     dist.barrier()
        
        streaming_batch_size = self.micro_batch_size
        
        # Get rank for unique local directories
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Create truly unique local directories using rank, process ID, and timestamp
        import time
        timestamp = int(time.time())
        process_id = os.getpid()
        
        train_local_path = f"{self.dataset_path}/train_rank_{rank}_pid_{process_id}_{timestamp}"
        val_local_path = f"{self.dataset_path}/val_rank_{rank}_pid_{process_id}_{timestamp}"
        print("SOMETHING ")
        # Ensure directories exist
        os.makedirs(train_local_path, exist_ok=True)
        os.makedirs(val_local_path, exist_ok=True)
        
        train_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/train',
            local=train_local_path,
            shuffle=True,
            shuffle_seed=self.seed,
            batch_size=streaming_batch_size,
            predownload=max(self.num_workers * 2, 8),
        )
        
        val_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/val',
            local=val_local_path,
            shuffle=False,
            batch_size=streaming_batch_size,
            predownload=max(self.num_workers * 2, 8),
        )
        
        self._train_ds = NonIterableStreamingDataset(
            train_streaming_ds, 
            estimated_length=self.estimated_dataset_length
        )
        self._validation_ds = NonIterableStreamingDataset(
            val_streaming_ds, 
            estimated_length=self.estimated_dataset_length // 10
        )
        
        if self._streaming_state:
            train_streaming_ds.load_state_dict(self._streaming_state)

    def train_dataloader(self) -> StreamingDataLoader:
        self._train_dataloader = StreamingDataLoader(
            dataset=self._train_ds,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            # pin_memory=True,
            # persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=_streaming_worker_init_fn,
        )
        
        print(f"Train DataLoader created: {self._train_dataloader}")
        return self._train_dataloader

    def val_dataloader(self) -> StreamingDataLoader:
        self._val_dataloader = StreamingDataLoader(
            dataset=self._validation_ds,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            # pin_memory=True,
            # persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=_streaming_worker_init_fn,
        )
        
        print(f"Validation DataLoader created: {self._val_dataloader}")
        return self._val_dataloader

    def state_dict(self):
        if hasattr(self._train_ds, 'streaming_dataset'):
            return self._train_dataloader.state_dict()
        elif hasattr(self._train_ds, 'state_dict'):
            return self._train_ds.state_dict()
        return {}
    
    def load_state_dict(self, state_dict):
        self._train_dataloader.load_state_dict(state_dict)
    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule._train_ds, "reset_iterator"):
            self.trainer.datamodule._train_ds.reset_iterator()
