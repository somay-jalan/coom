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

logger = logging.getLogger(__name__)

class NonIterableStreamingDataset(torch.utils.data.Dataset):
    """
    Wrapper to convert iterable StreamingDataset to non-iterable (map-style) dataset.
    This version is safe for multi-worker data loading by using a FileLock to
    synchronize iterator re-initialization at epoch boundaries.
    """
    def __init__(self, streaming_dataset: StreamingDataset):
        self.streaming_dataset = streaming_dataset
        
        # The true length per epoch for one device/worker. StreamingDataset calculates this.
        self._length = self.streaming_dataset.length
        
        self.iterator = None
        self.lock = None

        # Define a unique path for the lock file based on the underlying dataset's
        # unique shared memory prefix. This ensures the lock is specific to this
        # dataset instance (e.g., train vs. val).
        lock_name = f"non_iterable_wrapper.{self.streaming_dataset._shm_prefix_int}.lock"
        self.lock_path = os.path.join(self.streaming_dataset._filelock_root, lock_name)
        logger.info(f"NonIterableStreamingDataset created with lock path: {self.lock_path}")

    def __len__(self):
        # Return the true length of one epoch. The Trainer handles multiple epochs.
        return self._length

    def _init_iterator_and_lock(self):
        """Initializes the iterator and lock within a worker."""
        if self.lock is None:
            # Create a file lock that is shared across all workers for this dataset.
            # Timeout is important to prevent indefinite hangs.
            os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
            self.lock = FileLock(self.lock_path, timeout=120)

        # Each worker gets its own iterator. StreamingDataset's __iter__ is designed
        # to coordinate these workers using its own internal barriers.
        self.iterator = iter(self.streaming_dataset)

    def __getitem__(self, idx: int):
        # The iterator should be initialized by the worker_init_fn.
        # This is a fallback for worker_num=0 or other edge cases.
        if self.iterator is None:
            self._init_iterator_and_lock()

        try:
            return next(self.iterator)
        except StopIteration:
            # The epoch has ended. All workers will hit this.
            # Use the lock to ensure only one worker at a time handles re-initialization.
            with self.lock:
                # After acquiring the lock, it's possible another worker already
                # finished re-initializing. The simplest way to check is to try fetching
                # again from the (potentially new) iterator.
                try:
                    return next(self.iterator)
                except StopIteration:
                    # If it still fails, it's this worker's turn to re-initialize.
                    logger.debug(f"Worker {get_worker_info().id if get_worker_info() else 0} is re-initializing the iterator.")
                    self._init_iterator_and_lock()
                    return next(self.iterator)

# This is the required worker_init_fn for the DataLoader
def _streaming_worker_init_fn(worker_id: int):
    """
    Worker init function to properly initialize the NonIterableStreamingDataset.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, NonIterableStreamingDataset):
            dataset._init_iterator_and_lock()

class StreamingPreTrainingDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 seq_length: int,
                 micro_batch_size: int,
                 global_batch_size: int,
                 num_workers: int,
                 remote_path: str | None = None,
                 eod_token_id: int = 50256,
                 seed: int = 1234,
                 rampup_batch_size: Optional[List[int]] = None):
        super().__init__()
        self.remote_path = remote_path
        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.seed = seed

        self._streaming_state = None
        self._train_ds: Optional[NonIterableStreamingDataset] = None
        self._validation_ds: Optional[NonIterableStreamingDataset] = None
        
        self.collator = StreamingCollator(
            seq_length=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            eod_token_id=eod_token_id,
        )
        self.rampup_batch_size = rampup_batch_size
        # Your MegatronDataSampler would be used by the trainer framework
        # self.data_sampler = ...

    def setup(self, stage: str = ""):
        # The logic to create unique local paths per rank is good. Keep it.
        # However, it's better to let StreamingDataset handle this internally if possible.
        # For simplicity, we'll keep your existing path logic.
        rank = dist.get_rank() if dist.is_initialized() else 0
        train_local_path = f"{self.dataset_path}/train_rank_{rank}"
        val_local_path = f"{self.dataset_path}/val_rank_{rank}"
        os.makedirs(train_local_path, exist_ok=True)
        os.makedirs(val_local_path, exist_ok=True)

        # Batch size for StreamingDataset partitioning should be the micro_batch_size
        streaming_partitioning_batch_size = self.micro_batch_size

        train_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/train', # Example path
            local=train_local_path,
            shuffle=True,
            shuffle_seed=self.seed,
            batch_size=streaming_partitioning_batch_size,
            predownload=max(self.num_workers * 2, 8),
        )

        val_streaming_ds = StreamingDataset(
            remote='s3://erpbucketerp/dummy_streaming_dataset/val', # Example path
            local=val_local_path,
            shuffle=False,
            batch_size=streaming_partitioning_batch_size,
            predownload=max(self.num_workers * 2, 8),
        )
        
        self._train_ds = NonIterableStreamingDataset(train_streaming_ds)
        self._validation_ds = NonIterableStreamingDataset(val_streaming_ds)

        # Load state dict if resuming. This must be done on the underlying dataset.
        if self._streaming_state:
            # We need to know how many samples were processed. This state dict should
            # be saved by the Trainer/Callback which has this info.
            self._train_ds.streaming_dataset.load_state_dict(self._streaming_state)

    def train_dataloader(self) -> DataLoader:
        # CRITICAL: Use a standard torch.utils.data.DataLoader.
        # Do NOT use StreamingDataLoader with a map-style dataset.
        
        # The DataLoader batch_size determines how many items (from __getitem__)
        # are passed to the collate_fn. Let's provide enough to build a microbatch.
        # A safe bet is a bit more than the micro_batch_size.
        dl_batch_size = self.micro_batch_size * 2
        
        return DataLoader(
            dataset=self._train_ds,
            batch_size=dl_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=_streaming_worker_init_fn, # IMPORTANT
        )

    def val_dataloader(self) -> DataLoader:
        dl_batch_size = self.micro_batch_size * 2
        return DataLoader(
            dataset=self._validation_ds,
            batch_size=dl_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=_streaming_worker_init_fn, # IMPORTANT
        )

    # NOTE: state_dict and load_state_dict for a DataModule are complex.
    # Lightning doesn't have a standard hook for saving dataloader state that provides
    # the number of samples processed. This usually requires a custom Callback.
    # The methods below show the *intent*. You will need a mechanism to get `num_samples_processed`.

    def state_dict(self, num_samples_processed: int) -> dict:
        """
        Creates the state dict. This needs to be called from a callback
        that has access to the number of samples processed in the epoch.
        """
        if self._train_ds:
            return self._train_ds.streaming_dataset.state_dict(num_samples=num_samples_processed, from_beginning=False)
        return {}

    def load_state_dict(self, state_dict: dict):
        """
        Loads the state dict. Called before setup().
        """
        if state_dict:
            self._streaming_state = state_dict
