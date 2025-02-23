import torch
from torch.utils.data import DataLoader

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset

from coom.training.utils import print_rank_0
from coom.training.tokenizer.tokenizer import _NullTokenizer

_SEQUENCE_LENGTH = 64

def get_train_data_iterator():

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator
