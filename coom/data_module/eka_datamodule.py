from nemo.collections.llm import PreTrainingDataModule

class EKAPreTrainingDataModule(PreTrainingDataModule):
    """
    Currently identical to NeMo's PreTrainingDataModule.
    Defined separately for modularity, to allow future changes or extensions.
    """
    pass