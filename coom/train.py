import os
import torch
import importlib
from omegaconf import OmegaConf
from nemo import lightning as nl
from megatron.core.optimizer import OptimizerConfig
from nemo.collections import llm
from . import model
from . import config_classes

def trainer(config_path):
    """
    Trains a model using the main config file path.
    
    Args:
        config_path (str): Path to the main configuration YAML file
    """

    main_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    prefix = main_cfg["config_path_prefix"]

    model_cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(prefix, main_cfg["base_model_configuartion_path"])), resolve=True)
    data_cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(prefix, main_cfg["dataLoader_config_path"])), resolve=True)
    opt_cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(prefix, main_cfg["optimizer_config_path"])), resolve=True)
    trainer_cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(prefix, main_cfg["trainer_config_path"])), resolve=True)
    logger_cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(prefix, main_cfg["logger_config_path"])), resolve=True)

    ModelClass = getattr(model, main_cfg["base_model"])
    ConfigClass = getattr(config_classes, main_cfg["base_model_config"])

    model_config = ConfigClass(**model_cfg)
    modelClasss = ModelClass(model_config)

    strategy = nl.MegatronStrategy(**trainer_cfg["strategy"])
    optimizer_config = OptimizerConfig(**opt_cfg)
    optimizer = nl.MegatronOptimizerModule(config=optimizer_config)

    trainer = nl.Trainer(
        devices=trainer_cfg["devices"],
        max_steps=trainer_cfg["max_steps"],
        accelerator=trainer_cfg["accelerator"],
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision=trainer_cfg["precision"]),
    )

    logger = nl.NeMoLogger(log_dir=logger_cfg["log_dir"])

    data = llm.MockDataModule(
        seq_length=data_cfg["seq_length"],
        global_batch_size=data_cfg["global_batch_size"],
    )

    llm.train(
        model=modelClasss,
        data=data,
        trainer=trainer,
        log=logger,
        tokenizer="data",
        optim=optimizer,
    )

    print("Training completed successfully!")
