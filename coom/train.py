import os
import torch
import importlib
from omegaconf import OmegaConf
from nemo import lightning as nl
from megatron.core.optimizer import OptimizerConfig
from nemo.collections import llm
import model
import config_classes

main_cfg = OmegaConf.load("configs/main_config.yaml")
prefix = main_cfg.config_path_prefix

model_cfg = OmegaConf.load(os.path.join(prefix, main_cfg.base_model_configuartion_path))
data_cfg = OmegaConf.load(os.path.join(prefix, main_cfg.dataLoader_config_path))
opt_cfg = OmegaConf.load(os.path.join(prefix, main_cfg.optimizer_config_path))
trainer_cfg = OmegaConf.load(os.path.join(prefix, main_cfg.trainer_config_path))
logger_cfg = OmegaConf.load(os.path.join(prefix, main_cfg.logger_config_path))

model_cfg.seq_length = data_cfg.seq_length


ModelClass = getattr(model, main_cfg.base_model)
ConfigClass = getattr(config_classes, main_cfg.base_model_config)

model_config = ConfigClass(**model_cfg)
modelClasss = ModelClass(model_config)

strategy = nl.MegatronStrategy(**trainer_cfg.strategy)

optimizer_config = OptimizerConfig(**opt_cfg)
optimizer = nl.MegatronOptimizerModule(config=optimizer_config)

trainer = nl.Trainer(
    devices=trainer_cfg.devices,
    max_steps=trainer_cfg.max_steps,
    accelerator=trainer_cfg.accelerator,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(precision=trainer_cfg.precision),
)

logger = nl.NeMoLogger(log_dir=logger_cfg.log_dir)

data = llm.MockDataModule(
    seq_length=data_cfg.seq_length,
    global_batch_size=data_cfg.global_batch_size,
)

llm.train(
    model=modelClasss,
    data=data,
    trainer=trainer,
    log=logger,
    tokenizer="data",
    optim=optimizer,
)
