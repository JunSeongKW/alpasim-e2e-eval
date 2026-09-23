import datetime
import logging
from pathlib import Path
from typing import Tuple
from functools import partial

import torch
# H200 (sm_90): torch 2.0.x's fused scaled-dot-product-attention is broken on
# Hopper — the flash backend has no sm_90 kernel and the mem-efficient backend
# raises "CUDA error: an illegal instruction was encountered" in the ViT
# attention backward. Force the pure-PyTorch math backend there. torch >=2.1
# ships working flash/mem-efficient kernels for sm_90, so leave the fast fused
# backends enabled (the `drivesuprim_flash` env). Set before any model runs;
# this module is re-imported in every DDP rank's process.
_torch_mm = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
if _torch_mm < (2, 1):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
# TF32 on tensor cores: free speedup for fp32 matmuls outside autocast.
torch.set_float32_matmul_precision("high")
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset_ssl import CacheOnlyDataset, DatasetSSL
from navsim.planning.training.agent_lightning_module_ssl import AgentLightningModuleSSL

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[DatasetSSL, DatasetSSL]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name
            for log_name in train_scene_filter.log_names
            if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name
            for log_name in val_scene_filter.log_names
            if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = DatasetSSL(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = DatasetSSL(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cfg=cfg.agent.config,
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def build_dataloaders(train_data, val_data, cfg):
    logger.info("Building Datasets")
    # drop_last=True: without it, the last batch of every epoch is whatever is
    # left over (1..batch_size-1 samples) once the (DDP-sharded) dataset is
    # exhausted. That ragged batch flows into the BEVFormer encoder's
    # deformable attention (spatial_cross_attention.py) as a smaller "batch"
    # dimension than every other step -- if it happens to land above mmcv's
    # im2col_step (64) without being an exact multiple of it, the CUDA kernel
    # asserts "batch(%d) must divide im2col_step(%d)" and the run crashes on
    # the last step of an epoch after training successfully up to that point.
    # See navsim/agents/backbones/bevformer/spatial_cross_attention.py for the
    # matching fix on the kernel-call side; this removes the ragged batch
    # itself so no batch-size configuration can hit that edge case at all.
    # At most one full global batch (batch_size * num_gpus) is skipped per
    # epoch this way -- a small fraction of navtrain's ~103k samples.
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True,
                                  drop_last=True)
    logger.info("Num training samples: %d", len(train_data))
    # A dataset may intentionally be train-only (NuRec has no held-out split).
    # Passing None makes Lightning skip the validation loop entirely instead of
    # constructing a misleading validation set from training logs.
    val_dataloader = (
        DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
        if len(val_data) > 0
        else None
    )
    logger.info("Num validation samples: %d", len(val_data))
    return train_dataloader, val_dataloader


def build_model(config, only_teacher=False):
    model_dict = dict()
    logger.info("Building Teacher Agent")
    # teacher_agent = instantiate(config.agent)
    agent = instantiate(config.agent)
    lightening_module = AgentLightningModuleSSL(
        config.agent.config,
        agent=agent,
    )
    return agent, lightening_module


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg, only_teacher=only_teacher)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    agent, lightning_module = build_model_from_cfg(cfg)

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
                cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    train_dataloader, val_dataloader = build_dataloaders(train_data, val_data, cfg)

    logger.info("Building Trainer")
    if isinstance(agent, TransfuserAgent):
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks())
    else:
        trainer = pl.Trainer(**cfg.trainer.params,
                             callbacks=agent.get_training_callbacks(),
                             strategy=DDPStrategy(static_graph=True,
                                                  timeout=datetime.timedelta(seconds=3600)))

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get('resume_ckpt_path', None)
    )


if __name__ == "__main__":
    main()
