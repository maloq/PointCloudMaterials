from __future__ import annotations

"""Utility helpers for merging evaluation and training configs."""

import os
from typing import Any, Dict
from omegaconf import OmegaConf, DictConfig


def load_training_config(checkpoint_path: str) -> DictConfig:
    """Load the Hydra config stored alongside *checkpoint_path*.

    Expects the typical ``.hydra/config.yaml`` structure.
    """

    ckpt_dir = os.path.dirname(checkpoint_path)
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    return OmegaConf.load(cfg_path)


def merge_and_save(train_cfg: DictConfig, eval_cfg: DictConfig, metrics: Dict[str, Any], *, checkpoint_path: str) -> str:
    """Merge configs and persist results next to the checkpoint.

    Returns path to the saved file.
    """

    merged = OmegaConf.merge(train_cfg, eval_cfg)
    merged.eval_results = metrics
    out_path = os.path.join(os.path.dirname(checkpoint_path), "eval_results.yaml")
    OmegaConf.save(merged, out_path)
    return out_path
