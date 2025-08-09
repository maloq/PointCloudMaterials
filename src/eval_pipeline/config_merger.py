from __future__ import annotations

"""Utility helpers for merging evaluation and training configs."""

import os
from typing import Any, Dict
from omegaconf import OmegaConf, DictConfig
from datetime import datetime
import shutil


def load_training_config(checkpoint_path: str) -> DictConfig:
    """Load the Hydra config stored alongside *checkpoint_path*.

    Expects the typical ``.hydra/config.yaml`` structure.
    """

    ckpt_dir = os.path.dirname(checkpoint_path)
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    return OmegaConf.load(cfg_path)


def merge_and_save(out_dir: str, train_cfg: DictConfig, eval_cfg: DictConfig, metrics: Dict[str, Any], *, checkpoint_path: str) -> str:
    """Merge configs and persist results under outputs/eval_results_{date}_{checkpoint_name}/eval_results.yaml.

    Returns path to the saved file.
    """
    merged = OmegaConf.merge(train_cfg, eval_cfg)
    merged.eval_results = metrics

    os.makedirs(out_dir, exist_ok=True)

    # Copy the checkpoint into the output directory
    dest_ckpt_path = os.path.join(out_dir, os.path.basename(checkpoint_path))
    try:
        shutil.copy2(checkpoint_path, dest_ckpt_path)
        print(f"Copied checkpoint to {dest_ckpt_path}")
    except Exception as e:
        print(f"Warning: failed to copy checkpoint: {e}")

    out_path = os.path.join(out_dir, "eval_results.yaml")
    OmegaConf.save(merged, out_path)

    print(f"Saved eval results to {out_path}")
    return out_path
