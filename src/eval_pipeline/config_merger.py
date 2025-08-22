from __future__ import annotations

"""Utility helpers for merging evaluation and training configs."""

import os
from typing import Any, Dict, Optional
from omegaconf import OmegaConf, DictConfig
import shutil
import subprocess
import hashlib


def load_training_config(checkpoint_path: str) -> DictConfig:
    """Load a training config associated with *checkpoint_path*.

    - If the checkpoint resides inside an ``eval_results`` run directory, load
      that directory's ``eval_results.yaml`` and strip evaluation-only keys.
    - Otherwise, load the Hydra config at ``.hydra/config.yaml`` next to the checkpoint.
    """

    ckpt_dir = os.path.dirname(checkpoint_path)

    # Case 1: checkpoint was copied into an eval run directory
    # e.g. output/eval_results/<run>/model.ckpt → use <run>/eval_results.yaml
    parts = os.path.normpath(ckpt_dir).split(os.sep)
    eval_yaml_path = os.path.join(ckpt_dir, "eval_results.yaml")
    if "eval_results" in parts and os.path.isfile(eval_yaml_path):
        cfg = OmegaConf.load(eval_yaml_path)
        cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=False)  # type: ignore
        if isinstance(cfg_dict, dict):
            cfg_dict.pop("eval_results", None)  # drop metrics blob
            cfg_dict.pop("prediction", None)    # drop eval-only predictor settings
        return OmegaConf.create(cfg_dict)

    # Case 2: standard Hydra-trained checkpoint
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    return OmegaConf.load(cfg_path)


def get_current_git_commit() -> Optional[str]:
    """Return current Git commit SHA if available, otherwise None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def compute_file_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """Compute SHA-256 hash of a file. Returns hex digest or None on failure."""
    try:
        if not os.path.isfile(file_path):
            return None
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def merge_and_save(out_dir: str, train_cfg: DictConfig, eval_cfg: DictConfig, metrics: Dict[str, Any], *, checkpoint_path: str) -> str:
    """Merge configs and persist results under outputs/eval_results_{date}_{checkpoint_name}/eval_results.yaml.

    Returns path to the saved file.
    """
    merged = OmegaConf.merge(train_cfg, eval_cfg)
    merged.eval_results = metrics
    git_commit = get_current_git_commit()
    if git_commit:
        merged.git_commit = git_commit
    ckpt_sha256 = compute_file_sha256(checkpoint_path)
    if ckpt_sha256:
        merged.checkpoint_sha256 = ckpt_sha256

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
