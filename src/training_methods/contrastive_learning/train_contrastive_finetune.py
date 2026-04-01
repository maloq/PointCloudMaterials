import os
import sys

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd())

from src.training_methods.contrastive_learning.train_contrastive import train
from src.utils.logging_config import setup_logging


torch.set_float32_matmul_precision("high")
logger = setup_logging()

_MISSING = object()

_COMMON_COMPATIBILITY_FIELDS = (
    "model_type",
    "latent_size",
    "encoder.name",
    "encoder.kwargs.latent_size",
    "encoder.kwargs.num_group",
    "encoder.kwargs.group_size",
    "encoder.kwargs.encoder_dims",
    "encoder.kwargs.trans_dim",
    "encoder.kwargs.depth",
    "encoder.kwargs.predictor_depth",
    "encoder.kwargs.num_heads",
    "encoder.kwargs.mlp_ratio",
    "data.num_points",
    "data.model_points",
    "barlow_enabled",
    "vicreg_enabled",
    "wmse_enabled",
    "pointcontrast_enabled",
)

_OBJECTIVE_COMPATIBILITY_FIELDS = {
    "barlow_enabled": (
        "barlow_embed_dim",
    ),
    "vicreg_enabled": (
        "vicreg_embed_dim",
    ),
    "wmse_enabled": (
        "wmse_embed_dim",
        "wmse_whitening_eps",
        "wmse_whitening_iters",
        "wmse_whitening_size",
        "wmse_normalize_embeddings",
    ),
    "pointcontrast_enabled": (
        "pointcontrast_embed_dim",
    ),
}


def _has_explicit_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "false", "none", "null"}
    return bool(value)


def _resolve_checkpoint_path(raw_value, *, field_name: str) -> str | None:
    if not _has_explicit_value(raw_value):
        return None

    checkpoint_path = os.path.expanduser(str(raw_value).strip())
    if not os.path.isabs(checkpoint_path):
        base_dir = os.getcwd()
        try:
            base_dir = HydraConfig.get().runtime.cwd
        except Exception as exc:
            logger.warning(
                "Hydra runtime cwd is unavailable; resolving %s relative to '%s'. Error: %s",
                field_name,
                base_dir,
                exc,
            )
        checkpoint_path = os.path.join(base_dir, checkpoint_path)

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{field_name} points to a missing checkpoint: {checkpoint_path}"
        )
    return checkpoint_path


def _to_resolved_dict(cfg_like, *, context: str) -> dict:
    if OmegaConf.is_config(cfg_like):
        resolved = OmegaConf.to_container(cfg_like, resolve=True)
    elif isinstance(cfg_like, dict):
        resolved = cfg_like
    else:
        raise TypeError(
            f"{context} must be an OmegaConf config or dict, got {type(cfg_like)}"
        )

    if not isinstance(resolved, dict):
        raise TypeError(
            f"{context} must resolve to a dict, got {type(resolved)}"
        )
    return resolved


def _nested_get(mapping: dict, path: str):
    value = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _load_checkpoint_hparams(checkpoint_path: str) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint '{checkpoint_path}' must load to a dict, got {type(checkpoint)}"
        )
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain a 'state_dict'."
        )
    if "hyper_parameters" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain Lightning 'hyper_parameters'. "
            "Use a checkpoint produced by the contrastive trainer."
        )

    return _to_resolved_dict(
        checkpoint["hyper_parameters"],
        context=f"checkpoint hyper_parameters from {checkpoint_path}",
    )


def _validate_checkpoint_compatibility(cfg: DictConfig, checkpoint_path: str) -> None:
    current_cfg = _to_resolved_dict(cfg, context="current finetune config")
    checkpoint_cfg = _load_checkpoint_hparams(checkpoint_path)

    mismatches: list[str] = []

    for field_path in _COMMON_COMPATIBILITY_FIELDS:
        current_value = _nested_get(current_cfg, field_path)
        checkpoint_value = _nested_get(checkpoint_cfg, field_path)
        if current_value is _MISSING and checkpoint_value is _MISSING:
            continue
        if current_value != checkpoint_value:
            mismatches.append(
                f"{field_path}: current={current_value!r}, checkpoint={checkpoint_value!r}"
            )

    for objective_flag, field_paths in _OBJECTIVE_COMPATIBILITY_FIELDS.items():
        current_enabled = _nested_get(current_cfg, objective_flag)
        checkpoint_enabled = _nested_get(checkpoint_cfg, objective_flag)
        if not _has_explicit_value(current_enabled) and not _has_explicit_value(checkpoint_enabled):
            continue
        for field_path in field_paths:
            current_value = _nested_get(current_cfg, field_path)
            checkpoint_value = _nested_get(checkpoint_cfg, field_path)
            if current_value is _MISSING and checkpoint_value is _MISSING:
                continue
            if current_value != checkpoint_value:
                mismatches.append(
                    f"{field_path}: current={current_value!r}, checkpoint={checkpoint_value!r}"
                )

    if mismatches:
        mismatch_summary = "\n".join(f"- {entry}" for entry in mismatches)
        raise ValueError(
            "Finetune config is incompatible with the selected checkpoint. "
            "Align the architecture/objective fields below or use a matching checkpoint:\n"
            f"{mismatch_summary}"
        )


def _validate_finetune_cfg(cfg: DictConfig) -> str:
    if _has_explicit_value(getattr(cfg, "load_supervised_checkpoint", None)):
        raise ValueError(
            "Contrastive finetuning does not use 'load_supervised_checkpoint'. "
            "Set it to false/null and use 'init_from_checkpoint' instead."
        )

    init_checkpoint = _resolve_checkpoint_path(
        getattr(cfg, "init_from_checkpoint", None),
        field_name="init_from_checkpoint",
    )
    resume_checkpoint = _resolve_checkpoint_path(
        getattr(cfg, "resume_from_checkpoint", None),
        field_name="resume_from_checkpoint",
    )

    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError(
            "Finetune entrypoint received both init_from_checkpoint and resume_from_checkpoint. "
            "Use init_from_checkpoint for a fresh finetune run, or resume_from_checkpoint to "
            "continue an existing finetune run, but not both."
        )

    selected_checkpoint = init_checkpoint or resume_checkpoint
    if selected_checkpoint is None:
        raise ValueError(
            "Finetune entrypoint requires a checkpoint. Set init_from_checkpoint to start "
            "from pretrained weights, or resume_from_checkpoint to continue an existing "
            "finetune run."
        )

    _validate_checkpoint_compatibility(cfg, selected_checkpoint)
    logger.print(f"Validated finetune checkpoint compatibility: {selected_checkpoint}")
    return selected_checkpoint


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="vicreg_vn_finetune.yaml",
)
def main(cfg: DictConfig):
    _validate_finetune_cfg(cfg)
    train(cfg)


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
