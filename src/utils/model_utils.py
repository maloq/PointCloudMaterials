from pathlib import Path
from contextlib import nullcontext
from typing import TypeVar

import torch


ModelT = TypeVar("ModelT", bound=torch.nn.Module)


def resolve_config_path(checkpoint_path: str) -> tuple[str, str]:
    """Return the Hydra config stored beside a repository training checkpoint."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    config_path = checkpoint.parent / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "A repository checkpoint must have its training config at "
            f"{config_path}; checkpoint={checkpoint}."
        )
    return str(config_path.parent), config_path.stem


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg,
    *,
    device: str = "cpu",
    module: type[ModelT],
) -> ModelT:
    """Restore a repository Lightning module from its saved state dictionary."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    target = torch.device(device)
    # MACE's accelerated constructor allocates on the current CUDA device.
    # Construct on the destination and stage saved tensors through CPU, avoiding
    # a CUDA:0 -> CUDA:1 copy when a caller selects a non-default device.
    context = torch.cuda.device(target) if target.type == "cuda" else nullcontext()
    with context:
        model = module(cfg)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = payload["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(target)
    model.eval()
    return model
