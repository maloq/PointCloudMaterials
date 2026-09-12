import os
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.training_methods.train_entrypoint import (  # noqa: E402
    run_post_training_analysis_safe,
    train as train_registered_method,
)


torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig, run_analysis: bool = True):
    return train_registered_method(
        cfg,
        method_name=getattr(cfg, "training_method", None),
        run_analysis=run_analysis,
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name=None,
)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("Select a training configuration with --config-name NAME; see configs/README.md.")
    train(cfg)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit("Select --config-name NAME; see configs/README.md for available training configurations.")
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${experiment_name}/${now:%Y%m%d-%H%M%S}/technical")
    main()


__all__ = ["main", "run_post_training_analysis_safe", "train"]
