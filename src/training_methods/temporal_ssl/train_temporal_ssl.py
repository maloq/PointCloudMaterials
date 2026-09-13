import os
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.training_methods.train_entrypoint import train as train_registered_method  # noqa: E402


torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig):
    return train_registered_method(cfg, method_name="temporal_ssl", run_analysis=False)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name=None,
)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("Select --config-dir DIR --config-name NAME; retired temporal recipes are documented in configs/README.md.")
    train(cfg)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit("Select --config-dir DIR --config-name NAME; see configs/README.md.")
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/runs/training/${experiment_name}/${now:%Y%m%d_%H%M%S}")
    main()


__all__ = ["main", "train"]
