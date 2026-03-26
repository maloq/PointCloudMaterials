import os
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.training_methods.ri_mae.ri_mae_module import RIMAEModule
from src.training_methods.trainer import train_model


torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig):
    """Train RI-MAE with the shared Lightning training pipeline."""
    return train_model(cfg, RIMAEModule)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="ri_mae_molecular",
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
