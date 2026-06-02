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
    config_name="temporal_vicreg_lammps.yaml",
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()


__all__ = ["main", "train"]
