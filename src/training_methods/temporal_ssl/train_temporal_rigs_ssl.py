import os
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.training_methods.temporal_ssl.temporal_rigs_ssl_module import TemporalRIGSSSLModule
from src.training_methods.trainer import train_model


torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig):
    run_test = bool(getattr(cfg, "run_test_after_training", True))
    return train_model(
        cfg,
        TemporalRIGSSSLModule,
        run_test=run_test,
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="temporal_rigs_vicreg_lammps.yaml",
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
