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
        method_name="vicreg_masked_latent",
        run_analysis=run_analysis,
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="vicreg_masked_latent.yaml",
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()


__all__ = ["main", "run_post_training_analysis_safe", "train"]
