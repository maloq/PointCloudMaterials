import os
# Hack to fix multi-GPU training on this server (NCCL P2P hang)
# os.environ["NCCL_P2P_DISABLE"] = "1"

import sys
import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.training_methods.contrastive_learning.contrastive_module import BarlowTwinsModule
from src.training_methods.spd.train_spd import train_model

torch.set_float32_matmul_precision('high')
logger = setup_logging()


def train(cfg: DictConfig):
    """Barlow Twins contrastive training."""
    return train_model(cfg, BarlowTwinsModule)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='barlow_twins_vn_molecular.yaml')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
