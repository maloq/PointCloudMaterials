import os
import sys
import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from src.training_methods.equivariant_autoencoder.eq_ae_module import EquivariantAutoencoder
from src.training_methods.spd.train_spd import train_model

torch.set_float32_matmul_precision('high')


def train(cfg: DictConfig):
    """Equivariant Autoencoder-specific training function."""
    train_model(cfg, EquivariantAutoencoder)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_vn_equivariant.yaml')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
