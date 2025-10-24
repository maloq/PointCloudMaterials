import torch
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf
import sys,os
sys.path.append(os.getcwd())

from src.training_methods.spd.spd_module import ShapePoseDisentanglement


def _cfg(n_points: int = 32, latent: int = 16) -> OmegaConf:
    return OmegaConf.create({
        'latent_size': latent,
        'learning_rate': 1e-3,
        'decay_rate': 0.0,
        'enable_swa': False,
        'swa_epoch_start': 0,
        'epochs': 1,
        'scheduler_name': 'OneCycle',
        'scheduler_gamma': 0.5,
        'encoder': {'name': 'PnE_L', 'kwargs': {'latent_size': latent}},
        'decoder': {
            'name': 'FoldingSphere',
            'kwargs': {
                'latent_size': latent,
                'num_points': n_points,
                'dropout_rate': 0.0,
                'n_shells': 1,
                'R1': 1.0,
                'R2': 1.0,
                'hidden_dim': 32,
            },
        },
        'rotation_mode': 'sixd_head',
        'rot_net': {
            'name': 'Rot6DHead',
            'kwargs': {
                'hidden': 32,
                'use_attention': False,
            },
        },
    })


def test_rotation_recovery(tmp_path):
    cfg = _cfg()
    model = ShapePoseDisentanglement(cfg)

    pc = torch.randn(2, 3, cfg.decoder.kwargs.num_points)
    rot = R.from_euler('xyz', [45, 20, 10], degrees=True).as_matrix()
    pc_rot = pc @ torch.tensor(rot, dtype=torch.float32).T

    _, _, rot_pred, _ = model(pc_rot)
    rot_pred = rot_pred.squeeze(0)

    assert rot_pred.shape == (3, 3)
    ident = rot_pred.T @ rot_pred
    assert torch.allclose(ident, torch.eye(3), atol=1e-4)

    error = torch.norm(rot_pred - torch.tensor(rot, dtype=torch.float32))
    print(f'Rotation L2 error: {error.item():.2f}')


