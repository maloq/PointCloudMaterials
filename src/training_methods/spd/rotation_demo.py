import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from .spd_module import ShapePoseDisentanglement
from src.eval_tools.vis_utils import plot_point_cloud_3d


def _default_cfg(num_points: int = 64, latent: int = 32) -> OmegaConf:
    """Return minimal config for ShapePoseDisentanglement."""
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
                'num_points': num_points,
                'dropout_rate': 0.0,
                'n_shells': 1,
                'R1': 1.0,
                'R2': 1.0,
                'hidden_dim': 32,
            },
        },
    })


def demo(save_html: str = 'rotation_demo.html') -> None:
    """Run a simple rotation estimation demo and save a visualization."""
    cfg = _default_cfg()
    model = ShapePoseDisentanglement(cfg)

    # generate canonical point cloud from random latent vector
    z = torch.randn(1, cfg.latent_size)
    cano = model.decoder(z).squeeze(0)
    if cano.shape[0] == 3:
        cano = cano.T

    rot_gt = R.from_euler('xyz', [20, 10, 30], degrees=True).as_matrix()
    trans_gt = torch.tensor([0.2, -0.1, 0.05])

    rotated = (cano @ torch.tensor(rot_gt, dtype=torch.float32).T) + trans_gt

    recon, _, rot_pred, trans_pred = model(rotated.unsqueeze(0))
    rot_pred = rot_pred.squeeze(0).detach()
    trans_pred = trans_pred.squeeze(0).detach()

    print('Ground truth rotation:\n', rot_gt)
    print('Predicted rotation:\n', rot_pred.numpy())

    # visualise
    pred_cano = (rot_pred @ (rotated - trans_pred).T).T
    fig = plot_point_cloud_3d(cano.numpy(), title='Canonical')
    fig2 = plot_point_cloud_3d(rotated.numpy(), title='Rotated Input')
    fig3 = plot_point_cloud_3d(pred_cano.numpy(), title='Predicted Canonical')
    fig.write_html('canonical_'+save_html)
    fig2.write_html('rotated_'+save_html)
    fig3.write_html('predicted_'+save_html)


if __name__ == '__main__':
    demo()
