import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import sys,os
from datetime import datetime
sys.path.append(os.getcwd())
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.vis_tools.vis_utils import plot_point_cloud_3d
from src.data_utils.data_load import PointCloudDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset


OmegaConf.register_new_resolver("now", lambda format_string: datetime.now().strftime(format_string))


def create_autoencoder_dataloader(cfg: OmegaConf, file_path, shuffle: bool = False, max_samples = None, return_coords: bool = False) -> DataLoader:
    """Create a dataloader for autoencoder inference from OFF file(s).
    
    Args:
        cfg: Configuration dictionary 
        file_path: Path to the OFF file or list of paths to OFF files
        shuffle: Whether to shuffle the samples
        max_samples: Maximum number of samples to include
    Returns:
        DataLoader containing point cloud samples
    """
    if isinstance(file_path, str):
        file_paths = [file_path]
    elif isinstance(file_path, (list, tuple)):
        file_paths = file_path
    
    dataset = PointCloudDataset(
                 root=cfg.data.data_path,
                 data_files=file_paths,
                 return_coords=return_coords,
                 sample_type='regular',
                 radius=cfg.data.radius,
                 overlap_fraction=cfg.data.overlap_fraction,
                 n_samples=cfg.data.n_samples,
                 num_points=cfg.data.num_points,
                 pre_normalize=True,
                 normalize=True)
    
    print(f"Number of samples in dataset: {len(dataset)}")
    
    if max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
        print(f"Dataset limited to {len(dataset)}")
    
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)


def demo(save_html: str = 'rotation_demo.html') -> None:
    """Run a simple rotation estimation demo and save a visualization."""

    checkpoint_path = 'output/2025-07-23/22-37-50/SPD_FoldingSphereAttn_l64_P80_Sinkhorn_4192_0723-epoch=247-val_loss=0.02.ckpt'
    base_cfg = OmegaConf.load('configs/spd.yaml')
    defaults = base_cfg.pop('defaults', [])

    configs_to_merge = []
    for default in defaults:
        for key, name in default.items():
            default_cfg_path = f"configs/{key}/{name}.yaml"
            loaded_cfg = OmegaConf.load(default_cfg_path)
            configs_to_merge.append(OmegaConf.create({key: loaded_cfg}))

    configs_to_merge.append(base_cfg)
    cfg = OmegaConf.merge(*configs_to_merge)
    OmegaConf.resolve(cfg)
    
    model = ShapePoseDisentanglement(cfg)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['state_dict'])
    model.eval()
    model.to('cuda')
    
    dataloader = create_autoencoder_dataloader(cfg, '240ps.off', max_samples=1)
    
    rotated = next(iter(dataloader))[0].to('cuda')

    recon, _, rot_pred, trans_pred = model(rotated.unsqueeze(0))
    rot_pred = rot_pred.squeeze(0).detach()
    trans_pred = trans_pred.squeeze(0).detach()

    # visualise
    pred_cano = (rot_pred @ (rotated.squeeze(0) - trans_pred).T).T
    fig = plot_point_cloud_3d(rotated.squeeze(0).detach().cpu().numpy(), title='Input')
    fig2 = plot_point_cloud_3d(recon.squeeze(0).detach().cpu().numpy(), title='Reconstructed')
    fig3 = plot_point_cloud_3d(pred_cano.detach().cpu().numpy(), title='Predicted Canonical')
    fig.write_html('input_'+save_html)
    fig2.write_html('reconstructed_'+save_html)
    fig3.write_html('predicted_canonical_'+save_html)


if __name__ == '__main__':
    demo()
