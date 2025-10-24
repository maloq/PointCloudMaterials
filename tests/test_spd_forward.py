from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader
import sys,os
sys.path.append(os.getcwd())
from src.data_utils.data_module import PointCloudDataModule
from src.training_methods.spd.spd_module import ShapePoseDisentanglement


def _compose_test_config(tmp_path: Path):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    overrides = [
        "data=data_synth_baseline",
        "batch_size=2",
        "num_workers=0",
        "max_samples=4",
        "gpu=False",
        "torch_compile=False",
        "data.synthetic.num_environments=8",
        "data.synthetic.regenerate=True",
        f'data.synthetic.cache_path="{(tmp_path / "scene.npz").as_posix()}"',
    ]
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="spd", overrides=overrides, return_hydra_config=False)
    return cfg


def test_spd_forward_pass_from_config(tmp_path):
    cfg = _compose_test_config(tmp_path)
    dm = PointCloudDataModule(cfg)
    dm.setup(stage="fit")

    # Use a lightweight single-process DataLoader for the test batch
    train_loader = DataLoader(
        dm.train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(train_loader))
    points = batch[0]

    model = ShapePoseDisentanglement(cfg)
    model.eval()

    with torch.no_grad():
        inv_z, recon, cano, rot = model(points)

    batch_size = points.shape[0]
    num_points = points.shape[1]

    assert inv_z.shape == (batch_size, int(cfg.latent_size))
    assert recon.shape == (batch_size, num_points, 3)
    assert cano.shape == (batch_size, num_points, 3)
    assert rot.shape[0] == batch_size
    assert rot.shape[-2:] == (3, 3)


if __name__ == "__main__":
    from pathlib import Path
    tmp_path = Path("output/tmp/test_spd_forward_2")
    tmp_path.mkdir(parents=True, exist_ok=True)
    test_spd_forward_pass_from_config(tmp_path=tmp_path)