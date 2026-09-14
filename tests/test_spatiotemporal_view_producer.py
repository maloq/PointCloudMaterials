from types import SimpleNamespace

import numpy as np

from src.data.spatiotemporal import local_views, periodic_tree, prepare_branch
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory


def test_configured_views_preserve_identity_scale_and_split(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    positions = rng.uniform(0, 20, (10, 512, 3)).astype(np.float16)
    trajectory = SimpleNamespace(frame_count=10, atom_count=512, positions=positions,
        timesteps=np.arange(10)*100, atom_ids=np.arange(512)+1,
        box_low=np.zeros((10,3)), box_high=np.full((10,3),20.))
    monkeypatch.setattr(TemporalLAMMPSBinaryTrajectory, 'load', lambda path: trajectory)
    source = dict(path='repository-test', material='Ti', snapshot='test', radius=9.5,
        frame_count=10, timestep_stride=100, lags=[1],
        splits=[dict(split='train', anchors=[2], centers=8),
                dict(split='val', anchors=[7], centers=8)])
    shards = prepare_branch((source, tmp_path, 0, 42))
    for shard in shards:
        pairs = np.load(tmp_path/shard['pairs'])
        views = np.load(tmp_path/shard['views'])
        assert views.shape == (8,3,80,3)
        assert views.dtype == np.float16
        rows = pairs[:,0]-1
        assert np.all(rows % 5 != 0) if shard['split']=='train' else np.all(rows % 5 == 0)
        assert np.all(pairs[:,3] == 1)
        for view, frame, centers in [(0, int(pairs[0,2]), rows),
                (1, int(pairs[0,2]), pairs[:,1]-1), (2, int(pairs[0,2])+1, rows)]:
            points, tree = periodic_tree(positions[frame], np.full(3,20.))
            expected = local_views(points, tree, np.full(3,20.), centers,
                                   num_points=80, radius=9.5).astype(np.float16)
            np.testing.assert_array_equal(views[:,view], expected)
