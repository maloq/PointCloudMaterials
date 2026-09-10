from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.data_utils.spatiotemporal_views import local_views, periodic_tree
from src.training_methods.contrastive_learning.vicreg import VICRegLoss


def test_temporal_view_tracks_same_atom_across_periodic_boundary():
    lengths = np.full(3, 10, dtype=np.float32)
    initial = np.array([[9.9, 1, 1], [0.1, 1, 1], [9.9, 1.2, 1], [5, 5, 5]], dtype=np.float32)
    future = np.mod(initial + [0.3, 0, 0], lengths).astype(np.float32)
    a, tree_a = periodic_tree(initial, lengths)
    b, tree_b = periodic_tree(future, lengths)
    local_a = local_views(a, tree_a, lengths, np.array([0]), num_points=3, radius=2)
    local_b = local_views(b, tree_b, lengths, np.array([0]), num_points=3, radius=2)
    # Compare unordered offsets; tied distances may change their neighbor order.
    np.testing.assert_allclose(np.sort(local_a, axis=1), np.sort(local_b, axis=1), atol=1.e-6)
    np.testing.assert_allclose(local_b[0, 0], 0, atol=1.e-6)


def test_float16_boundary_coordinate_is_wrapped_before_neighbor_query():
    points, tree = periodic_tree(np.array([[10, 1, 1], [0.2, 1, 1]], dtype=np.float16), np.full(3, 10, dtype=np.float32))
    assert points.dtype == np.float32 and points[0, 0] == 0
    distance, _ = tree.query(points[0], k=2)
    assert distance[1] == pytest.approx(0.2, abs=1.e-3)


@pytest.mark.parametrize("objective", ["vicreg", "visreg"])
def test_three_view_loss_has_temporal_and_spatial_gradients_and_detects_bad_values(objective):
    with initialize_config_dir(version_base=None, config_dir=str(Path("configs").resolve())):
        cfg = compose(config_name=f"{objective}_geoframe_v2_spatiotemporal_corrected_20260905")
    cfg.vicreg_projector_mode = "identity"
    cfg.vicreg_embed_dim = 8
    loss_fn = VICRegLoss.from_config(cfg, input_dim=8)
    features = tuple(torch.randn(32, 8, requires_grad=True) for _ in range(3))
    rng = torch.get_rng_state()
    loss, metrics, projected = loss_fn.compute_spatiotemporal_loss(features=features, temporal_weight=1)
    assert all(a is b for a,b in zip(projected, features))
    torch.set_rng_state(rng)
    expected = (loss_fn._loss(features[0], features[1])[0] + loss_fn._loss(features[0], features[2])[0]) / 2
    assert loss_fn.objective == objective
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in features)
    assert "encoder_temporal_relative_mse" in metrics
    with pytest.raises(FloatingPointError, match="Non-finite"):
        loss_fn.compute_spatiotemporal_loss(features=(features[0], features[1], features[2] * float("nan")), temporal_weight=1)
