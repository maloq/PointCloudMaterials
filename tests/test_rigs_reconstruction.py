from __future__ import annotations

import torch

from src.models.encoders.rigs_encoder import (
    RIGSNNEncoder,
    compute_rigs_geometry,
    compute_sparse_rigs_graph,
    orthogonal_procrustes_rmsd,
    reconstruct_points_from_distance_matrix,
)


def _make_point_cloud_batch() -> torch.Tensor:
    torch.manual_seed(7)
    points = torch.randn(2, 8, 3, dtype=torch.float32)
    points[:, 0] = 0.0
    return points


def _signed_volume(points: torch.Tensor) -> torch.Tensor:
    if points.dim() != 2 or points.shape != (4, 3):
        raise ValueError(f"_signed_volume expects shape (4, 3), got {tuple(points.shape)}.")
    a = points[1] - points[0]
    b = points[2] - points[0]
    c = points[3] - points[0]
    return torch.det(torch.stack([a, b, c], dim=1)) / 6.0


def test_rigs_round_trip_preserves_pairwise_geometry_up_to_orthogonal_transform():
    points = _make_point_cloud_batch()
    geometry = compute_rigs_geometry(points, local_k=4)
    reconstructed = reconstruct_points_from_distance_matrix(geometry.distances)
    reconstructed_geometry = compute_rigs_geometry(reconstructed, local_k=4)

    assert torch.allclose(
        reconstructed_geometry.distances,
        geometry.distances,
        atol=1e-5,
        rtol=1e-5,
    )

    rmsd = orthogonal_procrustes_rmsd(
        reconstructed,
        points,
        allow_reflection=True,
    )
    assert torch.all(rmsd < 1e-4), f"Expected near-lossless reverse RIGS reconstruction, got rmsd={rmsd}."


def test_rigs_reverse_loses_chirality_for_mirror_images():
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.2],
            [0.2, 0.8, 0.3],
            [0.3, 0.4, 1.1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    mirror = base.clone()
    mirror[..., 0] *= -1.0

    base_geometry = compute_rigs_geometry(base, local_k=3)
    mirror_geometry = compute_rigs_geometry(mirror, local_k=3)

    assert torch.allclose(base_geometry.radii, mirror_geometry.radii, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_geometry.distances, mirror_geometry.distances, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base_geometry.cosines, mirror_geometry.cosines, atol=1e-6, rtol=1e-6)
    assert torch.equal(base_geometry.angular_valid, mirror_geometry.angular_valid)

    signed_base = _signed_volume(base[0])
    signed_mirror = _signed_volume(mirror[0])
    assert signed_base * signed_mirror < 0.0, "The mirror example must flip chirality."


def test_rigs_nn_encoder_is_rotation_invariant():
    points = _make_point_cloud_batch()
    angle = torch.tensor(0.73, dtype=torch.float32)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.tensor(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rotated = points @ rotation.T

    model = RIGSNNEncoder(
        latent_size=64,
        node_dim=32,
        depth=3,
        num_radial=12,
        num_density=8,
        local_k=4,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        z_ref, _, _ = model(points)
        z_rot, _, _ = model(rotated)

    assert torch.allclose(z_ref, z_rot, atol=1e-5, rtol=1e-5)


def test_rigs_nn_encoder_matches_precomputed_sparse_graph_input():
    points = _make_point_cloud_batch()
    model = RIGSNNEncoder(
        latent_size=64,
        node_dim=32,
        depth=2,
        num_radial=12,
        num_density=8,
        local_k=4,
        dropout=0.0,
    )
    model.eval()

    graph = compute_sparse_rigs_graph(points, local_k=4)
    graph_dict = {
        "radii": graph.radii,
        "edge_index": graph.edge_index,
        "edge_distance": graph.edge_distance,
        "edge_cosine": graph.edge_cosine,
    }

    with torch.no_grad():
        z_points, _, _ = model(points)
        z_graph, _, _ = model(graph_dict)

    assert torch.allclose(z_points, z_graph, atol=1e-6, rtol=1e-6)
