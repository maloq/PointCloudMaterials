from __future__ import annotations

import os
from contextlib import contextmanager
import sys

import torch

sys.path.append(os.getcwd())

from src.models.autoencoders.encoders.vn_encoders import VNRevnetBackboneEncoder


def _random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mat = torch.randn(3, 3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(mat)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _random_point_cloud(
    batch_size: int, num_points: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    points = torch.randn(batch_size, num_points, 3, device=device, dtype=dtype)
    points = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scales = torch.rand(batch_size, 1, 1, device=device, dtype=dtype) * 0.5 + 0.5
    return points * scales


@contextmanager
def _torch_seed(seed: int, *, use_cuda: bool):
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if use_cuda else None
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _encode_latents(encoder: VNRevnetBackboneEncoder, pc: torch.Tensor, seed: int):
    with _torch_seed(seed, use_cuda=pc.is_cuda), torch.inference_mode():
        return encoder(pc)


def _summarize_errors(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    vals = torch.tensor(errors, dtype=torch.float32)
    return {
        "mean": float(vals.mean().item()),
        "p95": float(torch.quantile(vals, 0.95).item()),
        "max": float(vals.max().item()),
    }


def test_vn_revnet_backbone_equivariance():
    device = torch.device("cpu")
    dtype = torch.float32

    # Defaults mirror configs/eq_ae_vn.yaml to match the training pipeline encoder settings.
    latent_size = int(os.getenv("VN_REVNET_LATENT_SIZE", "720"))
    encoder = VNRevnetBackboneEncoder(latent_size=latent_size, sa_npoints=(128, 32)).to(device)
    encoder.eval()

    num_trials = int(os.getenv("VN_REVNET_EQ_TRIALS", "2"))
    num_rots = int(os.getenv("VN_REVNET_EQ_ROTS", "4"))
    base_seed = int(os.getenv("VN_REVNET_EQ_SEED", "1337"))

    eq_errors = []
    inv_errors = []

    batch_size = 2
    num_points = int(os.getenv("VN_REVNET_NUM_POINTS", "512"))

    for trial in range(num_trials):
        pc = _random_point_cloud(batch_size, num_points, device, dtype)
        for rot_idx in range(num_rots):
            seed = base_seed + trial * 100 + rot_idx
            rot = _random_rotation_matrix(device, dtype)
            pc_rot = pc @ rot.T

            inv_z, eq_z, _ = _encode_latents(encoder, pc, seed)
            inv_z_rot, eq_z_rot, _ = _encode_latents(encoder, pc_rot, seed)

            assert inv_z.shape == (batch_size, latent_size)
            assert eq_z is not None
            assert eq_z.shape == (batch_size, latent_size, 3)
            assert eq_z_rot is not None
            assert eq_z_rot.shape == (batch_size, latent_size, 3)

            expected_eq = eq_z @ rot.T
            eq_diff = torch.linalg.norm(eq_z_rot - expected_eq, dim=-1)
            eq_norm = torch.linalg.norm(expected_eq, dim=-1).clamp_min(1e-6)
            eq_rel = (eq_diff / eq_norm).mean(dim=1)
            eq_errors.extend(eq_rel.detach().cpu().tolist())

            inv_diff = torch.linalg.norm(inv_z - inv_z_rot, dim=-1)
            inv_norm = torch.linalg.norm(inv_z, dim=-1).clamp_min(1e-6)
            inv_rel = inv_diff / inv_norm
            inv_errors.extend(inv_rel.detach().cpu().tolist())

    eq_stats = _summarize_errors(eq_errors)
    inv_stats = _summarize_errors(inv_errors)

    eq_mean_max = float(os.getenv("VN_REVNET_EQ_MEAN_MAX", "1e-3"))
    eq_p95_max = float(os.getenv("VN_REVNET_EQ_P95_MAX", "5e-3"))
    inv_mean_max = float(os.getenv("VN_REVNET_INV_MEAN_MAX", "1e-4"))
    inv_p95_max = float(os.getenv("VN_REVNET_INV_P95_MAX", "5e-4"))

    print("Equivariant latent relative error:", eq_stats)
    print("Invariant latent relative error:", inv_stats)

    assert eq_stats["mean"] <= eq_mean_max, f"eq mean {eq_stats['mean']} > {eq_mean_max}"
    assert eq_stats["p95"] <= eq_p95_max, f"eq p95 {eq_stats['p95']} > {eq_p95_max}"
    assert inv_stats["mean"] <= inv_mean_max, f"inv mean {inv_stats['mean']} > {inv_mean_max}"
    assert inv_stats["p95"] <= inv_p95_max, f"inv p95 {inv_stats['p95']} > {inv_p95_max}"


if __name__ == "__main__":
    test_vn_revnet_backbone_equivariance()
