"""Continuity and symmetry requirements of the temporal density pilot."""
import torch
from src.models.encoders.smooth_density import (
    DensityEncoder, SmoothDensity, TensorTransport, motion_matrix,
)


def cloud():
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(3, 48, 3, generator=generator, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True) * torch.linspace(.25, 1.2, 48)[None, :, None]
    r, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    r[:, -1] *= torch.linalg.det(r)
    return x, r


def test_density_rotation_permutation_and_outside_support():
    x, r = cloud()
    density = SmoothDensity().double()
    p = density.power(density(x))
    torch.testing.assert_close(p, density.power(density(x @ r.T)), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(p, density.power(density(x.flip(1))), rtol=1e-10, atol=1e-10)
    # Only the first 38 atoms are within support on this explicitly made cloud.
    keep = x[0].norm(dim=-1) < 1
    torch.testing.assert_close(density(x), density(x[:, keep]), rtol=1e-10, atol=1e-10)


def test_cutoff_crossing_difference_vanishes_cubically():
    density = SmoothDensity().double()
    distances = []
    for delta in (.01, .001, .0001):
        pair = torch.tensor([[[1-delta, 0., 0.]], [[1+delta, 0., 0.]]], dtype=torch.float64)
        values = density(pair)
        distances.append((values[0]-values[1]).norm())
    assert distances[1] < distances[0] * .01
    assert distances[2] < distances[1] * .01


def test_mace_product_encoder_is_invariant_and_has_finite_gradient():
    x, r = cloud()
    model = DensityEncoder(SmoothDensity(), "mace_product", channels=8, hidden_dim=32).double()
    torch.testing.assert_close(model(x), model(x @ r.T), rtol=1e-9, atol=1e-9)
    x.requires_grad_()
    model(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.mix.weight.grad).all()
    assert model.mix.weight.grad.abs().sum() > 0


def test_tensor_transport_matches_rotated_density_through_ell6():
    x, r = cloud()
    density = SmoothDensity().double()
    transport = TensorTransport().double()
    predicted = transport(density(x), transport.matrices(r))
    torch.testing.assert_close(predicted, density(x @ r.T), rtol=1e-10, atol=1e-10)


def test_smooth_transport_equivariance_and_rank_crossing():
    x, r = cloud()
    c = x.transpose(-1, -2) @ x
    u = r.T
    transformed = r @ c @ u.T
    torch.testing.assert_close(motion_matrix(transformed, "smooth", .1),
                               r @ motion_matrix(c, "smooth", .1) @ u.T, rtol=1e-10, atol=1e-10)
    zero = torch.zeros(2, 3, 3, dtype=torch.float64)
    assert torch.equal(motion_matrix(zero, "smooth", .1), zero)
    differences = []
    for delta in (.01, .001, .0001):
        pair = torch.diag_embed(torch.tensor([[1., .5, -delta], [1., .5, delta]], dtype=torch.float64))
        value = motion_matrix(pair, "smooth", .1)
        differences.append((value[0]-value[1]).norm())
    assert differences[1] < differences[0]*.11
    assert differences[2] < differences[1]*.11


def test_ptm_assay_keeps_neighbor_particles():
    import numpy as np
    from ase.build import bulk
    from scipy.spatial import cKDTree
    from experiments.smooth_temporal_encoder_20260905.prepare import ptm_labels
    points = bulk("Al", "fcc", a=4.05, cubic=True).repeat((9, 9, 9)).positions
    center = points[len(points)//2]
    _, ids = cKDTree(points).query(center, k=193)
    offsets = ((points[ids[1:]]-center)/6.9)[None]
    assert np.array_equal(ptm_labels(offsets, .15), [1])


def test_learned_transport_state_preserves_rotation_symmetry():
    from src.temporal_vamp.smooth_state import SmoothTemporalState
    x, r = cloud()
    encoder = DensityEncoder(SmoothDensity(), "mace_product", channels=8, hidden_dim=32).double()
    model = SmoothTemporalState(encoder, torch.zeros(128, dtype=torch.float64),
                                torch.ones(128, dtype=torch.float64), mode="gated_smooth").double().eval()
    density, transport = encoder.density, TensorTransport().double()
    q = density(x)
    qr = density(x @ r.T)
    q = torch.stack((q, q*.99, q*1.01), 1)
    qr = torch.stack((qr, qr*.99, qr*1.01), 1)
    raw = encoder.forward_moments(q.flatten(0, 1)).reshape(3, 3, 128)
    rawr = encoder.forward_moments(qr.flatten(0, 1)).reshape(3, 3, 128)
    c = torch.diag(torch.tensor([1., .5, .1], dtype=torch.float64))
    t = motion_matrix(c, "smooth", .05)
    tr = motion_matrix(r @ c @ r.T, "smooth", .05)
    matrices = [d.expand(3, 3, *d.shape) for d in transport.matrices(t)]
    matricesr = [d.expand(3, 3, *d.shape) for d in transport.matrices(tr)]
    material = torch.tensor([0, 1, 2])
    left = model(q, raw, matrices, material)
    right = model(qr, rawr, matricesr, material)
    for a,b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8)
    left[0].square().mean().backward()
    assert torch.isfinite(model.gate[-1].weight.grad).all()
    assert model.gate[-1].weight.grad.abs().sum() > 0
