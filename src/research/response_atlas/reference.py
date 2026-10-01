"""Reference core for a derivative-augmented conditional-future atlas.

This is a tested small-scale prototype, not an integration into PointCloudMaterials.
The simulator and future-feature definition are fixed. No MACE weights are supplied.
All matrix conventions are explicit: q[D], basis[D,R], values[B,M], responses[B,M,R].
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence
import math
import torch
from torch import Tensor, nn


class Oracle(Protocol):
    def query(self, q: Tensor, basis: Tensor, seeds: Sequence[int]) -> "ShotBundle": ...


@dataclass
class ShotBundle:
    q: Tensor
    basis: Tensor
    values: Tensor
    responses: Tensor
    seeds: tuple[int, ...]
    force_calls: int = 0
    hvp_calls: int = 0

    def validate(self) -> None:
        if self.q.ndim != 1 or self.basis.ndim != 2:
            raise ValueError("Expected q[D] and basis[D,R].")
        d, r = self.basis.shape
        if d != self.q.numel() or self.values.ndim != 2:
            raise ValueError("Input or value dimensions do not match.")
        b, m = self.values.shape
        if self.responses.shape != (b, m, r) or b != len(self.seeds):
            raise ValueError("Expected responses[B,M,R] and one seed per branch.")
        if b == 0 or len(set(self.seeds)) != b:
            raise ValueError("Use nonempty, unique branch seeds (independence is a protocol assumption).")
        for x in (self.q, self.basis, self.values, self.responses):
            if not torch.isfinite(x).all():
                raise ValueError("Non-finite oracle data; do not silently clip or omit it.")
            if x.device != self.q.device or x.dtype != self.q.dtype:
                raise ValueError("All bundle tensors must have the same dtype and device.")


def random_basis(d: int, r: int, *, seed: int, dtype=torch.float64,
                 device: str | torch.device = "cpu", allowed: Tensor | None = None) -> Tensor:
    """Orthonormal basis, optionally inside columns of allowed[D,K].

    Build allowed from permitted perturbations before calling (e.g. no translation).
    Coordinates must already use the chosen physical/nondimensional metric.
    """
    k = d if allowed is None else allowed.shape[1]
    if not 1 <= r <= k:
        raise ValueError("Require 1 <= r <= dimension of allowed subspace.")
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(k, r, generator=gen, dtype=dtype, device=device)
    if allowed is not None:
        if allowed.shape[0] != d:
            raise ValueError("allowed must have D rows.")
        x = allowed @ x
    if torch.linalg.matrix_rank(x) < r:
        raise ValueError("Requested perturbation directions are linearly dependent.")
    return torch.linalg.qr(x, mode="reduced").Q


def response_columns(fn: Callable[[Tensor], Tensor], q: Tensor, basis: Tensor,
                     *, create_graph: bool = False) -> tuple[Tensor, Tensor]:
    """Function value and J_fn(q) @ basis, without constructing a full Jacobian.

    create_graph=True is required when derivatives enter the student training loss.
    The reference uses autograd.functional.jvp for compatibility, not peak speed.
    """
    if q.ndim != 1 or basis.ndim != 2 or basis.shape[0] != q.numel():
        raise ValueError("Expected q[D] and basis[D,R].")
    y = fn(q)
    if y.ndim != 1:
        raise ValueError("fn must return a feature vector [M].")
    cols = [torch.autograd.functional.jvp(fn, q, basis[:, j],
                                         create_graph=create_graph)[1]
            for j in range(basis.shape[1])]
    return y, (torch.stack(cols, dim=-1) if cols else y.new_empty(y.numel(), 0))


def unbiased_gram(samples: Tensor) -> Tensor:
    """Unbiased estimate of (E H)^T(E H), H[B,M,R], conditional on a fixed query.

    Distinct rows must represent independent branches. It is intentionally not
    projected to the PSD cone: negative finite-sample eigenvalues are legitimate.
    """
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError("Need independent samples[B,M,R], B >= 2.")
    if not torch.isfinite(samples).all():
        raise ValueError("Non-finite derivative samples.")
    b = samples.shape[0]
    s = samples.sum(dim=0)
    diagonal = torch.einsum("bmr,bms->rs", samples, samples)
    out = (s.T @ s - diagonal) / (b * (b - 1))
    return (out + out.T) / 2


def training_loss(model: nn.Module, bundle: ShotBundle, *, derivative_weight: float = 1.0,
                  value_scale: float = 1.0, derivative_scale: float = 1.0) -> tuple[Tensor, dict]:
    """Standard value + directional Sobolev loss against branch means.

    No U-statistic is needed for vector derivative regression. Branch-mean and
    branchwise squared losses have the same gradient for a fixed bundle.
    Use train-only, frozen scales; weight zero supplies the value-only baseline.
    """
    bundle.validate()
    if derivative_weight < 0 or value_scale <= 0 or derivative_scale <= 0:
        raise ValueError("Invalid loss weights/scales.")
    if derivative_weight > 0 and bundle.basis.shape[1] == 0:
        raise ValueError("Derivative training requires at least one direction.")
    if derivative_weight:
        pred, jac = response_columns(model, bundle.q, bundle.basis, create_graph=True)
    else:
        pred = model(bundle.q)
        jac = pred.new_empty(pred.numel(), 0)
    lv = ((pred - bundle.values.detach().mean(dim=0)) / value_scale).square().mean()
    ld = pred.new_zeros(())
    if derivative_weight:
        ld = ((jac - bundle.responses.detach().mean(dim=0)) / derivative_scale).square().mean()
    loss = lv + derivative_weight * ld
    return loss, {"value_mse": float(lv.detach()), "response_mse": float(ld.detach())}


@dataclass
class Proposal:
    direction: Tensor
    coefficient: Tensor
    estimated_error: float
    predicted_response_norm: float


def propose_direction(model: nn.Module, discovery: ShotBundle) -> Proposal:
    """Largest derivative ERROR in a small permitted subspace.

    This is a candidate, not a verified physical distinction. Discovery seeds must
    be independent of the fitted model; verification needs another seed namespace.
    The score measures response mismatch, not necessarily a false invariance.
    """
    discovery.validate()
    if discovery.basis.shape[1] == 0:
        raise ValueError("No candidate directions.")
    _, predicted = response_columns(model, discovery.q, discovery.basis)
    predicted = predicted.detach()
    errors = discovery.responses - predicted.unsqueeze(0)
    gram = unbiased_gram(errors)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    a = eigenvectors[:, -1]
    return Proposal(discovery.basis @ a, a, float(eigenvalues[-1]),
                    float(torch.linalg.vector_norm(predicted @ a)))


def verify_proposal(model: nn.Module, oracle: Oracle, q: Tensor, proposal: Proposal,
                    epsilon: float, seeds: Sequence[int], *, discovery_seeds: Sequence[int],
                    admissible: Callable[[Tensor], bool] | None = None) -> dict:
    """Fresh-shot, finite-change check with common random numbers across +/-.

    Returns an unbiased squared finite-difference response-error estimate for the
    selected fixed candidate. No confidence guarantee or significance test is implied.
    No automatic accept/reject is performed; choose thresholds on development data.
    """
    if epsilon <= 0 or len(seeds) < 2:
        raise ValueError("Require epsilon > 0 and at least two verification branches.")
    if set(seeds) & set(discovery_seeds):
        raise ValueError("Discovery and verification seeds must be disjoint.")
    qp, qm = q + epsilon * proposal.direction, q - epsilon * proposal.direction
    if admissible is not None and (not admissible(qp) or not admissible(qm)):
        raise ValueError("Proposed configuration fails the physical admissibility policy.")
    empty_basis = q.new_empty(q.numel(), 0)
    plus = oracle.query(qp, empty_basis, seeds)
    minus = oracle.query(qm, empty_basis, seeds)
    measured = (plus.values - minus.values) / (2 * epsilon)
    with torch.no_grad():
        predicted = (model(qp) - model(qm)) / (2 * epsilon)
    error = measured - predicted.unsqueeze(0)
    score = unbiased_gram(error.unsqueeze(-1))[0, 0]
    signal = unbiased_gram(measured.unsqueeze(-1))[0, 0]
    return {"squared_response_error": float(score),
            "squared_simulator_response": float(signal),
            "predicted_response_norm": float(predicted.norm()),
            "epsilon": epsilon, "verification_seeds": list(seeds),
            "force_calls": plus.force_calls + minus.force_calls,
            "hvp_calls": plus.hvp_calls + minus.hvp_calls}


class AutogradOracle:
    """Small-system oracle: branch_feature(q, seed) returns fixed features [M]."""
    def __init__(self, branch_feature: Callable[[Tensor, int], Tensor]):
        self.branch_feature = branch_feature

    def query(self, q: Tensor, basis: Tensor, seeds: Sequence[int]) -> ShotBundle:
        values, responses = [], []
        for seed in seeds:
            # The function MUST deterministically regenerate exactly this branch's noise.
            fn = lambda x, seed=seed: self.branch_feature(x, int(seed))
            val, res = response_columns(fn, q, basis)
            values.append(val.detach()); responses.append(res.detach())
        out = ShotBundle(q.detach().clone(), basis.detach().clone(),
                         torch.stack(values), torch.stack(responses), tuple(seeds))
        out.validate()
        return out


class BAOABOracle:
    """Small-system, underdamped Langevin oracle with explicit tangent propagation.

    All inputs use one internally consistent REDUCED unit system. Positions are a
    flat complete state, not an isolated local patch. Potential(q) returns a scalar
    differentiable energy. It must handle its own geometry/boundaries. No neighbor
    list, MACE adapter, constraints, box dynamics, or unit conversion is included.

    path_features(q_initial, positions_at_horizons[H,D]) must be smooth. Keeping the
    initial state as an explicit argument includes its derivative in change targets.
    Memory does not retain the entire force graph: only the current Hessian graph
    and the requested horizon states/tangents are kept.
    """
    def __init__(self, potential: Callable[[Tensor], Tensor],
                 path_features: Callable[[Tensor, Tensor], Tensor], *, mass: Tensor,
                 dt: float, steps: int, kbt: float, friction: float,
                 horizon_steps: Sequence[int]):
        if mass.ndim != 1 or not torch.all(mass > 0):
            raise ValueError("mass[D] must be positive.")
        if dt <= 0 or steps < 1 or kbt < 0 or friction < 0:
            raise ValueError("Invalid integrator parameters.")
        hs = tuple(int(h) for h in horizon_steps)
        if not hs or tuple(sorted(set(hs))) != hs or hs[0] < 1 or hs[-1] > steps:
            raise ValueError("Unique ascending horizon steps must lie in [1, steps].")
        self.potential, self.path_features, self.mass = potential, path_features, mass
        self.dt, self.steps, self.kbt, self.friction, self.horizons = dt, steps, kbt, friction, hs

    def _force_and_tangent(self, q: Tensor, dq: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            x = q.detach().requires_grad_(True)
            energy = self.potential(x)
            if energy.numel() != 1 or not energy.requires_grad:
                raise ValueError("Potential must return one differentiable scalar energy.")
            grad, = torch.autograd.grad(energy, x, create_graph=True)
            cols = []
            for j in range(dq.shape[1]):
                if grad.requires_grad:
                    hv, = torch.autograd.grad(grad, x, grad_outputs=dq[:, j],
                                             retain_graph=True, allow_unused=True)
                    hv = torch.zeros_like(x) if hv is None else hv
                else:
                    hv = torch.zeros_like(x)
                cols.append(-hv.detach())
            tangent = torch.stack(cols, -1) if cols else x.new_empty(x.numel(), 0)
            return -grad.detach(), tangent

    def _branch(self, initial: Tensor, basis: Tensor, seed: int) -> tuple[Tensor, Tensor]:
        gen = torch.Generator(device=initial.device).manual_seed(seed)
        mass = self.mass.to(initial)
        if mass.shape != initial.shape:
            raise ValueError("mass and complete q must have identical shape.")
        q = initial.detach().clone()
        p = (mass * self.kbt).sqrt() * torch.randn(q.shape, generator=gen, dtype=q.dtype, device=q.device)
        dq, dp = basis.detach().clone(), torch.zeros_like(basis)
        c = math.exp(-self.friction * self.dt)
        noise_scale = (mass * self.kbt * (1 - c*c)).sqrt()
        half = self.dt / 2
        saved_q, saved_dq = [], []
        force, df = self._force_and_tangent(q, dq)
        for step in range(1, self.steps + 1):
            p, dp = p + half * force, dp + half * df
            q, dq = q + half * p / mass, dq + half * dp / mass[:, None]
            noise = torch.randn(q.shape, generator=gen, dtype=q.dtype, device=q.device)
            p, dp = c * p + noise_scale * noise, c * dp
            q, dq = q + half * p / mass, dq + half * dp / mass[:, None]
            force, df = self._force_and_tangent(q, dq)
            p, dp = p + half * force, dp + half * df
            if step in self.horizons:
                saved_q.append(q.clone()); saved_dq.append(dq.clone())
        path, path_dq = torch.stack(saved_q), torch.stack(saved_dq)
        value = self.path_features(initial, path)
        responses = []
        for j in range(basis.shape[1]):
            _, tangent = torch.autograd.functional.jvp(
                self.path_features, (initial, path), (basis[:, j], path_dq[:, :, j]))
            responses.append(tangent)
        response = torch.stack(responses, -1) if responses else value.new_empty(value.numel(), 0)
        return value.detach(), response.detach()

    def query(self, q: Tensor, basis: Tensor, seeds: Sequence[int]) -> ShotBundle:
        pairs = [self._branch(q, basis, int(seed)) for seed in seeds]
        calls = len(seeds) * (self.steps + 1)
        out = ShotBundle(q.detach().clone(), basis.detach().clone(),
                         torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs]),
                         tuple(seeds), force_calls=calls, hvp_calls=calls*basis.shape[1])
        out.validate()
        return out


class SmallAtlas(nn.Module):
    """Toy-only MLP bottleneck. Replace with a smooth invariant atomic encoder."""
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 4, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(),
                                     nn.Linear(width, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width), nn.Tanh(),
                                     nn.Linear(width, output_dim))

    def forward(self, q: Tensor) -> Tensor:
        return self.decoder(self.encoder(q))
