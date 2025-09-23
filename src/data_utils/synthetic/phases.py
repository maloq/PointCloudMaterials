"""Phase and motif library for synthetic point-cloud scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - optional dependency
    linear_sum_assignment = None  # type: ignore[assignment]

from .config import PhaseName


@dataclass
class MotifParams:
    radius: float
    lattice_spacing: float | None = None
    shell_thickness: float | None = None
    aspect_ratio: tuple[float, float, float] | None = None
    extra: Dict[str, float] | None = None


class PhasePrototype:
    name: PhaseName
    default_params: MotifParams

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


def _sphere_mask(points: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points, axis=1) <= radius


def _lattice_grid(max_extent: float, spacing: float) -> np.ndarray:
    n = max(1, int(math.ceil(max_extent / spacing)) + 1)
    grid_1d = np.arange(-n, n + 1) * spacing
    return grid_1d


def _fcc_points(radius: float, spacing: float) -> np.ndarray:
    grid = _lattice_grid(radius, spacing)
    offsets = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]) * spacing
    pts = []
    for ox, oy, oz in offsets:
        gx, gy, gz = np.meshgrid(grid + ox, grid + oy, grid + oz, indexing="ij")
        block = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        pts.append(block)
    pts = np.vstack(pts)
    mask = _sphere_mask(pts, radius)
    return pts[mask]


def _bcc_points(radius: float, spacing: float) -> np.ndarray:
    grid = _lattice_grid(radius, spacing)
    base = []
    gx, gy, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    base.append(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]))
    gx, gy, gz = np.meshgrid(grid + spacing / 2, grid + spacing / 2, grid + spacing / 2, indexing="ij")
    base.append(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]))
    pts = np.vstack(base)
    mask = _sphere_mask(pts, radius)
    return pts[mask]


def _hcp_points(radius: float, spacing: float) -> np.ndarray:
    a = spacing
    c = math.sqrt(8 / 3) * a
    n = int(math.ceil(radius / a)) + 2
    pts = []
    for k in range(-n, n + 1):
        z = k * (c / 2)
        shift = (k % 2) * 0.5
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * (math.sqrt(3) / 2) * a
                pts.append([x + shift * a, y, z])
    pts = np.array(pts)
    mask = _sphere_mask(pts, radius)
    return pts[mask]


def _icosahedron_shell(radius: float, thickness: float) -> np.ndarray:
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer = verts * radius
    inner = verts * max(radius - thickness, 0.0)
    return np.vstack([outer, inner, np.zeros((1, 3))])


def _amorphous_points(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    pts = []
    while len(pts) < count:
        candidate = rng.uniform(-radius, radius, size=(count * 2, 3))
        mask = np.linalg.norm(candidate, axis=1) <= radius
        pts.extend(candidate[mask])
    return np.asarray(pts[:count])


def _anisotropic_ball(radius: float, aspect: tuple[float, float, float], count: int, rng: np.random.Generator) -> np.ndarray:
    base = _amorphous_points(radius, count, rng)
    return base * np.array(aspect)


class FCCLocal(PhasePrototype):
    name = "fcc"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.1, lattice_spacing=0.05)

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        spacing = params.lattice_spacing or self.default_params.lattice_spacing
        return _fcc_points(params.radius, spacing)


class BCCLocal(PhasePrototype):
    name = "bcc"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.1, lattice_spacing=0.055)

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        spacing = params.lattice_spacing or self.default_params.lattice_spacing
        return _bcc_points(params.radius, spacing)


class HCPLocal(PhasePrototype):
    name = "hcp"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.1, lattice_spacing=0.05)

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        spacing = params.lattice_spacing or self.default_params.lattice_spacing
        return _hcp_points(params.radius, spacing)


class IcosaLocal(PhasePrototype):
    name = "icosa"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.1, shell_thickness=0.02)

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        thickness = params.shell_thickness or self.default_params.shell_thickness or 0.0
        return _icosahedron_shell(params.radius, thickness)


class AmorphousLocal(PhasePrototype):
    name = "amorphous"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.1)

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        rng = np.random.default_rng(1234)
        return _amorphous_points(params.radius, 800, rng)


class RodLocal(PhasePrototype):
    name = "rod"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.12, aspect_ratio=(0.5, 0.5, 2.5))

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        aspect = params.aspect_ratio or self.default_params.aspect_ratio or (1.0, 1.0, 3.0)
        rng = np.random.default_rng(5678)
        return _anisotropic_ball(params.radius, aspect, 900, rng)


class PlateLocal(PhasePrototype):
    name = "plate"  # type: ignore[assignment]
    default_params = MotifParams(radius=0.12, aspect_ratio=(1.5, 1.5, 0.4))

    def canonical_points(self, params: MotifParams | None = None) -> np.ndarray:
        params = params or self.default_params
        aspect = params.aspect_ratio or self.default_params.aspect_ratio or (1.0, 1.0, 0.5)
        rng = np.random.default_rng(91011)
        return _anisotropic_ball(params.radius, aspect, 900, rng)


_DEFAULT_PROTOTYPES: Dict[PhaseName, PhasePrototype] = {
    "fcc": FCCLocal(),
    "bcc": BCCLocal(),
    "hcp": HCPLocal(),
    "icosa": IcosaLocal(),
    "amorphous": AmorphousLocal(),
    "rod": RodLocal(),
    "plate": PlateLocal(),
}


class PhaseLibrary:
    def __init__(self, prototypes: Iterable[PhasePrototype] | None = None) -> None:
        if prototypes is None:
            self._prototypes = dict(_DEFAULT_PROTOTYPES)
        else:
            self._prototypes = {p.name: p for p in prototypes}

    def get(self, name: PhaseName) -> PhasePrototype:
        try:
            return self._prototypes[name]
        except KeyError as exc:
            raise KeyError(f"Unknown phase prototype {name!r}") from exc

    def names(self) -> List[PhaseName]:
        return list(self._prototypes.keys())


def instantiate_motif(
    phase: PhasePrototype,
    params: MotifParams,
    M: int,
    *,
    rng: np.random.Generator,
    jitter: float | None = None,
) -> np.ndarray:
    canonical = phase.canonical_points(params)
    if canonical.shape[0] == 0:
        raise ValueError(f"Phase {phase.name} produced zero canonical points.")
    radius = params.radius
    jitter = jitter if jitter is not None else 0.01 * radius
    if canonical.shape[0] >= M:
        idx = rng.choice(canonical.shape[0], size=M, replace=False)
        points = canonical[idx]
    else:
        repeats = int(math.ceil(M / canonical.shape[0]))
        tiled = np.tile(canonical, (repeats, 1))
        points = tiled[:M]
    noise = rng.normal(scale=jitter, size=points.shape)
    return points + noise


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Inputs must be of shape (N, 3) and (M, 3).")
    diff_a = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    diff_b = diff_a.T
    return float(diff_a.min(axis=1).mean() + diff_b.min(axis=1).mean())


def earth_mover_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("EMD expects equal number of points; resample beforehand.")
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for earth_mover_distance")
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())
