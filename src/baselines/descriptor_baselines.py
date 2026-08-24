from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA

try:
    from scipy.special import sph_harm as _scipy_sph_harm
except ImportError:
    from scipy.special import sph_harm_y as _scipy_sph_harm_y

    def _scipy_sph_harm(m, n, theta, phi):
        # SciPy >= 1.17 removed sph_harm in favor of sph_harm_y, which uses
        # the argument order (n, m, polar, azimuth) instead of the legacy
        # (m, n, azimuth, polar) convention used throughout this file.
        return _scipy_sph_harm_y(n, m, phi, theta)


@dataclass(frozen=True)
class CenterShell:
    center_idx: int
    cutoff: float
    shell_indices: np.ndarray
    shell_distances: np.ndarray


def infer_center_shell(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
    shell_min_neighbors: int,
    shell_max_neighbors: int,
) -> CenterShell:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {tuple(pts.shape)}.")
    if pts.shape[0] < shell_min_neighbors + 2:
        raise ValueError(
            "Point cloud does not contain enough atoms to infer a center shell: "
            f"num_points={int(pts.shape[0])}, shell_min_neighbors={shell_min_neighbors}."
        )

    norms = np.linalg.norm(pts, axis=1)
    center_idx = int(np.argmin(norms))
    center_norm = float(norms[center_idx])
    if center_norm > float(center_atom_tolerance):
        raise ValueError(
            "Point cloud is expected to be centered on an atom at the origin, but the nearest atom is too far away: "
            f"center_idx={center_idx}, center_distance={center_norm:.6e}, "
            f"center_atom_tolerance={float(center_atom_tolerance):.6e}."
        )

    center = pts[center_idx]
    distances = np.linalg.norm(pts - center[None, :], axis=1)
    keep = np.arange(pts.shape[0]) != center_idx
    neighbor_indices = np.flatnonzero(keep)
    neighbor_distances = distances[keep]
    order = np.argsort(neighbor_distances)
    ordered_indices = neighbor_indices[order]
    ordered_distances = neighbor_distances[order]
    if ordered_distances.size < shell_min_neighbors + 1:
        raise ValueError(
            "Need at least shell_min_neighbors + 1 neighbors to infer a shell cutoff, "
            f"got {int(ordered_distances.size)}."
        )

    max_rank = min(int(shell_max_neighbors), int(ordered_distances.size - 1))
    if max_rank < shell_min_neighbors:
        raise ValueError(
            "shell_max_neighbors must allow at least one cutoff gap candidate: "
            f"shell_min_neighbors={shell_min_neighbors}, shell_max_neighbors={shell_max_neighbors}, "
            f"available_neighbors={int(ordered_distances.size)}."
        )
    candidate_distances = ordered_distances[: max_rank + 1]
    gaps = np.diff(candidate_distances)
    start_idx = int(shell_min_neighbors - 1)
    if start_idx >= gaps.size:
        raise RuntimeError(
            "Internal shell-inference error: no candidate gap remains after applying shell_min_neighbors. "
            f"candidate_distances.shape={tuple(candidate_distances.shape)}, start_idx={start_idx}."
        )
    search_gaps = gaps[start_idx:]
    gap_tol = max(1e-10, 1e-8 * float(candidate_distances[-1]))
    if float(np.max(search_gaps)) <= gap_tol:
        shell_size = int(candidate_distances.size)
        cutoff = float(candidate_distances[-1] + gap_tol)
    else:
        gap_idx = int(start_idx + np.argmax(search_gaps))
        shell_size = int(gap_idx + 1)
        cutoff = float(0.5 * (candidate_distances[gap_idx] + candidate_distances[gap_idx + 1]))
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError(
            f"Inferred an invalid shell cutoff={cutoff!r} from candidate distances {candidate_distances.tolist()}."
        )
    return CenterShell(
        center_idx=center_idx,
        cutoff=cutoff,
        shell_indices=ordered_indices[:shell_size].copy(),
        shell_distances=ordered_distances[:shell_size].copy(),
    )


class DescriptorBaseline(ABC):
    requires_fit: bool = False

    @abstractmethod
    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, point_clouds: np.ndarray) -> None:
        return None

    def fit_transform(self, point_clouds: np.ndarray) -> np.ndarray:
        self.fit(point_clouds)
        return self.transform(point_clouds)

    def metadata(self) -> dict[str, Any]:
        return {}


class SteinhardtDescriptorBaseline(DescriptorBaseline):
    def __init__(
        self,
        *,
        l_values: Sequence[int],
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        append_shell_size: bool,
    ) -> None:
        self.l_values = [int(v) for v in l_values]
        if not self.l_values:
            raise ValueError("Steinhardt baseline requires at least one l value.")
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.append_shell_size = bool(append_shell_size)

    @staticmethod
    def _ql(vectors: np.ndarray, l: int) -> float:
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(
                f"Steinhardt q_{l} received zero-length neighbor vectors; cannot evaluate spherical harmonics."
            )
        x = vectors[:, 0]
        y = vectors[:, 1]
        z = vectors[:, 2]
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        phi = np.arccos(np.clip(z / norms, -1.0, 1.0))
        qlm = []
        for m in range(-l, l + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                qlm.append(np.mean(_scipy_sph_harm(m, l, theta, phi)))
        qlm_arr = np.asarray(qlm, dtype=np.complex128)
        prefactor = 4.0 * np.pi / float(2 * l + 1)
        return float(np.sqrt(prefactor * np.sum(np.abs(qlm_arr) ** 2)).real)

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        rows: list[np.ndarray] = []
        for sample_idx, points in enumerate(pcs):
            shell = infer_center_shell(
                points,
                center_atom_tolerance=self.center_atom_tolerance,
                shell_min_neighbors=self.shell_min_neighbors,
                shell_max_neighbors=self.shell_max_neighbors,
            )
            center = points[shell.center_idx]
            vectors = points[shell.shell_indices] - center[None, :]
            values = [self._ql(vectors, l) for l in self.l_values]
            if self.append_shell_size:
                values.append(float(shell.shell_indices.size))
            rows.append(np.asarray(values, dtype=np.float32))
            if not np.isfinite(rows[-1]).all():
                raise ValueError(
                    "Steinhardt baseline produced non-finite features: "
                    f"sample_idx={sample_idx}, values={rows[-1].tolist()}."
                )
        return np.vstack(rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "l_values": list(self.l_values),
            "append_shell_size": self.append_shell_size,
            "shell_min_neighbors": self.shell_min_neighbors,
            "shell_max_neighbors": self.shell_max_neighbors,
        }


class SOAPDescriptorBaseline(DescriptorBaseline):
    requires_fit = True

    def __init__(
        self,
        *,
        species: str,
        point_scale: float,
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        r_cut: float | None,
        r_cut_multiplier: float,
        r_cut_min: float,
        n_max: int,
        l_max: int,
        sigma: float,
        pca_components: int | None,
        fit_max_pointclouds: int,
        n_jobs: int,
    ) -> None:
        self.species = str(species)
        self.point_scale = float(point_scale)
        if self.point_scale <= 0.0:
            raise ValueError(f"SOAP point_scale must be > 0, got {self.point_scale}.")
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.requested_r_cut = None if r_cut is None else float(r_cut)
        self.r_cut_multiplier = float(r_cut_multiplier)
        self.r_cut_min = float(r_cut_min)
        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.sigma = float(sigma)
        self.pca_components = None if pca_components is None else int(pca_components)
        self.fit_max_pointclouds = int(fit_max_pointclouds)
        self.n_jobs = int(n_jobs)
        self.soap: Any | None = None
        self.pca: PCA | None = None
        self.effective_r_cut: float | None = None

    def _scale_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) * self.point_scale

    @staticmethod
    def _build_soap_descriptor(
        *,
        species: str,
        r_cut: float,
        n_max: int,
        l_max: int,
        sigma: float,
    ):
        try:
            from dscribe.descriptors import SOAP
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SOAPDescriptorBaseline requires DScribe. "
                "Install the repository requirements before using it."
            ) from exc
        return SOAP(
            species=[species],
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=False,
            compression={"mode": "off"},
            sparse=False,
            dtype="float64",
        )

    def _infer_r_cut(self, point_clouds: np.ndarray) -> float:
        if self.requested_r_cut is not None:
            if self.requested_r_cut <= 0.0:
                raise ValueError(f"SOAP r_cut must be > 0, got {self.requested_r_cut}.")
            return float(self.requested_r_cut)

        if self.fit_max_pointclouds <= 0:
            raise ValueError(
                "SOAP baseline requires fit_max_pointclouds > 0 when r_cut is not explicitly configured."
            )
        shell_cutoffs: list[float] = []
        max_clouds = min(int(point_clouds.shape[0]), int(self.fit_max_pointclouds))
        for idx in range(max_clouds):
            scaled_points = self._scale_points(point_clouds[idx])
            shell = infer_center_shell(
                scaled_points,
                center_atom_tolerance=self.center_atom_tolerance,
                shell_min_neighbors=self.shell_min_neighbors,
                shell_max_neighbors=self.shell_max_neighbors,
            )
            shell_cutoffs.append(float(shell.cutoff))
        if not shell_cutoffs:
            raise RuntimeError("Failed to infer any shell cutoff while estimating SOAP r_cut.")
        r_cut = max(float(np.median(shell_cutoffs)) * self.r_cut_multiplier, self.r_cut_min)
        if not np.isfinite(r_cut) or r_cut <= 0.0:
            raise ValueError(
                f"Estimated an invalid SOAP r_cut={r_cut!r} from shell cutoffs {shell_cutoffs[:10]!r}."
            )
        return float(r_cut)

    def _center_soap_vector(self, points: np.ndarray) -> np.ndarray:
        if self.soap is None:
            raise RuntimeError("SOAP baseline has not been fitted yet.")
        try:
            from ase import Atoms
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SOAPDescriptorBaseline requires ASE. "
                "Install the repository requirements before using it."
            ) from exc
        scaled_points = self._scale_points(points)
        shell = infer_center_shell(
            scaled_points,
            center_atom_tolerance=self.center_atom_tolerance,
            shell_min_neighbors=self.shell_min_neighbors,
            shell_max_neighbors=self.shell_max_neighbors,
        )
        atoms = Atoms(
            symbols=[self.species] * int(scaled_points.shape[0]),
            positions=np.asarray(scaled_points, dtype=np.float64),
        )
        raw = self.soap.create(atoms, centers=[int(shell.center_idx)], n_jobs=self.n_jobs)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                "Expected SOAP.create(..., centers=[center_idx]) to return shape (1, F), "
                f"got {tuple(arr.shape)}."
            )
        return arr[0]

    def fit(self, point_clouds: np.ndarray) -> None:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if pcs.shape[0] <= 0:
            raise ValueError("SOAP baseline received zero training point clouds.")
        r_cut = self._infer_r_cut(pcs)
        if r_cut <= 1.0:
            raise ValueError(
                "SOAP baseline requires r_cut > 1.0 for DScribe's Gaussian basis, but got "
                f"r_cut={r_cut:.6f}. Increase descriptor.soap.point_scale or set descriptor.soap.r_cut explicitly."
            )
        self.effective_r_cut = float(r_cut)
        self.soap = self._build_soap_descriptor(
            species=self.species,
            r_cut=float(r_cut),
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
        )

        max_clouds = int(pcs.shape[0]) if self.fit_max_pointclouds <= 0 else min(int(pcs.shape[0]), self.fit_max_pointclouds)
        rows = [self._center_soap_vector(pcs[idx]) for idx in range(max_clouds)]
        if not rows:
            raise RuntimeError("SOAP baseline could not build any training descriptor rows.")
        X = np.vstack(rows)
        requested_components = X.shape[1] if self.pca_components is None else int(self.pca_components)
        n_components = min(int(requested_components), int(X.shape[0]), int(X.shape[1]))
        if n_components <= 0:
            raise ValueError(
                "SOAP PCA resolved to zero components: "
                f"requested={requested_components}, training_shape={tuple(X.shape)}."
            )
        self.pca = PCA(n_components=n_components, whiten=False, random_state=0)
        self.pca.fit(X)

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if self.soap is None or self.pca is None:
            raise RuntimeError("SOAP baseline must be fitted before calling transform().")
        rows = [self._center_soap_vector(points) for points in pcs]
        X = np.vstack(rows)
        return np.asarray(self.pca.transform(X), dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "point_scale": self.point_scale,
            "effective_r_cut": self.effective_r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
            "pca_components": None if self.pca is None else int(self.pca.n_components_),
            "fit_max_pointclouds": self.fit_max_pointclouds,
        }


class CNADescriptorBaseline(DescriptorBaseline):
    requires_fit = True

    def __init__(
        self,
        *,
        center_atom_tolerance: float,
        shell_min_neighbors: int,
        shell_max_neighbors: int,
        max_signatures: int,
        append_shell_size: bool,
        fit_max_pointclouds: int,
    ) -> None:
        self.center_atom_tolerance = float(center_atom_tolerance)
        self.shell_min_neighbors = int(shell_min_neighbors)
        self.shell_max_neighbors = int(shell_max_neighbors)
        self.max_signatures = int(max_signatures)
        self.append_shell_size = bool(append_shell_size)
        self.fit_max_pointclouds = int(fit_max_pointclouds)
        self.signature_vocab: list[str] = []

    @staticmethod
    def _longest_chain_length(nodes: Sequence[int], adjacency: dict[int, set[int]]) -> int:
        if not nodes:
            return 0
        node_set = {int(v) for v in nodes}
        seen: set[int] = set()
        best = 0

        # Use the maximum geodesic distance within each connected component of the
        # common-neighbor bond graph. For CNA motifs this preserves the intended
        # chain/ring ordering while keeping runtime bounded on noisy dense graphs.
        for start in sorted(node_set):
            if start in seen:
                continue
            component: list[int] = []
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in node_set or neighbor in seen:
                        continue
                    seen.add(neighbor)
                    stack.append(neighbor)

            if len(component) <= 1:
                continue

            component_set = set(component)
            for source in component:
                distances = {source: 0}
                queue = [source]
                for current in queue:
                    cur_dist = distances[current]
                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in component_set or neighbor in distances:
                            continue
                        distances[neighbor] = cur_dist + 1
                        queue.append(neighbor)
                best = max(best, max(distances.values(), default=0))

        return int(best)

    def _signature_counts(self, points: np.ndarray) -> tuple[Counter, int]:
        shell = infer_center_shell(
            points,
            center_atom_tolerance=self.center_atom_tolerance,
            shell_min_neighbors=self.shell_min_neighbors,
            shell_max_neighbors=self.shell_max_neighbors,
        )
        local_indices = np.concatenate(
            [
                np.asarray([int(shell.center_idx)], dtype=np.int64),
                np.asarray(shell.shell_indices, dtype=np.int64),
            ]
        )
        local_points = np.asarray(points[local_indices], dtype=np.float64)
        pairwise = np.linalg.norm(
            local_points[:, None, :] - local_points[None, :, :],
            axis=-1,
        )
        within_cutoff = pairwise <= float(shell.cutoff)
        adjacency: dict[int, set[int]] = {}
        for idx in range(local_points.shape[0]):
            neighbor_idx = set(np.flatnonzero(within_cutoff[idx]).tolist())
            neighbor_idx.discard(idx)
            adjacency[int(idx)] = neighbor_idx

        center_idx = 0
        shell_neighbor_set = set(range(1, int(local_points.shape[0])))
        counts: Counter = Counter()
        for neighbor_idx in sorted(shell_neighbor_set):
            common = sorted(adjacency[center_idx].intersection(adjacency[neighbor_idx]))
            n_common = len(common)
            common_set = set(common)
            subgraph: dict[int, set[int]] = {
                node: adjacency[node].intersection(common_set)
                for node in common
            }
            n_bonds = int(sum(len(neigh) for neigh in subgraph.values()) // 2)
            longest_chain = self._longest_chain_length(common, subgraph)
            signature = f"{n_common}-{n_bonds}-{longest_chain}"
            counts[signature] += 1

        if sum(counts.values()) != len(shell_neighbor_set):
            raise RuntimeError(
                "CNA signature counting mismatch: "
                f"counted_bonds={sum(counts.values())}, shell_size={len(shell_neighbor_set)}."
            )
        return counts, len(shell_neighbor_set)

    def fit(self, point_clouds: np.ndarray) -> None:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        max_clouds = int(pcs.shape[0]) if self.fit_max_pointclouds <= 0 else min(int(pcs.shape[0]), self.fit_max_pointclouds)
        global_counts: Counter = Counter()
        for idx in range(max_clouds):
            counts, _ = self._signature_counts(pcs[idx])
            global_counts.update(counts)
        if not global_counts:
            raise RuntimeError("CNA baseline did not observe any signatures during fit().")
        most_common = global_counts.most_common(self.max_signatures)
        self.signature_vocab = [signature for signature, _ in most_common]

    def transform(self, point_clouds: np.ndarray) -> np.ndarray:
        pcs = np.asarray(point_clouds, dtype=np.float64)
        if pcs.ndim != 3 or pcs.shape[-1] != 3:
            raise ValueError(
                f"Expected batched point clouds with shape (B, N, 3), got {tuple(pcs.shape)}."
            )
        if not self.signature_vocab:
            raise RuntimeError("CNA baseline must be fitted before calling transform().")

        rows: list[np.ndarray] = []
        for sample_idx, points in enumerate(pcs):
            counts, shell_size = self._signature_counts(points)
            total = max(1, sum(counts.values()))
            values = np.zeros(len(self.signature_vocab) + 1 + int(self.append_shell_size), dtype=np.float32)
            for sig_idx, signature in enumerate(self.signature_vocab):
                values[sig_idx] = float(counts.get(signature, 0)) / float(total)
            other_count = total - sum(int(counts.get(signature, 0)) for signature in self.signature_vocab)
            values[len(self.signature_vocab)] = float(other_count) / float(total)
            if self.append_shell_size:
                values[-1] = float(shell_size)
            if not np.isfinite(values).all():
                raise ValueError(
                    "CNA baseline produced non-finite features: "
                    f"sample_idx={sample_idx}, values={values.tolist()}."
                )
            rows.append(values)
        return np.vstack(rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "signature_vocab": list(self.signature_vocab),
            "max_signatures": self.max_signatures,
            "append_shell_size": self.append_shell_size,
            "fit_max_pointclouds": self.fit_max_pointclouds,
        }
