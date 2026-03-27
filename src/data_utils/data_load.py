import sys,os
sys.path.append(os.getcwd())
import json
import numpy as np
import logging 
import torch
from torch.utils.data import Dataset
from src.data_utils.prepare_data import get_regular_samples, get_random_samples
from src.data_utils.prepare_data import read_off_file
from src.utils.logging_config import setup_logging
import pandas as pd
from typing import Union, Optional, Sequence, Dict, Any, List, Tuple
from pathlib import Path
import math
from scipy.spatial import cKDTree

logger = setup_logging()



def pc_normalize(pc, radius = None):
    # centroid = np.mean(pc, axis=0)
    # pc = pc - centroid
    if radius:
        pc = pc / radius
    else:
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
    return pc


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: (3, 3) rotation matrix as numpy array
        
    Returns:
        (4,) quaternion as numpy array [w, x, y, z]
    """
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {R.shape}")
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def rotation_matrix_to_quaternion_batch(R: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to quaternions (w, x, y, z).
    
    Args:
        R: (..., 3, 3) rotation matrices as torch tensor
        
    Returns:
        (..., 4) quaternions as torch tensor [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    batch_size = R.shape[0]
    
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    
    tr = m00 + m11 + m22
    quat = torch.zeros(batch_size, 4, dtype=R.dtype, device=R.device)
    
    # Case 1: tr > 0
    mask1 = tr > 0
    if mask1.any():
        S = torch.sqrt(tr[mask1] + 1.0) * 2.0
        quat[mask1, 0] = 0.25 * S
        quat[mask1, 1] = (m21[mask1] - m12[mask1]) / S
        quat[mask1, 2] = (m02[mask1] - m20[mask1]) / S
        quat[mask1, 3] = (m10[mask1] - m01[mask1]) / S
    
    # Case 2: m00 > m11 and m00 > m22
    mask2 = ~mask1 & (m00 > m11) & (m00 > m22)
    if mask2.any():
        S = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2.0
        quat[mask2, 0] = (m21[mask2] - m12[mask2]) / S
        quat[mask2, 1] = 0.25 * S
        quat[mask2, 2] = (m01[mask2] + m10[mask2]) / S
        quat[mask2, 3] = (m02[mask2] + m20[mask2]) / S
    
    # Case 3: m11 > m22
    mask3 = ~mask1 & ~mask2 & (m11 > m22)
    if mask3.any():
        S = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2.0
        quat[mask3, 0] = (m02[mask3] - m20[mask3]) / S
        quat[mask3, 1] = (m01[mask3] + m10[mask3]) / S
        quat[mask3, 2] = 0.25 * S
        quat[mask3, 3] = (m12[mask3] + m21[mask3]) / S
    
    # Case 4: else
    mask4 = ~mask1 & ~mask2 & ~mask3
    if mask4.any():
        S = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2.0
        quat[mask4, 0] = (m10[mask4] - m01[mask4]) / S
        quat[mask4, 1] = (m02[mask4] + m20[mask4]) / S
        quat[mask4, 2] = (m12[mask4] + m21[mask4]) / S
        quat[mask4, 3] = 0.25 * S
    
    # Normalize
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    return quat.reshape(*batch_shape, 4)

def _load_points(filepath: str) -> np.ndarray:
    """Load point cloud from .npy or .off file. Returns float32 (N, 3) array."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.npy':
        points = np.load(filepath)
    elif ext == '.off':
        points = read_off_file(filepath, verbose=False)
    else:
        raise ValueError(f"Unsupported file extension {ext!r} for {filepath}. Use .npy or .off")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array from {filepath}, got shape {points.shape}")
    return points.astype(np.float32, copy=False)


def _sample_single_file(
    filepath,
    sample_type,
    radius,
    n_points,
    overlap_fraction,
    n_samples,
    return_coords,
    sampling_method,
    drop_edge_samples=True,
    edge_drop_layers: int | None = None,
):
    """Load one file and generate samples. Standalone function for multiprocessing."""
    points = _load_points(filepath)
    if sample_type == 'regular':
        return get_regular_samples(
            points, size=radius, n_points=n_points,
            overlap_fraction=overlap_fraction,
            return_coords=return_coords,
            drop_edge_samples=bool(drop_edge_samples),
            edge_drop_layers=edge_drop_layers,
            sampling_method=sampling_method,
        )
    elif sample_type == 'random':
        return get_random_samples(
            points, n_samples=n_samples, size=radius,
            n_points=n_points, return_coords=return_coords,
            sampling_method=sampling_method,
        )
    else:
        raise ValueError(f"Invalid sample type: {sample_type!r}")


def read_and_sample_files(
    root,
    data_files,
    radius,
    n_points,
    overlap_fraction,
    sample_type,
    n_samples,
    return_coords,
    sampling_method="drop_farthest",
    drop_edge_samples=True,
    edge_drop_layers: int | None = None,
):
    """Read point cloud files and sample sub-clouds. Processes files in parallel when possible."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    filepaths = [os.path.join(root, f) for f in data_files]
    n_files = len(filepaths)
    max_workers = min(n_files, max(1, multiprocessing.cpu_count() // 2))
    args = (
        sample_type,
        radius,
        n_points,
        overlap_fraction,
        n_samples,
        return_coords,
        sampling_method,
        drop_edge_samples,
        edge_drop_layers,
    )

    all_samples: list = []

    if n_files > 1 and max_workers > 1:
        logger.info(f"Processing {n_files} files in parallel with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_sample_single_file, fp, *args) for fp in filepaths]
            for future in futures:
                file_samples = future.result()
                all_samples.extend(file_samples)
    else:
        for fp in filepaths:
            file_samples = _sample_single_file(fp, *args)
            all_samples.extend(file_samples)

    if not all_samples:
        raise ValueError(f"No samples found for {data_files} in {root}")
    return all_samples


class PointCloudDataset(Dataset):
    def __init__(self,
                 root: str = "",
                 data_files: list[str] | None = None,
                 data_sources: list[dict] | None = None,
                 return_coords=False,
                 sample_type='regular',
                 radius=8,
                 overlap_fraction=0.0,
                 n_samples=1000,
                 num_points=100,
                 drop_edge_samples=True,
                 edge_drop_layers: int | None = None,
                 pre_normalize=True,
                 normalize=True,
                 sampling_method="drop_farthest",
                 auto_cutoff_config: dict[str, Any] | None = None):
        """Initialize the dataset with samples from point cloud files.

        Supports two configuration modes:

        1. **Single source** (backward-compatible): provide ``root`` + ``data_files``.
        2. **Multi source**: provide ``data_sources``, a list of dicts each with
           ``data_path`` and ``data_files`` keys.  Samples from all sources are
           concatenated into a single dataset.

        Args:
            root: Directory containing data files (single-source mode).
            data_files: List of filenames relative to *root* (single-source mode).
            data_sources: List of ``{"data_path": str, "data_files": [str, ...]}``
                dicts (multi-source mode).  When provided, *root* and *data_files*
                are ignored.
        """
        self.pre_normalize = pre_normalize
        self.normalize = normalize
        self.return_coords = return_coords
        self.sample_type = sample_type
        self.sampling_method = sampling_method
        self.radius = float(radius)
        self.num_points = int(num_points)
        self.drop_edge_samples = bool(drop_edge_samples)
        self.edge_drop_layers = edge_drop_layers

        auto_cfg = self._resolve_auto_cutoff_config(
            auto_cutoff_config,
            default_target_points=self.num_points,
            default_radius=self.radius,
        )

        sources = self._resolve_sources(root, data_files, data_sources)
        self.source_radii: dict[str, float] = {}
        self.sample_source_names: list[str] = []
        all_sample_radii: list[float] = []
        all_samples: list = []

        for source in sources:
            src_name = source["name"]
            src_root = source["root"]
            src_files = source["files"]

            src_radius = self._resolve_source_cutoff_radius(
                source=source,
                default_radius=self.radius,
                auto_cutoff_config=auto_cfg,
                num_points=self.num_points,
            )
            self.source_radii[src_name] = src_radius

            samples = read_and_sample_files(
                src_root, src_files, src_radius, self.num_points,
                overlap_fraction, sample_type, n_samples,
                return_coords, sampling_method,
                self.drop_edge_samples, self.edge_drop_layers,
            )
            all_samples.extend(samples)
            all_sample_radii.extend([src_radius] * len(samples))
            self.sample_source_names.extend([src_name] * len(samples))

        if not all_samples:
            raise ValueError("No samples generated from any data source")

        self.samples = all_samples
        self.sample_radii = all_sample_radii

        if self.return_coords:
            self.samples, self.coords = zip(*self.samples)
            self.samples = list(self.samples)
            self.coords = list(self.coords)
        else:
            self.coords = None

        if self.pre_normalize and self.normalize:
            self.samples = [
                pc_normalize(sample, sample_radius).astype(np.float32)
                for sample, sample_radius in zip(self.samples, self.sample_radii)
            ]
        elif not self.normalize:
            print("Point Cloud normalization skipped")
        logger.info(f"Point set shape: {self.samples[0].shape}")
        if len(self.source_radii) > 1:
            formatted = ", ".join(
                f"{name}: {radius_val:.4f}"
                for name, radius_val in sorted(self.source_radii.items(), key=lambda kv: kv[0])
            )
            logger.print(f"Per-source cutoff radii: {formatted}")

    @staticmethod
    def _resolve_sources(
        root: str,
        data_files: list[str] | None,
        data_sources: list[dict] | None,
    ) -> list[dict[str, Any]]:
        """Return list of source descriptors."""
        if data_sources:
            sources = []
            for source_index, src in enumerate(data_sources):
                src_path = src["data_path"]
                src_files = src["data_files"]
                if isinstance(src_files, str):
                    src_files = [src_files]
                source_name_raw = src.get("name", None)
                source_name = (
                    str(source_name_raw)
                    if source_name_raw is not None
                    else (Path(str(src_path)).name or f"source_{source_index}")
                )
                sources.append(
                    {
                        "index": int(source_index),
                        "name": source_name,
                        "root": str(src_path),
                        "files": list(src_files),
                        "radius_override": src.get("radius", None),
                    }
                )
            return sources

        if isinstance(data_files, str):
            data_files = [data_files]
        if not data_files:
            data_files = []
        source_name = Path(str(root)).name if str(root) else "single_source"
        return [
            {
                "index": 0,
                "name": source_name or "single_source",
                "root": str(root),
                "files": list(data_files),
                "radius_override": None,
            }
        ]

    @staticmethod
    def _resolve_auto_cutoff_config(
        auto_cutoff_config: dict[str, Any] | None,
        *,
        default_target_points: int,
        default_radius: float,
    ) -> dict[str, Any] | None:
        if not auto_cutoff_config or not auto_cutoff_config.get("enabled", False):
            return None
        return {
            "target_points": int(auto_cutoff_config.get("target_points", default_target_points)),
            "quantile": float(auto_cutoff_config.get("quantile", 1.0)),
            "estimation_samples_per_file": int(auto_cutoff_config.get("estimation_samples_per_file", 4096)),
            "seed": int(auto_cutoff_config.get("seed", 0)),
            "safety_factor": float(auto_cutoff_config.get("safety_factor", 1.0)),
            "boundary_margin": auto_cutoff_config.get("boundary_margin", default_radius),
        }

    def _resolve_source_cutoff_radius(
        self,
        *,
        source: dict[str, Any],
        default_radius: float,
        auto_cutoff_config: dict[str, Any] | None,
        num_points: int,
    ) -> float:
        source_name = str(source["name"])
        source_root = str(source["root"])
        source_files = source["files"]
        radius_override = source.get("radius_override", None)

        if radius_override is not None:
            return float(radius_override)

        if auto_cutoff_config is None:
            return float(default_radius)

        target_points = max(int(auto_cutoff_config["target_points"]), int(num_points))

        seed = int(auto_cutoff_config["seed"]) + int(source["index"])
        estimated_radius, coverage = self._estimate_source_cutoff_radius(
            source_root=source_root,
            source_files=source_files,
            target_points=target_points,
            quantile=float(auto_cutoff_config["quantile"]),
            estimation_samples_per_file=int(auto_cutoff_config["estimation_samples_per_file"]),
            seed=seed,
            safety_factor=float(auto_cutoff_config["safety_factor"]),
            boundary_margin=auto_cutoff_config["boundary_margin"],
        )
        logger.print(
            "[auto_cutoff] "
            f"source={source_name!r}, target_points={target_points}, "
            f"quantile={float(auto_cutoff_config['quantile']):.4f}, "
            f"coverage~{coverage * 100.0:.2f}%, "
            f"radius={estimated_radius:.4f} (default={default_radius:.4f})."
        )
        return estimated_radius

    @staticmethod
    def _estimate_source_cutoff_radius(
        *,
        source_root: str,
        source_files: list[str],
        target_points: int,
        quantile: float,
        estimation_samples_per_file: int,
        seed: int,
        safety_factor: float,
        boundary_margin: float | None,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        kth_distances_all: list[np.ndarray] = []

        for file_name in source_files:
            filepath = os.path.join(source_root, file_name)
            points = _load_points(filepath)
            num_atoms = len(points)
            candidate_indices = np.arange(num_atoms, dtype=np.int64)
            if boundary_margin is not None and boundary_margin > 0.0:
                boundary_margin = float(boundary_margin)
                min_coords = points.min(axis=0)
                max_coords = points.max(axis=0)
                interior_mask = np.all(
                    (points >= (min_coords + boundary_margin))
                    & (points <= (max_coords - boundary_margin)),
                    axis=1,
                )
                interior_indices = np.flatnonzero(interior_mask)
                if interior_indices.size > 0:
                    candidate_indices = interior_indices.astype(np.int64, copy=False)

            centers_to_sample = min(estimation_samples_per_file, int(candidate_indices.size))
            center_indices = rng.choice(candidate_indices, size=centers_to_sample, replace=False)

            tree = cKDTree(points)
            k = min(int(target_points), num_atoms)
            dists, _ = tree.query(points[center_indices], k=k)
            dists = np.asarray(dists, dtype=np.float64)
            if k == 1:
                kth_dist = dists.reshape(-1)
            else:
                kth_dist = dists[:, k - 1]
            kth_distances_all.append(kth_dist)

        kth_all = np.concatenate(kth_distances_all).astype(np.float64, copy=False)
        estimated_radius = float(np.quantile(kth_all, quantile)) * float(safety_factor)

        coverage = float(np.mean(kth_all <= estimated_radius))
        return estimated_radius, coverage

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_set = self.samples[index]
        if not self.pre_normalize and self.normalize:
            point_set = pc_normalize(point_set, float(self.sample_radii[index])).astype(np.float32)
        point_set_tensor = torch.tensor(point_set, dtype=torch.float32)
        
        if self.return_coords:
            return point_set_tensor, self.coords[index]
        else:
            return point_set_tensor

 
class SyntheticPointCloudDataset(Dataset):
    """Dataset for synthetic atomistic point clouds with metadata labels.

    Optimized for millions of atoms by using lazy metadata computation
    and pre-converted tensors.
    
    Returns dict with keys:
        - "points": (N, 3) point cloud tensor
        - "class_id": scalar int64 tensor (category/phase index)
        - "instance_id": scalar int64 tensor (grain/instance index, -1 if unknown)
        - "rotation": (3, 3) float32 rotation matrix tensor
        - "coords": (3,) float32 tensor with sample center in simulation space
    """

    # Dataset metadata
    domain: str = "materials"

    def __init__(
        self,
        env_dirs: Sequence[Union[str, Path]],
        *,
        radius: float,
        sample_type: str,
        overlap_fraction: float,
        n_samples: int,
        num_points: int,
        drop_edge_samples: bool = True,
        pre_normalize: bool = True,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        discard_mixed_phase: bool = False,
        sampling_method: str = "drop_farthest",
        rotation_scale: float = 0.0,
        noise_scale: float = 0.0,
        jitter_scale: float = 0.0,
        scaling_range: float = 0.0,
        normalization_scale: float = 1.0,
        track_augmentation: bool = False,
        allowed_classes: Optional[Sequence[str]] = None,
        auto_cutoff_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not env_dirs:
            raise ValueError("env_dirs must contain at least one synthetic dataset directory")
        self.radius = float(radius)
        self.sample_type = sample_type
        self.overlap_fraction = overlap_fraction
        self.n_samples = int(n_samples) if n_samples else 0
        self.num_points = num_points
        self.drop_edge_samples = drop_edge_samples
        self.pre_normalize = pre_normalize
        self.normalize = normalize
        self.max_samples = max_samples if max_samples is not None and max_samples > 0 else None
        self.discard_mixed_phase = discard_mixed_phase
        self.sampling_method = sampling_method
        self.rotation_scale = float(rotation_scale)
        self.noise_scale = float(noise_scale)
        self.jitter_scale = float(jitter_scale)
        self.scaling_range = float(scaling_range)
        self.normalization_scale = float(normalization_scale)
        self.track_augmentation = bool(track_augmentation)
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        self._augmentation_metadata: Optional[List[Dict[str, Any]]] = None
        self.auto_cutoff_config = PointCloudDataset._resolve_auto_cutoff_config(
            auto_cutoff_config,
            default_target_points=int(self.num_points),
            default_radius=float(self.radius),
        )

        # Store as tensors to avoid conversion overhead
        self.samples: List[torch.Tensor] = []
        self.sample_radii: List[float] = []
        self.sample_source_names: List[str] = []
        self._class_ids: List[int] = []
        self._instance_ids: List[int] = []
        self._rotations: List[torch.Tensor] = []
        self._coords: List[torch.Tensor] = []
        self.source_radii: Dict[str, float] = {}

        # Class mapping (class_name -> class_id)
        self._class_to_idx: Dict[str, int] = {}
        self._instance_to_idx: Dict[Tuple[str, str], int] = {}
        
        # Class properties for domain-specific info
        self._class_properties: Dict[str, Dict[str, Any]] = {}

        for env_index, env_dir in enumerate(env_dirs):
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break
            self._ingest_environment(env_dir, env_index)

        if not self.samples:
            raise RuntimeError("SyntheticPointCloudDataset constructed with zero samples")

        # Build class properties based on detected classes
        self._build_class_properties()

        if self.track_augmentation:
            self._augmentation_metadata = [None] * len(self.samples)

        if self.source_radii and (len(self.source_radii) > 1 or self.auto_cutoff_config is not None):
            formatted = ", ".join(
                f"{name}: {radius_val:.4f}"
                for name, radius_val in sorted(self.source_radii.items(), key=lambda kv: kv[0])
            )
            logger.print(f"Synthetic per-environment cutoff radii: {formatted}")

    def _ingest_environment(self, env_dir: Union[str, Path], env_index: int) -> None:
        env_path = Path(env_dir)
        if not env_path.exists():
            raise FileNotFoundError(f"Synthetic environment directory {env_path} does not exist")
        if self.max_samples is not None and len(self.samples) >= self.max_samples:
            return
        atoms_path = env_path / "atoms.npy"
        metadata_path = env_path / "metadata.json"
        if not atoms_path.exists():
            raise FileNotFoundError(f"atoms.npy missing in {env_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json missing in {env_path}")

        points = np.load(atoms_path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"atoms.npy at {atoms_path} must have shape (N, 3)")

        with metadata_path.open("r") as handle:
            metadata = json.load(handle)

        env_label = env_path.name or f"env_{env_index}"
        env_radius = self._resolve_environment_cutoff_radius(
            env_path=env_path,
            env_label=env_label,
            env_index=env_index,
        )
        self.source_radii[env_label] = env_radius

        # Build efficient metadata lookup instead of storing for all atoms
        atom_to_meta_idx = self._build_efficient_atom_metadata(metadata, env_label, points.shape[0])
        position_tree = cKDTree(points)

        # Build atom-to-phase mapping if needed for phase purity checking
        atom_phases = None
        if self.discard_mixed_phase:
            atom_phases = self._build_atom_phase_map(metadata, points.shape[0])

        samples = self._sample_points(points, env_radius)
        if not samples:
            logger.print(f"No samples generated for {env_label}")
            return

        samples_before = len(self.samples)
        discarded_mixed_phase = 0

        for sample_points, center in samples:
            center = np.asarray(center, dtype=np.float64)
            _, idx = position_tree.query(center.reshape(1, -1), k=1)
            atom_idx = int(idx[0])

            # Check for phase purity if enabled
            if self.discard_mixed_phase:
                # Query all atoms within the sampling radius
                atom_indices = position_tree.query_ball_point(center, env_radius)
                if len(atom_indices) > 0:
                    sample_phases = atom_phases[atom_indices]
                    unique_phases = np.unique(sample_phases)
                    # Discard if multiple phases are present
                    if len(unique_phases) > 1:
                        discarded_mixed_phase += 1
                        continue

            # Look up metadata on-demand
            meta = atom_to_meta_idx[atom_idx]

            # Filter by allowed classes if specified
            if self.allowed_classes is not None:
                if meta["phase_id"] not in self.allowed_classes:
                    continue

            processed = self._prepare_sample(sample_points, env_radius)
            class_idx = self._encode_class(meta["phase_id"])
            instance_idx = self._encode_instance(meta["grain_key"])

            # Store as tensors to avoid conversion overhead in __getitem__
            self.samples.append(torch.tensor(processed, dtype=torch.float32))
            self.sample_radii.append(float(env_radius))
            self.sample_source_names.append(env_label)
            self._class_ids.append(class_idx)
            self._instance_ids.append(instance_idx)
            self._rotations.append(torch.tensor(meta["orientation"], dtype=torch.float32))
            self._coords.append(torch.tensor(center, dtype=torch.float32))

            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break

        samples_added = len(self.samples) - samples_before
        if self.discard_mixed_phase and discarded_mixed_phase > 0:
            logger.print(
                f"Ingested {samples_added} samples from {env_label} "
                f"({discarded_mixed_phase} mixed-phase samples discarded); "
                f"dataset total now {len(self.samples)}"
            )
        else:
            logger.print(
                f"Ingested {samples_added} samples from {env_label}; "
                f"dataset total now {len(self.samples)}"
            )

    def _resolve_environment_cutoff_radius(
        self,
        *,
        env_path: Path,
        env_label: str,
        env_index: int,
    ) -> float:
        if self.auto_cutoff_config is None:
            return float(self.radius)

        target_points = max(
            int(self.auto_cutoff_config["target_points"]),
            int(self.num_points),
        )
        seed = int(self.auto_cutoff_config["seed"]) + int(env_index)
        estimated_radius, coverage = PointCloudDataset._estimate_source_cutoff_radius(
            source_root=str(env_path),
            source_files=["atoms.npy"],
            target_points=target_points,
            quantile=float(self.auto_cutoff_config["quantile"]),
            estimation_samples_per_file=int(self.auto_cutoff_config["estimation_samples_per_file"]),
            seed=seed,
            safety_factor=float(self.auto_cutoff_config["safety_factor"]),
            boundary_margin=self.auto_cutoff_config["boundary_margin"],
        )
        logger.print(
            "[auto_cutoff] "
            f"synthetic_env={env_label!r}, target_points={target_points}, "
            f"quantile={float(self.auto_cutoff_config['quantile']):.4f}, "
            f"coverage~{coverage * 100.0:.2f}%, "
            f"radius={estimated_radius:.4f} (default={float(self.radius):.4f})."
        )
        return estimated_radius

    def _sample_points(
        self,
        points: np.ndarray,
        sample_radius: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.sample_type == "regular":
            max_samples = self.n_samples if self.n_samples > 0 else int(2e9)
            raw = get_regular_samples(
                points,
                size=float(sample_radius),
                overlap_fraction=self.overlap_fraction,
                return_coords=True,
                n_points=self.num_points,
                max_samples=max_samples,
                drop_edge_samples=self.drop_edge_samples,
                sampling_method=self.sampling_method,
            )
        elif self.sample_type == "random":
            if self.n_samples <= 0:
                raise ValueError("n_samples must be > 0 for random sampling")
            raw = get_random_samples(
                points,
                n_samples=self.n_samples,
                size=float(sample_radius),
                n_points=self.num_points,
                return_coords=True,
                sampling_method=self.sampling_method,
            )
        else:
            raise ValueError(f"Invalid sample type: {self.sample_type!r}")
        return [(np.asarray(s, dtype=np.float32), np.asarray(c, dtype=np.float32)) for s, c in raw]

    def _prepare_sample(self, sample_points: np.ndarray, sample_radius: float) -> np.ndarray:
        if self.pre_normalize and self.normalize:
            norm = pc_normalize(sample_points, float(sample_radius)).astype(np.float32)
            return norm * self.normalization_scale
        if self.normalize:
            return sample_points.astype(np.float32) * self.normalization_scale
        return sample_points

    def _build_atom_phase_map(
        self,
        metadata: Dict[str, Any],
        num_atoms: int,
    ) -> np.ndarray:
        """Build efficient atom-to-phase mapping for phase purity checking.

        Returns:
            Array of shape (num_atoms,) where each element is the phase_id string.
            Uses object dtype for variable-length strings.
        """
        # Initialize all atoms as "unknown" phase
        atom_phases = np.full(num_atoms, "unknown", dtype=object)

        # Assign phases from grains
        for grain in metadata.get("grains", []):
            phase_id = grain["base_phase_id"]
            atom_indices = np.array(grain.get("atom_indices", []), dtype=np.int32)
            if len(atom_indices) > 0:
                valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
                atom_phases[atom_indices[valid_mask]] = phase_id

        # Assign phases from intermediate regions (overwrites grain assignments)
        for region in metadata.get("intermediate_regions", []):
            phase_id = region.get("intermediate_phase_id", "intermediate")
            atom_indices = np.array(region.get("atom_indices", []), dtype=np.int32)
            if len(atom_indices) > 0:
                valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
                atom_phases[atom_indices[valid_mask]] = phase_id

        return atom_phases

    def _build_efficient_atom_metadata(
        self,
        metadata: Dict[str, Any],
        env_label: str,
        num_atoms: int,
    ) -> np.ndarray:
        """Build efficient atom metadata using numpy arrays and lazy dict creation.

        Instead of creating num_atoms dictionaries upfront, we use numpy arrays
        to map atoms to grain indices, and only create dicts when queried.
        This is much faster and uses far less memory for millions of atoms.
        """
        default_orientation = np.eye(3, dtype=np.float32)
        default_quaternion = rotation_matrix_to_quaternion(default_orientation)

        # Use numpy arrays for efficient storage - map atom_idx -> grain_id
        atom_to_grain_idx = np.full(num_atoms, -1, dtype=np.int32)  # -1 = unknown

        # Store unique grain metadata
        grain_metadata = {}  # grain_local_idx -> metadata
        grain_key_to_idx = {}  # grain_key -> local index
        next_grain_idx = 0

        # Helper to register a grain
        def register_grain(grain_key, phase_id, orientation, quaternion):
            nonlocal next_grain_idx
            if grain_key not in grain_key_to_idx:
                grain_key_to_idx[grain_key] = next_grain_idx
                grain_metadata[next_grain_idx] = {
                    "phase_id": phase_id,
                    "grain_key": grain_key,
                    "orientation": orientation,
                    "quaternion": quaternion,
                }
                next_grain_idx += 1
            return grain_key_to_idx[grain_key]

        # Register default "unknown" grain
        unknown_key = (env_label, "unknown")
        unknown_idx = register_grain(unknown_key, "unknown", default_orientation, default_quaternion)

        # Process grains
        grain_meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for grain in metadata.get("grains", []):
            grain_id = str(grain["grain_id"])
            orientation = np.asarray(grain["orientation_matrix"], dtype=np.float32)
            quaternion = np.asarray(
                grain.get("orientation_quaternion"),
                dtype=np.float32,
            ) if grain.get("orientation_quaternion") is not None else rotation_matrix_to_quaternion(orientation)
            grain_key = (env_label, grain_id)

            grain_meta_cache[grain_key] = {
                "phase_id": grain["base_phase_id"],
                "orientation": orientation,
                "quaternion": quaternion,
            }

            local_idx = register_grain(grain_key, grain["base_phase_id"], orientation, quaternion)

            # Efficiently assign atoms to this grain using numpy indexing
            atom_indices = np.array(grain.get("atom_indices", []), dtype=np.int32)
            valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
            atom_to_grain_idx[atom_indices[valid_mask]] = local_idx

        # Process intermediate regions
        for region in metadata.get("intermediate_regions", []):
            grain_a = region.get("grain_A_id")
            grain_b = region.get("grain_B_id")
            grain_identifier = f"interface_{grain_a}_{grain_b}"
            parent_identifier = str(grain_a) if grain_a is not None else str(grain_b)
            parent_key = (env_label, parent_identifier)
            parent_meta = grain_meta_cache.get(parent_key, None)
            orientation = parent_meta["orientation"] if parent_meta else default_orientation
            quaternion = parent_meta["quaternion"] if parent_meta else default_quaternion
            grain_key = (env_label, grain_identifier)
            phase_id = region.get("intermediate_phase_id", "intermediate")

            local_idx = register_grain(grain_key, phase_id, orientation, quaternion)

            # Efficiently assign atoms to this region
            atom_indices = np.array(region.get("atom_indices", []), dtype=np.int32)
            valid_mask = (atom_indices >= 0) & (atom_indices < num_atoms)
            atom_to_grain_idx[atom_indices[valid_mask]] = local_idx

        # Create a lookup array that can be indexed directly
        # Each element will be a metadata dict when accessed
        class LazyMetadataArray:
            """Lazy array-like object that creates metadata dicts on access."""
            def __init__(self, atom_to_grain, grain_meta, unknown_idx):
                self.atom_to_grain = atom_to_grain
                self.grain_meta = grain_meta
                self.unknown_idx = unknown_idx

            def __getitem__(self, idx):
                grain_idx = self.atom_to_grain[idx]
                if grain_idx == -1:
                    grain_idx = self.unknown_idx
                return self.grain_meta[grain_idx]

            def __len__(self):
                return len(self.atom_to_grain)

        return LazyMetadataArray(atom_to_grain_idx, grain_metadata, unknown_idx)

    def _build_atom_metadata(
        self,
        metadata: Dict[str, Any],
        env_label: str,
        num_atoms: int,
    ) -> List[Dict[str, Any]]:
        """Legacy method - kept for compatibility but deprecated."""
        default_orientation = np.eye(3, dtype=np.float32)
        default_quaternion = rotation_matrix_to_quaternion(default_orientation)
        atom_meta = [{
            "phase_id": "unknown",
            "grain_key": (env_label, "unknown"),
            "orientation": default_orientation,
            "quaternion": default_quaternion,
        } for _ in range(num_atoms)]

        grain_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for grain in metadata.get("grains", []):
            grain_id = str(grain["grain_id"])
            orientation = np.asarray(grain["orientation_matrix"], dtype=np.float32)
            quaternion = np.asarray(
                grain.get("orientation_quaternion"),
                dtype=np.float32,
            ) if grain.get("orientation_quaternion") is not None else rotation_matrix_to_quaternion(orientation)
            grain_key = (env_label, grain_id)
            grain_meta[grain_key] = {
                "phase_id": grain["base_phase_id"],
                "orientation": orientation,
                "quaternion": quaternion,
            }
            for idx in grain.get("atom_indices", []):
                if 0 <= idx < num_atoms:
                    atom_meta[idx] = {
                        "phase_id": grain["base_phase_id"],
                        "grain_key": grain_key,
                        "orientation": orientation,
                        "quaternion": quaternion,
                    }

        for region in metadata.get("intermediate_regions", []):
            grain_a = region.get("grain_A_id")
            grain_b = region.get("grain_B_id")
            grain_identifier = f"interface_{grain_a}_{grain_b}"
            parent_identifier = str(grain_a) if grain_a is not None else str(grain_b)
            parent_key = (env_label, parent_identifier)
            parent_meta = grain_meta.get(parent_key, None)
            orientation = parent_meta["orientation"] if parent_meta else default_orientation
            quaternion = parent_meta["quaternion"] if parent_meta else default_quaternion
            grain_key = (env_label, grain_identifier)
            phase_id = region.get("intermediate_phase_id", "intermediate")
            for idx in region.get("atom_indices", []):
                if 0 <= idx < num_atoms:
                    atom_meta[idx] = {
                        "phase_id": phase_id,
                        "grain_key": grain_key,
                        "orientation": orientation,
                        "quaternion": quaternion,
                    }

        return atom_meta

    @staticmethod
    def _group_class(class_name: str) -> str:
        """Group amorphous phases (but not intermediate) into one class."""
        if class_name.startswith('amorphous_') and not class_name.startswith('intermediate_'):
            return 'amorphous'
        return class_name

    def _encode_class(self, class_name: str) -> int:
        """Encode class name to integer index."""
        grouped_name = self._group_class(class_name)
        if grouped_name not in self._class_to_idx:
            self._class_to_idx[grouped_name] = len(self._class_to_idx)
        return self._class_to_idx[grouped_name]

    def _encode_instance(self, instance_key: Tuple[str, str]) -> int:
        """Encode instance key to integer index."""
        if instance_key[1] == "unknown":
            return -1
        if instance_key not in self._instance_to_idx:
            self._instance_to_idx[instance_key] = len(self._instance_to_idx)
        return self._instance_to_idx[instance_key]
    
    def _build_class_properties(self) -> None:
        """Build class properties dict with domain-specific info."""
        for class_name in self._class_to_idx.keys():
            if class_name.startswith('crystal_'):
                structure = class_name.replace('crystal_', '')
                is_cubic = structure in ('fcc', 'bcc')
                self._class_properties[class_name] = {
                    "structure": structure,
                    "symmetry": "cubic" if is_cubic else "other",
                    "cubic_symmetric": is_cubic,
                }
            elif class_name == 'amorphous':
                self._class_properties[class_name] = {
                    "structure": "disordered",
                    "symmetry": None,
                    "cubic_symmetric": False,
                }
            else:
                self._class_properties[class_name] = {
                    "structure": "unknown",
                    "symmetry": None,
                    "cubic_symmetric": False,
                }
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Return mapping from class_id to class name."""
        return {v: k for k, v in self._class_to_idx.items()}
    
    @property
    def class_properties(self) -> Dict[str, Dict[str, Any]]:
        """Return class properties dict."""
        return self._class_properties
    
    @property 
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self._class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return sample as dictionary with standardized keys.
        
        Returns:
            Dict with keys:
                - "points": (N, 3) point cloud tensor
                - "class_id": scalar int64 tensor
                - "instance_id": scalar int64 tensor  
                - "rotation": (3, 3) float32 rotation matrix tensor
                - "coords": (3,) float32 sample center coordinates
        """
        # Samples are already stored as tensors for efficiency
        pc_tensor = self.samples[index].clone()
        if not self.pre_normalize and self.normalize:
            point_set = pc_tensor.numpy()
            point_set = pc_normalize(point_set, float(self.sample_radii[index])).astype(np.float32)
            pc_tensor = torch.tensor(point_set, dtype=torch.float32)

        rotation = self._rotations[index].to(dtype=pc_tensor.dtype)
        aug_info: Dict[str, Any] = {}
        did_augment = False

        if self.rotation_scale > 0:
            rot = self._random_rotation_matrix(pc_tensor.device, pc_tensor.dtype)
            pc_tensor = (rot @ pc_tensor.transpose(0, 1)).transpose(0, 1).contiguous()
            rotation = rot @ rotation
            aug_info["rotation"] = rot.cpu().numpy()
            did_augment = True

        if self.scaling_range > 0:
            scale = (torch.rand(1, dtype=pc_tensor.dtype, device=pc_tensor.device) * 2.0 - 1.0) * self.scaling_range + 1.0
            pc_tensor = pc_tensor * scale
            aug_info["scale"] = float(scale.item())
            did_augment = True

        if self.noise_scale > 0:
            pc_tensor = pc_tensor + torch.randn_like(pc_tensor) * self.noise_scale
            aug_info["noise_scale"] = self.noise_scale
            did_augment = True

        if self.jitter_scale > 0:
            jitter = torch.randn_like(pc_tensor) * self.jitter_scale
            pc_tensor = pc_tensor + jitter
            aug_info["jitter_scale"] = self.jitter_scale
            did_augment = True

        if did_augment:
            pc_tensor = pc_tensor - pc_tensor.mean(dim=0, keepdim=True)

        if self._augmentation_metadata is not None:
            self._augmentation_metadata[index] = aug_info

        return {
            "points": pc_tensor,
            "class_id": torch.tensor(self._class_ids[index], dtype=torch.long),
            "instance_id": torch.tensor(self._instance_ids[index], dtype=torch.long),
            "rotation": rotation.to(dtype=torch.float32),
            "coords": self._coords[index].clone(),
        }

    @staticmethod
    def _random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rand_mat = torch.randn(3, 3, device=device, dtype=dtype)
        q, r = torch.linalg.qr(rand_mat)
        d = torch.diagonal(r).sign()
        q *= d.unsqueeze(-1)
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q


class SoapCoordDataset(Dataset):
    """
    PyTorch Dataset over one or more Parquet files that contain:
      • first num_coord_dims columns → Cartesian coords
      • remaining columns            → SOAP features

    Returns (soap, coord) per sample as torch.Tensors.
    """

    def __init__(
        self,
        parquet_paths: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        dtype: torch.dtype = torch.float32,
        preload: bool = True,
    ) -> None:
        # unify to list of Paths
        if isinstance(parquet_paths, (str, Path)):
            self.files = [Path(parquet_paths)]
        else:
            self.files = [Path(p) for p in parquet_paths]
            
        self.num_coord_dims = 3
        self.dtype = dtype
        self.preload = preload

        if preload:
            print("SOAP dataset in preload mode")
            # load & stack everything
            coords_list = []
            soap_list = []
            for p in self.files:
                mat = pd.read_parquet(p, engine="pyarrow").to_numpy()
                if mat.shape[1] <= 3:
                    raise ValueError(
                        f"{p!r} has only {mat.shape[1]} cols but "
                        f"num_coord_dims={3}."
                    )
                coords_list.append(mat[:, :3])
                soap_list.append(mat[:, 3:])
            all_coords = torch.as_tensor(
                np.vstack(coords_list), dtype=dtype
            )
            all_soap   = torch.as_tensor(
                np.vstack(soap_list), dtype=dtype
            )
            self.coords = all_coords
            self.soap   = all_soap
            self._len   = all_coords.shape[0]
        else:
            raise NotImplementedError("Lazy mode not implemented")
   

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < self._len):
            raise IndexError(f"Index {idx} out of range 0..{self._len-1}")

        if self.preload:
            return self.soap[idx], self.coords[idx]
    

if __name__ == '__main__':
    data = PointCloudDataset(root="datasets/Al/inherent_configurations_off",
                         data_files=["240ps.npy"],
                         n_samples=6000,
                         num_points=200)
    
    loader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for i, (point, label) in enumerate(loader):
        print(point.shape)
        print(label.shape)
        if i > 3:
            break
        
    paths = ["datasets/Al/soap_features/166ps_selected.parquet",
             "datasets/Al/soap_features/170ps_selected.parquet",
             "datasets/Al/soap_features/174ps_selected.parquet",
             "datasets/Al/soap_features/175ps_selected.parquet",
             "datasets/Al/soap_features/177ps_selected.parquet"]


    ds = SoapCoordDataset(
        parquet_paths=paths,
        num_coord_dims=3,
        preload=True,     
    )
    print("SoapCoordDataset length: ", len(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, pin_memory=True)

    for i, (soap, xyz) in enumerate(loader):
        print(soap.shape)
        print(xyz.shape)
        if i > 3:
            break
