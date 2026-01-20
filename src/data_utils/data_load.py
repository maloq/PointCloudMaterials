import sys,os
sys.path.append(os.getcwd())
import json
import hashlib
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
    """Convert rotation matrix to quaternion (w, x, y, z)."""
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


def _get_fps_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _farthest_point_sample_torch(points: torch.Tensor, npoint: int, device: torch.device) -> torch.Tensor:
    """Fast FPS using torch (matches convert_to_fast_modelnet.py behavior)."""
    if points.shape[0] <= npoint:
        return points
    xyz = points[:, :3].to(device)
    n_points = xyz.shape[0]
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.full((n_points,), 1e10, device=device)
    farthest = torch.randint(0, n_points, (1,), device=device).item()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest:farthest + 1]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance).item()
    indices = centroids.cpu()
    return points[indices]

import trimesh

def read_and_sample_mesh(filename, n_points=2048):
    """
    Loads OFF file as a mesh and samples points from the SURFACE.
    """
    try:
        # Trimesh handles OFF files and face logic automatically
        mesh = trimesh.load(filename)
        
        # Some ModelNet files are "scenes" or broken; force into a single mesh
        if isinstance(mesh, trimesh.Scene):
            # Concatenate all geometries in the scene
            if len(mesh.geometry) == 0:
                return np.zeros((n_points, 3))
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                      for g in mesh.geometry.values())
            )
            
        # SAMPLE points from the surface (weighted by triangle area)
        # This fills in the wings!
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        
        return np.array(points, dtype=np.float32)
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        # Fallback to zeros or handle error
        return np.zeros((n_points, 3), dtype=np.float32)
        

def read_and_sample_off_file(root, data_files, radius, n_points, overlap_fraction, sample_type, n_samples, return_coords, sampling_method="drop_farthest"):
    """
    Read an OFF file and sample points from it.
    """
    samples = []
    for off_file in data_files:
        logger.debug(f"Reading {off_file}")
        points = read_off_file(os.path.join(root, off_file), verbose=False)
        if sample_type == 'regular':
            samples = get_regular_samples(points,
                                            size=radius,
                                            n_points=n_points,
                                            overlap_fraction=overlap_fraction,
                                            return_coords=return_coords,
                                            sampling_method=sampling_method)
        elif sample_type == 'random':
            samples = get_random_samples(points,
                                            n_samples=n_samples,
                                            size=radius,
                                            n_points=n_points,
                                            return_coords=return_coords,
                                            sampling_method=sampling_method)
        else:
            raise ValueError(f"Invalid sample type: {sample_type}")
    if samples:
        return samples
    else:
        raise ValueError(f"No samples found for {data_files}")


class PointCloudDataset(Dataset):
    def __init__(self,
                 root: str,
                 data_files: list[str],
                 return_coords=False,
                 sample_type='regular',
                 radius=8,
                 overlap_fraction=0.0,
                 n_samples=1000,
                 num_points=100,
                 pre_normalize=True,
                 normalize=True,
                 sampling_method="drop_farthest"):
        """Initialize the dataset with samples from OFF files.
        Args:
            root: Path to directory containing OFF files
        """
        self.root = root
        self.pre_normalize = pre_normalize
        self.normalize = normalize
        self.return_coords = return_coords
        self.radius = radius
        self.samples = read_and_sample_off_file(root,
                                                data_files,
                                                radius,
                                                num_points,
                                                overlap_fraction,
                                                sample_type,
                                                n_samples,
                                                return_coords,
                                                sampling_method)
        if self.return_coords:
            self.samples, self.coords = zip(*self.samples)
            self.samples = list(self.samples)
        else:
            self.coords = None

        if pre_normalize and normalize:
            self.samples = [pc_normalize(s, self.radius).astype(np.float32) for s in self.samples]
        elif not normalize:
            print("Point Cloud normalization skipped")
        logger.info(f"Point set shape: {self.samples[0].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_set = self.samples[index]
        if not self.pre_normalize and self.normalize:
            point_set = pc_normalize(point_set, self.radius).astype(np.float32)
        point_set_tensor = torch.tensor(point_set, dtype=torch.float32)
        
        if self.return_coords:
            return point_set_tensor, self.coords[index]
        else:
            return point_set_tensor

 
class SyntheticPointCloudDataset(Dataset):
    """Dataset for synthetic atomistic point clouds with metadata labels.

    Optimized for millions of atoms by using lazy metadata computation
    and pre-converted tensors.
    """

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
    ) -> None:
        super().__init__()
        if not env_dirs:
            raise ValueError("env_dirs must contain at least one synthetic dataset directory")
        self.radius = radius
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


        # Store as tensors to avoid conversion overhead
        self.samples: List[torch.Tensor] = []
        self._phase_labels: List[int] = []
        self._grain_labels: List[int] = []
        self._orientation: List[torch.Tensor] = []
        self._quaternions: List[torch.Tensor] = []

        self._phase_to_idx: Dict[str, int] = {}
        self._grain_to_idx: Dict[Tuple[str, str], int] = {}

        for env_index, env_dir in enumerate(env_dirs):
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break
            self._ingest_environment(env_dir, env_index)

        if not self.samples:
            raise RuntimeError("SyntheticPointCloudDataset constructed with zero samples")

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

        # Build efficient metadata lookup instead of storing for all atoms
        atom_to_meta_idx = self._build_efficient_atom_metadata(metadata, env_label, points.shape[0])
        position_tree = cKDTree(points)

        # Build atom-to-phase mapping if needed for phase purity checking
        atom_phases = None
        if self.discard_mixed_phase:
            atom_phases = self._build_atom_phase_map(metadata, points.shape[0])

        samples = self._sample_points(points)
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
                atom_indices = position_tree.query_ball_point(center, self.radius)
                if len(atom_indices) > 0:
                    sample_phases = atom_phases[atom_indices]
                    unique_phases = np.unique(sample_phases)
                    # Discard if multiple phases are present
                    if len(unique_phases) > 1:
                        discarded_mixed_phase += 1
                        continue

            # Look up metadata on-demand
            meta = atom_to_meta_idx[atom_idx]
            processed = self._prepare_sample(sample_points)
            phase_idx = self._encode_phase(meta["phase_id"])
            grain_idx = self._encode_grain(meta["grain_key"])

            # Store as tensors to avoid conversion overhead in __getitem__
            self.samples.append(torch.tensor(processed, dtype=torch.float32))
            self._phase_labels.append(phase_idx)
            self._grain_labels.append(grain_idx)
            self._orientation.append(torch.tensor(meta["orientation"], dtype=torch.float32))
            self._quaternions.append(torch.tensor(meta["quaternion"], dtype=torch.float32))

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

    def _sample_points(self, points: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.sample_type == "regular":
            max_samples = self.n_samples if self.n_samples > 0 else int(2e9)
            raw = get_regular_samples(
                points,
                size=self.radius,
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
                size=self.radius,
                n_points=self.num_points,
                return_coords=True,
                sampling_method=self.sampling_method,
            )
        else:
            raise ValueError(f"Invalid sample type: {self.sample_type!r}")
        return [(np.asarray(s, dtype=np.float32), np.asarray(c, dtype=np.float32)) for s, c in raw]

    def _prepare_sample(self, sample_points: np.ndarray) -> np.ndarray:
        if self.pre_normalize and self.normalize:
            return pc_normalize(sample_points, self.radius).astype(np.float32)
        if self.normalize:
            return sample_points.astype(np.float32)
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
    def _group_phase(phase_id: str) -> str:
        """Group amorphous phases (but not intermediate) into one class."""
        if phase_id.startswith('amorphous_') and not phase_id.startswith('intermediate_'):
            return 'amorphous'
        return phase_id

    def _encode_phase(self, phase_id: str) -> int:
        # Apply phase grouping
        grouped_phase_id = self._group_phase(phase_id)

        if grouped_phase_id not in self._phase_to_idx:
            self._phase_to_idx[grouped_phase_id] = len(self._phase_to_idx)
        return self._phase_to_idx[grouped_phase_id]

    def _encode_grain(self, grain_key: Tuple[str, str]) -> int:
        if grain_key[1] == "unknown":
            return -1
        if grain_key not in self._grain_to_idx:
            self._grain_to_idx[grain_key] = len(self._grain_to_idx)
        return self._grain_to_idx[grain_key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        # Samples are already stored as tensors for efficiency
        pc_tensor = self.samples[index]
        if not self.pre_normalize and self.normalize:
            # Need to convert back to numpy for normalization, then back to tensor
            point_set = pc_tensor.numpy()
            point_set = pc_normalize(point_set, self.radius).astype(np.float32)
            pc_tensor = torch.tensor(point_set, dtype=torch.float32)

        # Use cached tensors for orientation and quaternion
        phase_tensor = torch.tensor(self._phase_labels[index], dtype=torch.long)
        grain_tensor = torch.tensor(self._grain_labels[index], dtype=torch.long)
        orientation_tensor = self._orientation[index]
        quaternion_tensor = self._quaternions[index]
        return pc_tensor, phase_tensor, grain_tensor, orientation_tensor, quaternion_tensor


class CenteredModelNetDataset(Dataset):
    """ModelNet40 wrapper that returns centered objects with synthetic-like metadata.

    Note: when add_center_point=True, num_points includes the center point.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        split: str = "train",
        classes: Optional[Sequence[str]] = None,
        num_points: int = 1024,
        add_center_point: bool = True,
        pre_normalize: bool = True,
        normalize: bool = False,
        max_samples: Optional[int] = None,
        fps_cache: bool = True,
        fps_cache_dir: Optional[Union[str, Path]] = None,
        fps_use_gpu: bool = True,
        rotation_scale: float = 0.0,
        noise_scale: float = 0.0,
        jitter_scale: float = 0.0,
        scaling_range: float = 0.0,
        track_augmentation: bool = False,
    ) -> None:
        super().__init__()
        if num_points <= 0:
            raise ValueError("num_points must be > 0 for CenteredModelNetDataset")

        self.root_dir = Path(root_dir)
        self.split = split
        self.classes = list(classes) if classes is not None else None
        self.num_points = int(num_points)
        self.add_center_point = bool(add_center_point)
        self.pre_normalize = bool(pre_normalize)
        self.normalize = bool(normalize)
        self.max_samples = max_samples if max_samples is not None and max_samples > 0 else None
        self.fps_cache = bool(fps_cache)
        self.fps_cache_dir = Path(fps_cache_dir) if fps_cache_dir else (self.root_dir / "fps_cache")
        self.fps_use_gpu = bool(fps_use_gpu)

        # Augmentation params (applied in __getitem__)
        self.rotation_scale = float(rotation_scale)
        self.noise_scale = float(noise_scale)
        self.jitter_scale = float(jitter_scale)
        self.scaling_range = float(scaling_range)
        self.track_augmentation = bool(track_augmentation)
        self._augmentation_metadata: Optional[List[Dict[str, Any]]] = None

        if self.add_center_point:
            if self.num_points < 2:
                raise ValueError("num_points must be >= 2 when add_center_point=True")
            self.num_surface_points = self.num_points - 1
        else:
            self.num_surface_points = self.num_points

        # Lazy import to avoid loading unless needed
        from src.data_utils.modelnet_fast_loader import ModelNetFastDataset

        if self.split == "all":
            train_ds = ModelNetFastDataset(
                root_dir=str(self.root_dir),
                split="train",
                classes=self.classes,
                n_points=self.num_surface_points,
            )
            test_ds = ModelNetFastDataset(
                root_dir=str(self.root_dir),
                split="test",
                classes=self.classes,
                n_points=self.num_surface_points,
            )
            points = torch.cat([train_ds.points, test_ds.points], dim=0)
            labels = torch.cat([train_ds.labels, test_ds.labels], dim=0)
            class_names = train_ds.class_names + test_ds.class_names
            class_to_idx = train_ds.class_to_idx
        else:
            base_ds = ModelNetFastDataset(
                root_dir=str(self.root_dir),
                split=self.split,
                classes=self.classes,
                n_points=self.num_surface_points,
            )
            points = base_ds.points
            labels = base_ds.labels
            class_names = base_ds.class_names
            class_to_idx = base_ds.class_to_idx

        if len(points) == 0:
            raise RuntimeError("CenteredModelNetDataset constructed with zero samples")

        self._phase_to_idx = dict(class_to_idx)
        if self.classes is not None:
            class_set = set(self.classes)
            self._phase_to_idx = {name: idx for name, idx in self._phase_to_idx.items() if name in class_set}
        self._grain_to_idx: Dict[Tuple[str, str], int] = {}
        source_points = points.shape[1]
        if self.num_surface_points != source_points:
            points, labels, class_names = self._load_or_build_fps_cache(
                points, labels, class_names, source_points
            )

        if self.max_samples is not None:
            max_samples = min(self.max_samples, len(points))
            points = points[:max_samples]
            labels = labels[:max_samples]
            class_names = class_names[:max_samples]

        if self.pre_normalize and self.normalize:
            points = self._normalize_unit_sphere(points)

        self.points = points.contiguous()
        self.class_names = class_names
        self._phase_labels = labels.to(dtype=torch.long)
        self._grain_labels = torch.full((len(self._phase_labels),), -1, dtype=torch.long)

        if self.track_augmentation:
            self._augmentation_metadata = [None] * len(self.points)

    @staticmethod
    def _normalize_unit_sphere(points: torch.Tensor) -> torch.Tensor:
        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        max_dist = torch.max(torch.linalg.norm(centered, dim=2), dim=1, keepdim=True)[0]
        max_dist = torch.clamp(max_dist, min=1e-6)
        return centered / max_dist

    def _cache_key(self, source_points: int) -> str:
        classes_key = "all" if self.classes is None else ",".join(sorted(self.classes))
        classes_hash = hashlib.md5(classes_key.encode("utf-8")).hexdigest()[:8]
        return f"modelnet_{self.split}_fps{self.num_surface_points}_src{source_points}_{classes_hash}.pt"

    def _load_or_build_fps_cache(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        class_names: List[str],
        source_points: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        if not self.fps_cache:
            return self._fps_downsample(points), labels, class_names

        self.fps_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.fps_cache_dir / self._cache_key(source_points)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu")
            if (
                cached.get("num_points") == self.num_surface_points
                and cached.get("source_points") == source_points
            ):
                return cached["points"], cached["labels"], cached["class_names"]

        logger.print(
            f"FPS downsampling ModelNet ({self.split}) to {self.num_surface_points} points..."
        )
        downsampled = self._fps_downsample(points)
        payload = {
            "points": downsampled,
            "labels": labels,
            "class_names": class_names,
            "class_to_idx": self._phase_to_idx,
            "num_points": self.num_surface_points,
            "source_points": source_points,
        }
        torch.save(payload, cache_path)
        logger.print(f"Saved FPS cache: {cache_path}")
        return downsampled, labels, class_names

    def _fps_downsample(self, points: torch.Tensor) -> torch.Tensor:
        target_points = self.num_surface_points
        if points.shape[1] < target_points:
            return self._upsample(points, target_points)
        if points.shape[1] == target_points:
            return points

        device = _get_fps_device(self.fps_use_gpu)
        downsampled = torch.empty(
            (points.shape[0], target_points, 3),
            dtype=points.dtype,
        )
        for idx in range(points.shape[0]):
            downsampled[idx] = _farthest_point_sample_torch(points[idx], target_points, device)
            if idx > 0 and idx % 200 == 0:
                logger.print(f"  FPS progress: {idx}/{points.shape[0]}")
        return downsampled

    @staticmethod
    def _upsample(points: torch.Tensor, target_points: int) -> torch.Tensor:
        n_samples, curr_points, _ = points.shape
        if curr_points == 0:
            return torch.zeros((n_samples, target_points, 3), dtype=points.dtype)
        idx = torch.randint(0, curr_points, (n_samples, target_points))
        batch_idx = torch.arange(n_samples).unsqueeze(1).expand(-1, target_points)
        return points[batch_idx, idx]

    @staticmethod
    def _random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rand_mat = torch.randn(3, 3, device=device, dtype=dtype)
        q, r = torch.linalg.qr(rand_mat)
        d = torch.diagonal(r).sign()
        q *= d.unsqueeze(-1)
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q

    def get_augmentation_info(self, index: int) -> Optional[Dict[str, Any]]:
        if self._augmentation_metadata is None:
            return None
        return self._augmentation_metadata[index]

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int):
        pc = self.points[index].clone()
        if not self.pre_normalize and self.normalize:
            pc = self._normalize_unit_sphere(pc.unsqueeze(0)).squeeze(0)

        orientation = torch.eye(3, dtype=pc.dtype, device=pc.device)
        aug_info: Dict[str, Any] = {}

        if self.rotation_scale > 0:
            rot = self._random_rotation_matrix(pc.device, pc.dtype)
            pc = (rot @ pc.transpose(0, 1)).transpose(0, 1).contiguous()
            orientation = rot
            aug_info["rotation"] = rot.cpu().numpy()

        if self.scaling_range > 0:
            scale = (torch.rand(1, dtype=pc.dtype, device=pc.device) * 2.0 - 1.0) * self.scaling_range + 1.0
            pc = pc * scale
            aug_info["scale"] = float(scale.item())

        if self.noise_scale > 0:
            pc = pc + torch.randn_like(pc) * self.noise_scale
            aug_info["noise_scale"] = self.noise_scale

        if self.jitter_scale > 0:
            jitter = torch.randn_like(pc) * self.jitter_scale
            pc = pc + jitter
            aug_info["jitter_scale"] = self.jitter_scale

        pc = pc - pc.mean(dim=0, keepdim=True)

        if self.add_center_point:
            center = torch.zeros((1, 3), dtype=pc.dtype, device=pc.device)
            pc = torch.cat([pc, center], dim=0)

        if self._augmentation_metadata is not None:
            self._augmentation_metadata[index] = aug_info

        phase_tensor = self._phase_labels[index]
        grain_tensor = self._grain_labels[index]
        quaternion = rotation_matrix_to_quaternion(orientation.cpu().numpy())
        orientation_tensor = orientation.to(dtype=torch.float32)
        quaternion_tensor = torch.tensor(quaternion, dtype=torch.float32)
        return pc, phase_tensor, grain_tensor, orientation_tensor, quaternion_tensor


class CurriculumLearningDataset(Dataset):
    """Wrapper dataset that filters samples based on amorphous fraction for curriculum learning.

    This dataset wraps a SyntheticPointCloudDataset and dynamically filters samples to include
    only a specified fraction of amorphous samples. This enables curriculum learning where
    the model starts training with only crystalline samples and gradually introduces amorphous ones.

    Args:
        dataset: The underlying SyntheticPointCloudDataset (or Subset of one)
        amorphous_fraction: Fraction of amorphous samples to include (0.0 to 1.0)
    """

    def __init__(self, dataset, amorphous_fraction: float = 1.0):
        self.dataset = dataset
        self._amorphous_fraction = amorphous_fraction

        # Handle Subset objects from train/val split
        self.base_dataset = dataset
        self.subset_indices = None
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # This is a Subset object from random_split
            self.base_dataset = dataset.dataset
            self.subset_indices = dataset.indices

        self._update_indices()

    def _update_indices(self):
        """Update the list of valid indices based on current amorphous fraction."""
        crystalline_indices = []
        amorphous_indices = []

        # Determine which indices to iterate over
        if self.subset_indices is not None:
            # We're wrapping a Subset - only consider indices in the subset
            indices_to_check = self.subset_indices
        else:
            # We're wrapping the full dataset
            indices_to_check = range(len(self.base_dataset))

        # Separate indices by phase type
        for idx in indices_to_check:
            phase_label = self.base_dataset._phase_labels[idx]
            # Get the phase name from the index
            phase_name = None
            for name, phase_idx in self.base_dataset._phase_to_idx.items():
                if phase_idx == phase_label:
                    phase_name = name
                    break

            # Check if this is an amorphous phase
            if phase_name and phase_name == 'amorphous':
                amorphous_indices.append(idx)
            else:
                crystalline_indices.append(idx)

        # Calculate how many amorphous samples to include
        num_amorphous_to_include = int(len(amorphous_indices) * self._amorphous_fraction)

        # Combine indices: all crystalline + fraction of amorphous
        self._valid_indices = crystalline_indices + amorphous_indices[:num_amorphous_to_include]

        logger.print(
            f"CurriculumLearning: amorphous_fraction={self._amorphous_fraction:.3f}, "
            f"crystalline={len(crystalline_indices)}, "
            f"amorphous={num_amorphous_to_include}/{len(amorphous_indices)}, "
            f"total={len(self._valid_indices)}"
        )

    @property
    def amorphous_fraction(self) -> float:
        """Get current amorphous fraction."""
        return self._amorphous_fraction

    @amorphous_fraction.setter
    def amorphous_fraction(self, value: float):
        """Set amorphous fraction and update indices."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"amorphous_fraction must be in [0, 1], got {value}")
        if value != self._amorphous_fraction:
            self._amorphous_fraction = value
            self._update_indices()

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int):
        # Map the index to the actual dataset index
        actual_idx = self._valid_indices[index]
        return self.base_dataset[actual_idx]


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
                         data_files=["240ps.off"],
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
