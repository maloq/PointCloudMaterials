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
    
    Returns dict with keys:
        - "points": (N, 3) point cloud tensor
        - "class_id": scalar int64 tensor (category/phase index)
        - "instance_id": scalar int64 tensor (grain/instance index, -1 if unknown)
        - "rotation": (3, 3) float32 rotation matrix tensor
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
        track_augmentation: bool = False,
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
        self.rotation_scale = float(rotation_scale)
        self.noise_scale = float(noise_scale)
        self.jitter_scale = float(jitter_scale)
        self.scaling_range = float(scaling_range)
        self.track_augmentation = bool(track_augmentation)
        self._augmentation_metadata: Optional[List[Dict[str, Any]]] = None

        # Store as tensors to avoid conversion overhead
        self.samples: List[torch.Tensor] = []
        self._class_ids: List[int] = []
        self._instance_ids: List[int] = []
        self._rotations: List[torch.Tensor] = []

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
            class_idx = self._encode_class(meta["phase_id"])
            instance_idx = self._encode_instance(meta["grain_key"])

            # Store as tensors to avoid conversion overhead in __getitem__
            self.samples.append(torch.tensor(processed, dtype=torch.float32))
            self._class_ids.append(class_idx)
            self._instance_ids.append(instance_idx)
            self._rotations.append(torch.tensor(meta["orientation"], dtype=torch.float32))

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
        """
        # Samples are already stored as tensors for efficiency
        pc_tensor = self.samples[index].clone()
        if not self.pre_normalize and self.normalize:
            point_set = pc_tensor.numpy()
            point_set = pc_normalize(point_set, self.radius).astype(np.float32)
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


class CenteredModelNetDataset(Dataset):
    """ModelNet40 wrapper that returns centered objects with standardized metadata.

    Note: when add_center_point=True, num_points includes the center point.
    
    Returns dict with keys:
        - "points": (N, 3) point cloud tensor
        - "class_id": scalar int64 tensor (object class index)
        - "instance_id": scalar int64 tensor (-1, not used for ModelNet)
        - "rotation": (3, 3) float32 rotation matrix tensor (augmentation rotation)
    """

    # Dataset metadata
    domain: str = "objects"

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
            sample_class_names = train_ds.class_names + test_ds.class_names
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
            sample_class_names = base_ds.class_names
            class_to_idx = base_ds.class_to_idx

        if len(points) == 0:
            raise RuntimeError("CenteredModelNetDataset constructed with zero samples")

        self._class_to_idx = dict(class_to_idx)
        if self.classes is not None:
            class_set = set(self.classes)
            self._class_to_idx = {name: idx for name, idx in self._class_to_idx.items() if name in class_set}
        
        source_points = points.shape[1]
        if self.num_surface_points != source_points:
            points, labels, sample_class_names = self._load_or_build_fps_cache(
                points, labels, sample_class_names, source_points
            )

        if self.max_samples is not None:
            max_samples = min(self.max_samples, len(points))
            points = points[:max_samples]
            labels = labels[:max_samples]
            sample_class_names = sample_class_names[:max_samples]

        if self.pre_normalize and self.normalize:
            points = self._normalize_unit_sphere(points)

        self.points = points.contiguous()
        self._sample_class_names = sample_class_names  # per-sample class name list
        self._class_ids = labels.to(dtype=torch.long)
        self._instance_ids = torch.full((len(self._class_ids),), -1, dtype=torch.long)
        
        # Build class properties
        self._class_properties: Dict[str, Dict[str, Any]] = {}
        for class_name in self._class_to_idx.keys():
            self._class_properties[class_name] = {
                "source": "modelnet40",
                "axis_symmetric": class_name in ("vase", "bottle", "cone", "cup"),
            }

        if self.track_augmentation:
            self._augmentation_metadata = [None] * len(self.points)

    @staticmethod
    def _normalize_unit_sphere(points: torch.Tensor) -> torch.Tensor:
        centroid = points.mean(dim=1, keepdim=True)
        centered = points - centroid
        max_dist = torch.max(torch.linalg.norm(centered, dim=2), dim=1, keepdim=True)[0].unsqueeze(-1)
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
            "class_to_idx": self._class_to_idx,
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
        return len(self.points)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return sample as dictionary with standardized keys.
        
        Returns:
            Dict with keys:
                - "points": (N, 3) point cloud tensor
                - "class_id": scalar int64 tensor
                - "instance_id": scalar int64 tensor (-1 for ModelNet)
                - "rotation": (3, 3) float32 rotation matrix tensor
        """
        pc = self.points[index].clone()
        if not self.pre_normalize and self.normalize:
            pc = self._normalize_unit_sphere(pc.unsqueeze(0)).squeeze(0)

        rotation = torch.eye(3, dtype=pc.dtype, device=pc.device)
        aug_info: Dict[str, Any] = {}

        if self.rotation_scale > 0:
            rot = self._random_rotation_matrix(pc.device, pc.dtype)
            pc = (rot @ pc.transpose(0, 1)).transpose(0, 1).contiguous()
            rotation = rot
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

        return {
            "points": pc,
            "class_id": self._class_ids[index],
            "instance_id": self._instance_ids[index],
            "rotation": rotation.to(dtype=torch.float32),
        }


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
