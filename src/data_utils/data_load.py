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


def read_and_sample_off_file(root, data_files, radius, n_points, overlap_fraction, sample_type, n_samples, return_coords):
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
                                            return_coords=return_coords)
        elif sample_type == 'random':
            samples = get_random_samples(points,
                                            n_samples=n_samples,
                                            size=radius,
                                            n_points=n_points,
                                            return_coords=return_coords)
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
                 normalize=True ):
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
                                                return_coords)
        if self.return_coords:
            self.samples, self.coords = zip(*self.samples)
            self.samples = list(self.samples)
        else:
            self.coords = None

        if pre_normalize and normalize:
            self.samples = [pc_normalize(s, self.radius).astype(np.float32) for s in self.samples]
        elif not normalize:
            print("Point Cloud normalization skipped")
        # logger.info(f"Point set shape: {self.samples[0].shape}")

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
    """Dataset for synthetic atomistic point clouds with metadata labels."""

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

        self.samples: List[np.ndarray] = []
        self._phase_labels: List[int] = []
        self._grain_labels: List[int] = []
        self._orientation: List[np.ndarray] = []
        self._quaternions: List[np.ndarray] = []

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
        atom_meta = self._build_atom_metadata(metadata, env_label, points.shape[0])
        position_tree = cKDTree(points)

        samples = self._sample_points(points)
        if not samples:
            logger.print(f"No samples generated for {env_label}")
            return

        for sample_points, center in samples:
            center = np.asarray(center, dtype=np.float64)
            _, idx = position_tree.query(center.reshape(1, -1), k=1)
            atom_idx = int(idx[0])
            meta = atom_meta[atom_idx]
            processed = self._prepare_sample(sample_points)
            phase_idx = self._encode_phase(meta["phase_id"])
            grain_idx = self._encode_grain(meta["grain_key"])
            self.samples.append(processed)
            self._phase_labels.append(phase_idx)
            self._grain_labels.append(grain_idx)
            self._orientation.append(meta["orientation"])
            self._quaternions.append(meta["quaternion"])

            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break

        logger.print(
            f"Ingested {len(samples)} samples from {env_label}; "
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

    def _build_atom_metadata(
        self,
        metadata: Dict[str, Any],
        env_label: str,
        num_atoms: int,
    ) -> List[Dict[str, Any]]:
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

    def _encode_phase(self, phase_id: str) -> int:
        if phase_id not in self._phase_to_idx:
            self._phase_to_idx[phase_id] = len(self._phase_to_idx)
        return self._phase_to_idx[phase_id]

    def _encode_grain(self, grain_key: Tuple[str, str]) -> int:
        if grain_key[1] == "unknown":
            return -1
        if grain_key not in self._grain_to_idx:
            self._grain_to_idx[grain_key] = len(self._grain_to_idx)
        return self._grain_to_idx[grain_key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        point_set = self.samples[index]
        if not self.pre_normalize and self.normalize:
            point_set = pc_normalize(point_set, self.radius).astype(np.float32)
        pc_tensor = torch.tensor(point_set, dtype=torch.float32)
        phase_tensor = torch.tensor(self._phase_labels[index], dtype=torch.long)
        grain_tensor = torch.tensor(self._grain_labels[index], dtype=torch.long)
        orientation_tensor = torch.tensor(self._orientation[index], dtype=torch.float32)
        quaternion_tensor = torch.tensor(self._quaternions[index], dtype=torch.float32)
        return pc_tensor, phase_tensor, grain_tensor, orientation_tensor, quaternion_tensor
 
 
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
