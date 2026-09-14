"""Consume generated point clouds with explicit class and source metadata."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset

from src.data.sampling import (
    get_random_samples,
    get_regular_samples,
    pc_normalize,
)
from src.data.static_sources import (
    estimate_source_cutoff_radius,
    resolve_auto_cutoff_config,
)
from src.utils.logging_config import setup_logging

logger = setup_logging()


class SyntheticPointCloudDataset(Dataset):
    """Dataset for synthetic atomistic point clouds with metadata labels.

    Optimized for millions of atoms by using lazy metadata computation
    and pre-converted tensors.
    
    Returns dict with keys:
        - "points": (N, 3) point cloud tensor
        - "class_id": scalar int64 tensor (category/phase index)
        - "instance_id": scalar int64 tensor (grain/instance index)
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
        self.auto_cutoff_config = resolve_auto_cutoff_config(
            auto_cutoff_config,
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
        atom_table_path = env_path / "atoms_full.npy"
        metadata_path = env_path / "metadata.json"
        phase_mapping_path = env_path / "phase_mapping.json"
        if not atoms_path.exists():
            raise FileNotFoundError(f"atoms.npy missing in {env_path}")
        if not atom_table_path.exists():
            raise FileNotFoundError(f"atoms_full.npy missing in {env_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json missing in {env_path}")
        if not phase_mapping_path.exists():
            raise FileNotFoundError(f"phase_mapping.json missing in {env_path}")

        points = np.load(atoms_path)
        atom_table = np.load(atom_table_path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"atoms.npy at {atoms_path} must have shape (N, 3)")
        if atom_table.shape != (points.shape[0],):
            raise ValueError(
                f"atoms_full.npy at {atom_table_path} must have shape ({points.shape[0]},), "
                f"got {atom_table.shape}."
            )
        if not np.array_equal(points, atom_table["position"]):
            raise ValueError(
                "atoms.npy and atoms_full.npy contain different positions. "
                f"env_path={env_path}."
            )

        with metadata_path.open("r") as handle:
            metadata = json.load(handle)
        with phase_mapping_path.open("r") as handle:
            phase_mapping = json.load(handle)
        phase_id_to_name = phase_mapping["id_to_name"]

        env_label = env_path.name or f"env_{env_index}"
        schema_version = metadata["schema_version"]
        if schema_version != 3:
            raise ValueError(
                "SyntheticPointCloudDataset only consumes the repository-owned atomistic "
                "metadata schema version 3. "
                f"env_path={env_path}, schema_version={schema_version!r}."
            )
        if metadata["environment_name"] != env_label:
            raise ValueError(
                "Synthetic environment directory name does not match metadata.environment_name. "
                f"env_path={env_path}, directory_name={env_label!r}, "
                f"metadata_environment_name={metadata['environment_name']!r}."
            )
        env_radius = self._resolve_environment_cutoff_radius(
            env_path=env_path,
            env_label=env_label,
            env_index=env_index,
        )
        self.source_radii[env_label] = env_radius

        position_tree = cKDTree(points)
        atom_phase_ids = atom_table["phase_id"]
        atom_grain_ids = atom_table["grain_id"]
        atom_orientations = atom_table["orientation"].reshape(-1, 3, 3)

        samples = self._sample_points(points, env_radius)
        if not samples:
            raise RuntimeError(
                "SyntheticPointCloudDataset produced zero samples for an environment. "
                f"env_label={env_label!r}, env_path={env_path}, num_atoms={int(points.shape[0])}, "
                f"sample_type={self.sample_type!r}, radius={env_radius}, "
                f"n_samples={self.n_samples}, num_points={self.num_points}."
            )

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
                    sample_phases = atom_phase_ids[atom_indices]
                    unique_phases = np.unique(sample_phases)
                    # Discard if multiple phases are present
                    if len(unique_phases) > 1:
                        discarded_mixed_phase += 1
                        continue

            phase_name = phase_id_to_name[str(int(atom_phase_ids[atom_idx]))]

            # Filter by allowed classes if specified
            if self.allowed_classes is not None:
                if phase_name not in self.allowed_classes:
                    continue

            processed = self._prepare_sample(sample_points, env_radius)
            class_idx = self._encode_class(phase_name)
            instance_idx = self._encode_instance(
                (env_label, str(int(atom_grain_ids[atom_idx])))
            )

            # Store as tensors to avoid conversion overhead in __getitem__
            self.samples.append(torch.tensor(processed, dtype=torch.float32))
            self.sample_radii.append(float(env_radius))
            self.sample_source_names.append(env_label)
            self._class_ids.append(class_idx)
            self._instance_ids.append(instance_idx)
            self._rotations.append(
                torch.tensor(atom_orientations[atom_idx], dtype=torch.float32)
            )
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
        estimated_radius, coverage = estimate_source_cutoff_radius(
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
