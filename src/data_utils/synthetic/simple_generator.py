"""
Simple Synthetic Atomistic Dataset Generator
Generates simple geometric shapes filled with atoms instead of complex crystal structures.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy.spatial import cKDTree

# Import configuration classes from the main generator to maintain compatibility
import yaml
import torch

# ---------------------------------------------------------------------------
# Configuration Dataclasses (Decoupled)
# ---------------------------------------------------------------------------

@dataclass
class SimplePhaseConfig:
    """Configuration for a single phase in simple generator."""
    name: str
    phase_type: str
    structural_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimpleGrainAssignmentConfig:
    """Configuration for grain-to-phase assignment."""
    mode: str
    assignments: Optional[List[str]] = None
    probabilities: Optional[Dict[str, float]] = None

@dataclass
class SimpleGlobalConfig:
    """Global configuration for simple generator."""
    L: float
    rho_target: float
    avg_nn_dist: float
    grain_count: int
    objects_per_grain: int = 20
    points_per_object: int = 1000
    collision_safety_factor: float = 4.2
    random_seed: int = 0
    data_path: pathlib.Path = field(default_factory=lambda: pathlib.Path("output/simple_data"))
    modelnet_path: pathlib.Path = field(default_factory=lambda: pathlib.Path("datasets/ModelNet40_fast"))

# ---------------------------------------------------------------------------
# Structured Array Dtype for Atoms (Copied to ensure compatibility)
# ---------------------------------------------------------------------------

ATOM_DTYPE = np.dtype([
    ('position', np.float32, (3,)),
    ('phase_id', np.int32),
    ('grain_id', np.int32),
    ('orientation', np.float32, (9,)),
    ('alive', np.bool_),
    ('pre_index', np.int64),
    ('object_id', np.int32),
])

PHASE_ID_MAP: Dict[str, int] = {}
PHASE_NAME_MAP: Dict[int, str] = {}

def load_simple_config(path: str | pathlib.Path) -> Tuple[SimpleGlobalConfig, Dict[str, SimplePhaseConfig], SimpleGrainAssignmentConfig]:
    config_path = pathlib.Path(path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
        
    global_raw = raw.get("global", {})
    if "L" not in global_raw:
        raise ValueError("Global configuration must specify box side length 'L'")
    
    data_path = pathlib.Path(global_raw.get("output_dir", "output/simple_data"))
    
    global_cfg = SimpleGlobalConfig(
        L=float(global_raw["L"]),
        rho_target=float(global_raw["rho_target"]),
        avg_nn_dist=float(global_raw["avg_nn_dist"]),
        grain_count=int(global_raw["grain_count"]),
        objects_per_grain=int(global_raw.get("objects_per_grain", 20)),
        points_per_object=int(global_raw.get("points_per_object", 1000)),
        collision_safety_factor=float(global_raw.get("collision_safety_factor", 4.2)),
        random_seed=int(global_raw.get("random_seed", 0)),
        data_path=data_path,
        modelnet_path=pathlib.Path(global_raw.get("modelnet_path", "datasets/ModelNet40_fast")),
    )
    
    phases_section = raw.get("phases", {})
    phase_configs: Dict[str, SimplePhaseConfig] = {}
    
    for phase_name, phase_data in phases_section.items():
        phase_configs[phase_name] = SimplePhaseConfig(
            name=phase_name,
            phase_type=str(phase_data["phase_type"]),
            structural_params=phase_data.get("structural_params", {}),
        )
        
    grain_section = raw.get("grain_assignment", {})
    if "explicit" in grain_section:
        assignment_cfg = SimpleGrainAssignmentConfig(mode="explicit", assignments=list(grain_section["explicit"]))
    else:
        probs = grain_section.get("probabilities")
        if probs is None:
            # Fallback or error, for simple generator let's be strict
            if "explicit" not in grain_section: # Should have been caught
                 # Ensure at least probabilities or explicit exists
                 pass
        
        # If probabilities provided
        if probs:
            if not np.isclose(sum(probs.values()), 1.0):
                 raise ValueError("grain assignment probabilities must sum to 1.0")
            assignment_cfg = SimpleGrainAssignmentConfig(mode="probabilistic", probabilities=dict(probs))
        elif "explicit" in grain_section:
             pass # Already handled
        else:
             raise ValueError("grain_assignment must provide 'explicit' or 'probabilities'")

    return global_cfg, phase_configs, assignment_cfg

class SimpleSyntheticGenerator:
    """
    Generates simple geometric objects (spheres, boxes) filled with random points.
    Matches the output format of SyntheticAtomisticDatasetGenerator.
    """
    
    def __init__(
        self, config_path: str | pathlib.Path,
        rng: Optional[np.random.Generator] = None,
        progress: bool = True,
    ):
        self.global_cfg, self.phase_cfgs, self.grain_assignment_cfg = load_simple_config(config_path)
        self.rng = rng or np.random.default_rng(self.global_cfg.random_seed)
        self.progress = progress
        self._start_time = time.perf_counter()
        
        self.grains: List[Dict[str, Any]] = []
        self.atoms: Optional[np.ndarray] = None
        self.atom_count: int = 0
        self.metadata: Dict[str, Any] = {}
        
        # Build phase maps
        self._build_phase_maps()
    def _progress(self, message: str) -> None:
        if self.progress:
            elapsed = time.perf_counter() - self._start_time
            print(f"[{elapsed:8.2f}s] {message}")

    def _build_phase_maps(self) -> None:
        PHASE_ID_MAP.clear()
        PHASE_NAME_MAP.clear()
        for i, name in enumerate(sorted(self.phase_cfgs.keys())):
            PHASE_ID_MAP[name] = i
            PHASE_NAME_MAP[i] = name

    def sample_grains(self) -> None:
        self._progress("Sampling object seeds and types")
        n_grains = self.global_cfg.grain_count
        L = self.global_cfg.L
        
        # Sample seed positions (centers of objects)
        self.seed_positions = self.rng.uniform(0.1*L, 0.9*L, size=(n_grains, 3)).astype(np.float32)
        
        # Assign phases
        phases = self._assign_grain_phases()
        
        self.grains = []
        for idx in range(n_grains):
            self.grains.append({
                "grain_id": idx,
                "seed_position": self.seed_positions[idx],
                "base_phase_id": phases[idx],
                # Random rotation for boxes
                "base_rotation": self._random_rotation_matrix(), 
            })
        self._progress(f"Sampled {n_grains} objects")

    def _assign_grain_phases(self) -> List[str]:
        n_grains = self.global_cfg.grain_count
        if self.grain_assignment_cfg.mode == "explicit":
            return self.grain_assignment_cfg.assignments or []
        probs = self.grain_assignment_cfg.probabilities or {}
        phase_ids = list(probs.keys())
        prob_values = np.array(list(probs.values()))
        return list(self.rng.choice(phase_ids, size=n_grains, p=prob_values))

    def _random_rotation_matrix(self) -> np.ndarray:
        # Simple random rotation
        q = self.rng.normal(size=4)
        q /= np.linalg.norm(q)
        # Convert quat to matrix (simple formula)
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)

    def _get_box_points(self, n_p: int, dims: Sequence[float]) -> np.ndarray:
        """Generate points on the surface of a box."""
        if n_p <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        dx, dy, dz = dims
        # Areas of faces perpendicular to x, y, z
        areas = np.array([dy*dz, dx*dz, dx*dy])
        probs = areas / np.sum(areas)
        
        # 1. Choose axis (0=x, 1=y, 2=z)
        ax_choices = self.rng.choice([0, 1, 2], size=n_p, p=probs)
        
        # 2. Choose direction (+/-)
        dirs = self.rng.choice([-1, 1], size=n_p)
        
        pos = np.zeros((n_p, 3), dtype=np.float32)
        
        # Set the fixed coordinate for the chosen face
        # e.g. if axis=0 (x), x = +/- dx/2
        pos[np.arange(n_p), ax_choices] = dirs * (np.array(dims)[ax_choices] / 2)
        
        # 3. Sample uniformly on the other two coordinates
        for d in range(3):
            mask = (ax_choices != d)
            dim_val = dims[d]
            pos[mask, d] = self.rng.uniform(-dim_val/2, dim_val/2, size=np.sum(mask))
            
        return pos

    def _load_modelnet_data(self) -> None:
        """Loads ModelNet data for all required phases."""
        self._progress("Loading ModelNet data...")
        
        # Identify all required classes
        required_classes = set()
        for phase in self.phase_cfgs.values():
            # For ModelNet mode, phase_type is the class name
            required_classes.add(phase.phase_type)
            
        if not required_classes:
            return

        try:
            from src.data_utils.modelnet_fast_loader import ModelNetFastDataset
        except ImportError:
            # Fallback if running from a different directory structure or if module not found
            import sys
            # simple_generator.py is in src/data_utils/synthetic
            # we need to add the project root (containing 'src') to sys.path
            project_root = str(pathlib.Path(__file__).resolve().parents[3])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from src.data_utils.modelnet_fast_loader import ModelNetFastDataset

        modelnet_path = getattr(self.global_cfg, 'modelnet_path', 'datasets/ModelNet40_fast')
        
        # We load a single dataset containing all required classes
        # We ask for max points (e.g. 2048) and will sample down later
        self.modelnet_dataset = ModelNetFastDataset(
            root_dir=modelnet_path,
            split='train', # Use train split for generation usually
            classes=list(required_classes),
            n_points=2048 
        )
        
        # Organize by class for O(1) access
        self.modelnet_samples = {cls: [] for cls in required_classes}
        
        # We iterate manually to group them. 
        # Accessing self.modelnet_dataset.points directly is faster if possible
        # but let's be safe and use usage patterns
        
        # Optim: Directly access the internal lists if possible to avoid huge loop overhead
        points_tensor = self.modelnet_dataset.points
        labels_tensor = self.modelnet_dataset.labels
        
        for cls in required_classes:
            cls_idx = self.modelnet_dataset.class_to_idx.get(cls)
            if cls_idx is None:
                print(f"Warning: Class {cls} not found in ModelNet dataset!")
                continue
                
            # Find all indices for this class
            indices = torch.where(labels_tensor == cls_idx)[0]
            if len(indices) == 0:
                 print(f"Warning: No samples found for class {cls}")
            
            # Store as list of numpy arrays
            self.modelnet_samples[cls] = points_tensor[indices].numpy()

        self._progress(f"Loaded ModelNet data for classes: {list(self.modelnet_samples.keys())}")

    def populate_atoms(self) -> None:
        self._progress("Generating objects...")
        
        # Load data if not already (check if any phase looks like a modelnet class? 
        # Or just always try to load if we are in that mode?
        # A heuristic: if phase_type is NOT one of the known geometric shapes, assume ModelNet)
        
        known_shapes = {'box', 'sphere', 'cylinder', 'torus', 'helix', 'airplane', 'chair', 'simple_placeholder'}
        # airplane and chair are in both, but we want to use ModelNet for them if possible
        # Let's check config. If user provided a path or if we want to force it.
        # For now, let's load ModelNet data if we can.
        self._load_modelnet_data()

        all_atoms_list = []
        
        L = self.global_cfg.L
        points_per_object = self.global_cfg.points_per_object
        
        # 1. Determine number of objects
        n_grains = len(self.grains)
        if n_grains == 0:
            return

        objects_per_grain = self.global_cfg.objects_per_grain
        target_total_objects = n_grains * objects_per_grain
        
        # Heuristic for object size
        vol_per_object = (L**3 * 0.15) / target_total_objects
        est_radius = (vol_per_object / (4/3 * math.pi))**(1/3)
        base_size = est_radius
        
        collision_factor = self.global_cfg.collision_safety_factor
        collision_radius = base_size * collision_factor 
        
        self._progress(f"Targeting {target_total_objects} objects (size {base_size:.2f})...")
        
        # 2. Sample valid object centers
        object_centers = []
        max_attempts = target_total_objects * 100
        attempts = 0
        
        while len(object_centers) < target_total_objects and attempts < max_attempts:
            attempts += 1
            cand = self.rng.uniform(base_size, L - base_size, size=3).astype(np.float32)
            
            collision = False
            for existing in object_centers:
                dist = np.linalg.norm(cand - existing)
                if dist < collision_radius:
                    collision = True
                    break
            
            if not collision:
                object_centers.append(cand)
        
        object_centers = np.array(object_centers)
        total_objects = len(object_centers)
        self._progress(f"Placed {total_objects} non-overlapping objects")

        if total_objects == 0:
             return

        # 3. Assign to nearest grain seed
        grain_seeds = np.array([g['seed_position'] for g in self.grains])
        tree = cKDTree(grain_seeds)
        _, grain_indices = tree.query(object_centers)
        
        # 4. Generate objects
        for i in range(total_objects):
            center = object_centers[i]
            grain_idx = grain_indices[i]
            grain = self.grains[grain_idx]
            
            phase_name = grain['base_phase_id']
            phase_cfg = self.phase_cfgs[phase_name]
            class_name = phase_cfg.phase_type # Use phase_type as class name
            
            # Structural params
            structural_params = phase_cfg.structural_params
            noise_sigma = float(structural_params.get("noise_sigma", 0.0))
            
            # Use ModelNet sample if available
            if class_name in self.modelnet_samples and len(self.modelnet_samples[class_name]) > 0:
                # Pick a random sample index
                samples = self.modelnet_samples[class_name] # (N_samples, 2048, 3)
                sample_idx = self.rng.integers(0, len(samples))
                raw_points = samples[sample_idx] # (2048, 3)
                
                # Randomly sample points from the cloud
                # If we need more points than available, replace=True
                curr_n = raw_points.shape[0]
                if curr_n >= points_per_object:
                    choice = self.rng.choice(curr_n, points_per_object, replace=False)
                else:
                    choice = self.rng.choice(curr_n, points_per_object, replace=True)
                
                local_pos = raw_points[choice].copy()
                
                # Normalize just in case? ModelNet is usually in unit sphere/box.
                # Let's scale it to our base_size.
                # Usually ModelNet is in [-1, 1]. So radius is 1.
                # We want radius ~ base_size.
                local_pos *= base_size 

            else:
                 # Fallback to sphere
                 shape_type = 'sphere' 
                 radius = base_size
                 vecs = self.rng.normal(0, 1, size=(points_per_object, 3))
                 norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                 local_pos = (vecs / norms) * radius
                 local_pos = local_pos.astype(np.float32)

            # Apply Noise
            if noise_sigma > 0 and len(local_pos) > 0:
                noise = self.rng.normal(0, noise_sigma, size=local_pos.shape).astype(np.float32)
                local_pos += noise
            
            if len(local_pos) == 0:
                continue

            local_pos = local_pos.reshape(-1, 3)

            # Rotate and Translate
            # Grain rotation
            R = grain['base_rotation']
            
            # Additional small random rotation per object?
            # User said: "random objects of one with simular orientation creates grain"
            # "simular orientation" implies some deviation.
            # Let's add a small deviation if configured, or just use grain rotation?
            # "Grains might be of the same class but different orientation" -> This is handled by grain['base_rotation'] which is random per grain.
            # "simular orientation" -> maybe per-object deviation?
            # Let's assume strict grain orientation for now as in the original code, 
            # or maybe add a small jitter if requested.
            # For now, strict grain orientation.
            
            world_pos = (local_pos @ R.T) + center
            
            if len(world_pos) > 0:
                chunk = np.zeros(len(world_pos), dtype=ATOM_DTYPE)
                chunk['position'] = world_pos
                chunk['phase_id'] = PHASE_ID_MAP[phase_name]
                chunk['grain_id'] = grain['grain_id']
                chunk['orientation'] = R.flatten()
                chunk['alive'] = True
                chunk['object_id'] = i
                all_atoms_list.append(chunk)

        # Concatenate all
        if all_atoms_list:
            final_atoms = np.concatenate(all_atoms_list)
            # Clip to box (optional, but good for validity)
            L = self.global_cfg.L
            mask = (
                (final_atoms['position'][:, 0] >= 0) & (final_atoms['position'][:, 0] <= L) &
                (final_atoms['position'][:, 1] >= 0) & (final_atoms['position'][:, 1] <= L) &
                (final_atoms['position'][:, 2] >= 0) & (final_atoms['position'][:, 2] <= L)
            )
            self.atoms = final_atoms[mask]
            self.atom_count = len(self.atoms)
            
            # Assign pre_index
            self.atoms['pre_index'] = np.arange(self.atom_count)
            
        else:
            self.atoms = np.zeros(0, dtype=ATOM_DTYPE)
            self.atom_count = 0
            
        self._progress(f"Generated {self.atom_count} atoms in {total_objects} objects")

    def save_outputs(self) -> None:
        output_dir = self.global_cfg.data_path
        output_dir.mkdir(parents=True, exist_ok=True)
        self._progress(f"Saving outputs to {output_dir}")
        
        if self.atoms is None:
            return

        final_atoms = self.atoms[:self.atom_count]
        
        np.save(output_dir / "atoms.npy", final_atoms['position'])
        np.save(output_dir / "atoms_full.npy", final_atoms)
        
        # Create minimal reference structures (placeholders)
        ref_structs = {}
        for name in self.phase_cfgs:
             ref_structs[name] = {"phase_type": "simple_placeholder"}
        np.save(output_dir / "reference_structures.npy", ref_structs, allow_pickle=True)

        self._build_metadata()
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
            
        with open(output_dir / "phase_mapping.json", "w") as f:
            json.dump({"name_to_id": PHASE_ID_MAP, "id_to_name": {v: k for k, v in PHASE_ID_MAP.items()}}, f, indent=2)
            
        self._progress("Saved outputs")

    def _build_metadata(self) -> None:
        L = self.global_cfg.L
        n_final = self.atom_count
        
        self.metadata["global"] = {
            "box_size": float(L),
            "volume": float(L**3),
            "rho_target": float(self.global_cfg.rho_target),
            "rho_actual": float(n_final / L**3) if L > 0 else 0,
            "avg_nn_dist": float(self.global_cfg.avg_nn_dist),
            "grain_count": len(self.grains),
            "N_final": int(n_final),
            "phases": sorted(PHASE_ID_MAP.keys()),
            "generator": "SimpleSyntheticGenerator"
        }
        
        grain_records = []
        for grain in self.grains:
            # Count atoms for this grain
            mask = self.atoms['grain_id'] == grain['grain_id']
            indices = np.where(mask)[0]
            
            grain_records.append({
                "grain_id": int(grain['grain_id']),
                "base_phase_id": grain['base_phase_id'],
                "seed_position": [float(x) for x in grain['seed_position']],
                "orientation_matrix": grain['base_rotation'].tolist(),
                "n_atoms": len(indices),
                "atom_indices": indices.tolist(),
            })
        self.metadata["grains"] = grain_records

    def create_visualizations(self) -> None:
        try:
            import sys, os
            sys.path.append(os.getcwd())
            from src.data_utils.synthetic.visualization import generate_visualizations
            output_dir = self.global_cfg.data_path
            
            if self.atoms is None or self.atom_count == 0:
                 return

            final_atoms = self.atoms[:self.atom_count]
            
            # Prepare atoms list for visualization (needs specific format)
            atoms_list = []
            for i in range(self.atom_count):
                a = final_atoms[i]
                atoms_list.append({
                    "final_index": i,
                    "position": a['position'],
                    "phase_id": PHASE_NAME_MAP.get(a['phase_id'], f"unknown_{a['phase_id']}"),
                    "grain_id": int(a['grain_id']),
                    "orientation": a['orientation'].reshape(3,3),
                    "orientation": a['orientation'].reshape(3,3),
                    "alive": bool(a['alive']),
                    "object_id": int(a['object_id']),
                })

            generate_visualizations(self.global_cfg, self.grains, atoms_list, self.metadata, self.rng, output_dir)
        except ImportError:
            self._progress("Visualization module not available")
        except Exception as e:
            self._progress(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate simple synthetic atomistic datasets")
    parser.add_argument("config", type=str, nargs="?", default="configs/data/synth_modelnet.yaml")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")

    args = parser.parse_args()
    
    generator = SimpleSyntheticGenerator(args.config, progress=not args.quiet)
    generator.sample_grains()
    generator.populate_atoms()
    generator.save_outputs()
    
    if not args.skip_viz:
        generator.create_visualizations()

if __name__ == "__main__":
    main()
