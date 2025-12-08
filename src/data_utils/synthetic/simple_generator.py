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

    def populate_atoms(self) -> None:
        self._progress("Generating simple objects...")
        all_atoms_list = []
        
        avg_nn = self.global_cfg.avg_nn_dist
        L = self.global_cfg.L
        points_per_object = self.global_cfg.points_per_object
        
        # 1. Determine number of objects and size
        n_grains = len(self.grains)
        if n_grains == 0:
            return

        objects_per_grain = self.global_cfg.objects_per_grain
        target_total_objects = n_grains * objects_per_grain
        
        # Heuristic for object size to allow packing
        # We need to ensure bounding spheres don't overlap.
        # Max bounding radius ~ 2.0 * base_size (Torus/Box/Cylinder)
        # Bounding sphere vol ~ 33 * base_size^3
        # Packing fraction ~0.4 -> N * 33 * base^3 = 0.4 * L^3
        # base^3 = (0.4 * L^3) / (33 * N) = 0.012 * L^3/N
        # Let's use a slightly looser factor 0.15 for volume per object calculation to get smaller base_size
        
        vol_per_object = (L**3 * 0.15) / target_total_objects
        est_radius = (vol_per_object / (4/3 * math.pi))**(1/3)
        base_size = est_radius
        
        # Collision radius factor
        collision_factor = self.global_cfg.collision_safety_factor
        collision_radius = base_size * collision_factor 
        
        self._progress(f"Targeting {target_total_objects} objects (size {base_size:.2f})...")
        
        # 2. Sample valid object centers (Rejection Sampling)
        object_centers = []
        max_attempts = target_total_objects * 100
        attempts = 0
        
        # Optimization: use specific spatial structure if needed, but for <500 objects brute force is fast enough
        # For larger numbers, we can use a grid or existing KDTree, but let's stick to simple list for now as N~250
        while len(object_centers) < target_total_objects and attempts < max_attempts:
            attempts += 1
            cand = self.rng.uniform(base_size, L - base_size, size=3).astype(np.float32)
            
            # Simple collision check
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
        self._progress(f"Placed {total_objects} non-overlapping objects (after {attempts} attempts)")

        if total_objects == 0:
             return

        # 3. Assign to nearest grain seed (Voronoi)
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
            phase_type = phase_cfg.phase_type
            
            # Read configuration
            structural_params = phase_cfg.structural_params
            
            # Key change: phase_type IS the shape, or specified in structural_params
            shape_type = structural_params.get("shape", phase_type).lower()
            noise_sigma = float(structural_params.get("noise_sigma", 0.0))
            
            # FIXED POINT COUNT
            n_points = points_per_object
            
            # FIXED POINT COUNT
            n_points = points_per_object
            
            # Set dimensions
            if shape_type == 'box':
                side = base_size * 2
                # Surface sampling for box: 6 faces
                # Each face has area side^2. Total area 6*side^2.
                # Uniformly pick a face, then sample on it? 
                # Better: generate n_points, randomly assign to faces.
                
                local_pos = np.zeros((n_points, 3), dtype=np.float32)
                # Randomly pick axis (0=x, 1=y, 2=z) and direction (+/- 1)
                axes = self.rng.integers(0, 3, size=n_points)
                directions = self.rng.choice([-1, 1], size=n_points)
                
                # Fill fixed coords
                local_pos[np.arange(n_points), axes] = directions * (side / 2)
                
                # Fill other coords uniformly
                for d in range(3):
                    mask = (axes != d)
                    local_pos[mask, d] = self.rng.uniform(-side/2, side/2, size=np.sum(mask))
                
            elif shape_type == 'cylinder':
                radius = base_size * 0.8
                height = base_size * 3.0
                
                # Surface areas
                area_side = 2 * math.pi * radius * height
                area_cap = math.pi * radius**2
                total_area = area_side + 2 * area_cap
                
                prob_side = area_side / total_area
                
                # Assign points to side or caps
                is_side = self.rng.random(size=n_points) < prob_side
                n_side = np.sum(is_side)
                n_cap = n_points - n_side
                
                local_pos = np.zeros((n_points, 3), dtype=np.float32)
                
                # Side points
                if n_side > 0:
                    theta = self.rng.uniform(0, 2*np.pi, size=n_side)
                    h = self.rng.uniform(-height/2, height/2, size=n_side)
                    local_pos[is_side, 0] = radius * np.cos(theta)
                    local_pos[is_side, 1] = radius * np.sin(theta)
                    local_pos[is_side, 2] = h
                    
                # Cap points
                if n_cap > 0:
                    theta = self.rng.uniform(0, 2*np.pi, size=n_cap)
                    # Uniform on disk: r = R * sqrt(U)
                    r = radius * np.sqrt(self.rng.uniform(0, 1, size=n_cap))
                    is_top = self.rng.choice([-1, 1], size=n_cap)
                    
                    local_pos[~is_side, 0] = r * np.cos(theta)
                    local_pos[~is_side, 1] = r * np.sin(theta)
                    local_pos[~is_side, 2] = is_top * (height / 2)

            elif shape_type == 'torus':
                R_major = base_size * 1.5
                r_minor = base_size * 0.5
                
                # Parametric sampling given u, v in [0, 2pi)
                # Note: This is not perfectly uniform density without area weighting,
                # but it ensures points are ON the surface.
                # Area element dA = r_minor * (R_major + r_minor * cos(v)) du dv
                # Rejection sampling for uniform surface density:
                
                local_pos = []
                while len(local_pos) < n_points:
                    batch_size = n_points - len(local_pos)
                    u = self.rng.uniform(0, 2*np.pi, size=batch_size)
                    v = self.rng.uniform(0, 2*np.pi, size=batch_size)
                    
                    # Rejection based on major radius contribution
                    w = (R_major + r_minor * np.cos(v)) / (R_major + r_minor)
                    accept = self.rng.random(size=batch_size) < w
                    
                    valid_u = u[accept]
                    valid_v = v[accept]
                    
                    x = (R_major + r_minor * np.cos(valid_v)) * np.cos(valid_u)
                    y = (R_major + r_minor * np.cos(valid_v)) * np.sin(valid_u)
                    z = r_minor * np.sin(valid_v)
                    
                    pts = np.stack([x, y, z], axis=1)
                    local_pos.extend(pts)
                
                local_pos = np.array(local_pos[:n_points], dtype=np.float32)

            elif shape_type == 'ellipsoid':
                radii = np.array([base_size * 0.7, base_size, base_size * 1.3], dtype=np.float32)
                
                # Sample on sphere then stretch
                # Note: this crowds points at "poles" of longest axis if not careful?
                # Actually, stretching uniform sphere points makes density non-uniform (lower curvature -> lower density).
                # But it IS on the surface. For simple recognition this is likely fine.
                # Correct uniform sampling on ellipsoid is complex.
                
                # Generate vectors on unit sphere
                vecs = self.rng.normal(0, 1, size=(n_points, 3))
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                unit_vecs = vecs / norms
                
                # Scale by radii
                local_pos = unit_vecs * radii
                local_pos = local_pos.astype(np.float32)

            elif shape_type == 'sphere':
                radius = base_size
                # Uniform samples on sphere
                vecs = self.rng.normal(0, 1, size=(n_points, 3))
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                local_pos = (vecs / norms) * radius
                local_pos = local_pos.astype(np.float32)
            
            else:
                 # Fallback to sphere
                 shape_type = 'sphere' 
                 radius = base_size
                 vecs = self.rng.normal(0, 1, size=(n_points, 3))
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
            R = grain['base_rotation']
            world_pos = (local_pos @ R.T) + center
            
            if len(world_pos) > 0:
                chunk = np.zeros(len(world_pos), dtype=ATOM_DTYPE)
                chunk['position'] = world_pos
                chunk['phase_id'] = PHASE_ID_MAP[phase_name]
                chunk['grain_id'] = grain['grain_id']
                chunk['orientation'] = R.flatten()
                chunk['alive'] = True
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
                    "alive": bool(a['alive'])
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
    parser.add_argument("config", type=str, nargs="?", default="configs/data/synth_simple.yaml")
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
