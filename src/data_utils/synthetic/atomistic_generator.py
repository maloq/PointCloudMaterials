"""
Synthetic atomistic dataset generation pipeline (Enhanced for Metal Crystallization).

Improvements:
- Physics-based relaxation (ASE/EMT)
- Elastic strain augmentation
- Hard overlap removal (KDTree)
- Planar defects (Stacking faults/Twins)
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import itertools
import numpy as np
import yaml
import sys, os

# Scientific imports for realism
from scipy.spatial import cKDTree

# Try importing ASE for physics-based relaxation
try:
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.optimize import FIRE
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE (Atomic Simulation Environment) not found. Physics relaxation will be skipped.")

sys.path.append(os.getcwd())
from src.data_utils.synthetic.visualization import generate_visualizations

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhasePerturbationConfig:
    sigma_thermal: float
    p_dropout: float
    dropout_relax_radius: float # Kept for legacy, but ASE relaxation is preferred
    # New: Probability of a planar slip (stacking fault/twin)
    planar_fault_prob: float = 0.0
    density_bubbles: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PhaseConfig:
    name: str
    phase_type: str
    # Default mass/symbol for physics relaxation (e.g., 'Cu', 'Al')
    chemical_symbol: str 
    structural_params: Dict[str, Any]
    perturbations: PhasePerturbationConfig

@dataclass
class GrainAssignmentConfig:
    mode: str
    assignments: Optional[List[str]] = None
    probabilities: Optional[Dict[str, float]] = None

@dataclass
class GlobalConfig:
    L: float
    rho_target: float
    avg_nn_dist: float
    grain_count: int
    intermediate_layer_thickness_factor: float
    random_seed: int
    data_path: pathlib.Path
    # New: Maximum elastic strain (e.g., 0.02 for 2%)
    max_elastic_strain: float = 0.0
    # New: Minimum physical distance allowed between atoms (Angstroms)
    min_physical_dist: float = 1.5
    additional: Dict[str, Any] = field(default_factory=dict)
    t_layer: float = 0.0

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> Tuple[GlobalConfig, Dict[str, PhaseConfig], GrainAssignmentConfig]:
    config_path = pathlib.Path(path)
    with config_path.open("r") as handle:
        raw_cfg = yaml.safe_load(handle)

    global_raw = raw_cfg.get("global", {})
    data_path = pathlib.Path(global_raw.get("output_dir", "output/synthetic_data"))
    
    global_cfg = GlobalConfig(
        L=float(global_raw["L"]),
        rho_target=float(global_raw["rho_target"]),
        avg_nn_dist=float(global_raw["avg_nn_dist"]),
        grain_count=int(global_raw["grain_count"]),
        intermediate_layer_thickness_factor=float(global_raw.get("intermediate_layer_thickness_factor", 0.0)),
        random_seed=int(global_raw.get("random_seed", 0)),
        max_elastic_strain=float(global_raw.get("max_elastic_strain", 0.0)),
        min_physical_dist=float(global_raw.get("min_physical_dist", 1.5)),
        data_path=data_path,
        additional={k: v for k, v in global_raw.items() if k not in {
            "L", "rho_target", "avg_nn_dist", "grain_count", "intermediate_layer_thickness_factor",
            "random_seed", "output_dir", "max_elastic_strain", "min_physical_dist"
        }},
    )
    global_cfg.t_layer = global_cfg.intermediate_layer_thickness_factor * global_cfg.avg_nn_dist

    phases_section = raw_cfg.get("phases", {})
    phase_configs: Dict[str, PhaseConfig] = {}
    for phase_name, phase_payload in phases_section.items():
        perturb_raw = phase_payload.get("perturbations", {})
        perturb_cfg = PhasePerturbationConfig(
            sigma_thermal=float(perturb_raw.get("sigma_thermal", 0.0)),
            p_dropout=float(perturb_raw.get("p_dropout", 0.0)),
            dropout_relax_radius=float(perturb_raw.get("dropout_relax_radius", 0.0)),
            planar_fault_prob=float(perturb_raw.get("planar_fault_prob", 0.0)),
            density_bubbles=list(perturb_raw.get("density_bubbles", [])),
        )
        phase_configs[phase_name] = PhaseConfig(
            name=phase_name,
            phase_type=str(phase_payload["phase_type"]),
            chemical_symbol=str(phase_payload.get("chemical_symbol", "Cu")), # Default to Copper
            structural_params=phase_payload.get("structural_params", {}),
            perturbations=perturb_cfg,
        )

    grain_section = raw_cfg.get("grain_assignment", {})
    if "explicit" in grain_section:
        assignment_cfg = GrainAssignmentConfig(mode="explicit", assignments=list(grain_section["explicit"]))
    else:
        probs = grain_section.get("probabilities")
        if probs is None:
            # Default uniform if missing
            assignment_cfg = GrainAssignmentConfig(mode="probabilistic", probabilities={k: 1.0 for k in phase_configs})
        else:
            assignment_cfg = GrainAssignmentConfig(mode="probabilistic", probabilities=dict(probs))

    return global_cfg, phase_configs, assignment_cfg

def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    quat = np.array([qw, qx, qy, qz])
    return quat / np.linalg.norm(quat)

# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class SyntheticAtomisticDatasetGenerator:
    def __init__(
        self,
        config_path: str | pathlib.Path,
        rng: Optional[np.random.Generator] = None,
        progress: bool = True,
        skip_visualization: bool = False,
    ):
        self.global_cfg, self.phase_cfgs, self.grain_assignment_cfg = load_config(config_path)
        self.rng = rng or np.random.default_rng(self.global_cfg.random_seed)
        self.progress = progress
        self.skip_visualization = skip_visualization
        self._start_time = time.perf_counter()

        self.reference_structures: Dict[str, Dict[str, Any]] = {}
        self.grains: List[Dict[str, Any]] = []
        self.atoms: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._next_pre_index: int = 0
        self.seed_positions: Optional[np.ndarray] = None
        
        # Neighbors graph for metadata
        self.grain_neighbors: Dict[int, Set[int]] = defaultdict(set)

    def _progress(self, message: str) -> None:
        if not self.progress:
            return
        elapsed = time.perf_counter() - self._start_time
        print(f"[{elapsed:7.2f}s] {message}")

    # ------------------------------------------------------------------ #
    # Step 1 – Reference Structures
    # ------------------------------------------------------------------ #

    def build_reference_structures(self) -> None:
        self._progress("Preparing reference structures...")
        self.reference_structures = {}
        for phase_name, phase_cfg in self.phase_cfgs.items():
            recipe = self._build_phase_recipe(phase_cfg)
            self.reference_structures[phase_name] = recipe

    def _build_phase_recipe(self, phase_cfg: PhaseConfig) -> Dict[str, Any]:
            avg_nn = self.global_cfg.avg_nn_dist
            phase_type = phase_cfg.phase_type
            
            recipe = {
                "phase_type": phase_type,
                "name": phase_cfg.name,
                "chemical_symbol": phase_cfg.chemical_symbol,
                "structural_params": phase_cfg.structural_params # Pass through for nuclei config
            }

            # --- Cubic Lattices ---
            if phase_type == "crystal_fcc":
                # Face Centered Cubic (Cu, Al, Au, Ni)
                lattice_constant = avg_nn * math.sqrt(2.0)
                recipe["lattice_vectors"] = (np.eye(3) * lattice_constant).tolist()
                recipe["motif"] = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
                
            elif phase_type == "crystal_bcc":
                # Body Centered Cubic (Fe, W, Na)
                lattice_constant = (2.0 / math.sqrt(3.0)) * avg_nn
                recipe["lattice_vectors"] = (np.eye(3) * lattice_constant).tolist()
                recipe["motif"] = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]

            elif phase_type == "crystal_sc":
                # Simple Cubic (Polonium - rare, but good for contrast training)
                lattice_constant = avg_nn
                recipe["lattice_vectors"] = (np.eye(3) * lattice_constant).tolist()
                recipe["motif"] = [[0.0, 0.0, 0.0]]
                
            elif phase_type == "crystal_diamond":
                # Diamond Cubic (Si, Ge - valuable for semiconductor context)
                lattice_constant = (4.0 / math.sqrt(3.0)) * avg_nn
                recipe["lattice_vectors"] = (np.eye(3) * lattice_constant).tolist()
                # Diamond is FCC + basis at (1/4, 1/4, 1/4)
                fcc_motif = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
                shift = np.array([0.25, 0.25, 0.25])
                full_motif = np.vstack([fcc_motif, fcc_motif + shift])
                recipe["motif"] = full_motif.tolist()

            # --- Hexagonal Lattices ---
            elif phase_type == "crystal_hcp":
                # Hexagonal Close Packed (Mg, Ti, Zn, Co)
                # a = avg_nn, c = sqrt(8/3) * a
                a = avg_nn
                c = math.sqrt(8.0/3.0) * a
                
                # Hexagonal basis vectors: [a, 0, 0], [-a/2, a*sqrt(3)/2, 0], [0, 0, c]
                v1 = [a, 0.0, 0.0]
                v2 = [-0.5 * a, (math.sqrt(3.0) / 2.0) * a, 0.0]
                v3 = [0.0, 0.0, c]
                
                recipe["lattice_vectors"] = [v1, v2, v3]
                # Motif: atoms at (0,0,0) and (1/3, 2/3, 1/2) in lattice coords
                recipe["motif"] = [[0.0, 0.0, 0.0], [1.0/3.0, 2.0/3.0, 0.5]]

            # --- Liquid / Nucleated Liquid ---
            elif phase_type == "amorphous_random" or phase_type == "liquid_with_nuclei":
                # Just store the params, generation logic is handled in populate_atoms
                pass

            return recipe

    # ------------------------------------------------------------------ #
    # Step 2 – Grains
    # ------------------------------------------------------------------ #

    def sample_grains(self) -> None:
        self._progress("Sampling grains...")
        seeds = self.rng.uniform(0.0, self.global_cfg.L, size=(self.global_cfg.grain_count, 3))
        
        # Assign phases
        if self.grain_assignment_cfg.mode == "explicit":
            assignments = self.grain_assignment_cfg.assignments
        else:
            ids = list(self.phase_cfgs.keys())
            probs = list(self.grain_assignment_cfg.probabilities.values())
            # Normalize probs
            probs = np.array(probs) / np.sum(probs)
            assignments = self.rng.choice(ids, size=self.global_cfg.grain_count, p=probs)

        self.grains = []
        for idx, (seed, phase_id) in enumerate(zip(seeds, assignments)):
            rotation = random_rotation_matrix(self.rng)
            
            # NEW: Apply Elastic Strain per grain
            # We store the strain in the grain dict to apply during generation
            strain_tensor = np.eye(3)
            if self.global_cfg.max_elastic_strain > 0:
                mag = self.global_cfg.max_elastic_strain
                # Symmetric strain tensor
                eps = (self.rng.random((3,3)) - 0.5) * 2 * mag
                strain_tensor = np.eye(3) + eps
                
            self.grains.append({
                "grain_id": idx,
                "seed_position": seed,
                "base_phase_id": phase_id,
                "base_rotation": rotation,
                "elastic_strain": strain_tensor
            })
        self.seed_positions = seeds

    # ------------------------------------------------------------------ #
    # Step 3 – Populate Atoms (With Voronoi & Cleaning)
    # ------------------------------------------------------------------ #

    def populate_atoms(self) -> None:
        self._progress("Populating atoms...")
        self.atoms = []
        self._next_pre_index = 0
        
        for grain in self.grains:
            phase_id = grain["base_phase_id"]
            recipe = self.reference_structures[phase_id]
            phase_type = recipe["phase_type"]
            
            if phase_type.startswith("crystal"):
                cloud = self._generate_crystal_cloud(grain, recipe)
            elif phase_type == "liquid_with_nuclei":
                # --- NEW CALL HERE ---
                cloud = self._generate_nucleated_liquid(grain, recipe)
            else:
                # Assume amorphous
                cloud = self._generate_amorphous_cloud(grain, recipe)
                
            filtered_cloud = self._filter_to_voronoi_cell(cloud, grain["grain_id"])
            
            for pos in filtered_cloud:
                self.atoms.append({
                    "pre_index": self._next_pre_index,
                    "position": pos,
                    "phase_id": phase_id,
                    "grain_id": grain["grain_id"],
                    "orientation": grain["base_rotation"],
                    "alive": True
                })
                self._next_pre_index += 1

        # Run the overlap pruner (Crucial for the Liquid-Nucleus interface)
        self._prune_overlaps()


    def _generate_crystal_cloud(self, grain: Dict, recipe: Dict) -> np.ndarray:
        # Base lattice definition
        lattice_vecs = np.array(recipe["lattice_vectors"])
        motif = np.array(recipe["motif"])
        
        # Apply Elastic Strain (Affine transform on lattice vectors)
        lattice_vecs = lattice_vecs @ grain["elastic_strain"]
        
        # Determine Bounds (local coordinates relative to seed)
        rotation = grain["base_rotation"]
        seed = grain["seed_position"]
        
        # Create a bounding sphere roughly L*sqrt(3) to cover the box 
        # (Optimization: Only generate inside Voronoi approximation would be faster, 
        # but full box + filter is safer for periodicity)
        # Heuristic: Generate enough to cover the box, then filter.
        # Since this is expensive, we ideally project box corners to local frame.
        # Simplified here: Using a generous box around the seed.
        
        radius = self.global_cfg.L # Very generous
        
        # Tile
        # Estimate number of cells needed
        vol_cell = abs(np.linalg.det(lattice_vecs))
        n_cells_1d = int(radius / (vol_cell**(1/3))) + 2
        
        # Meshgrid-like generation
        ranges = [range(-n_cells_1d, n_cells_1d) for _ in range(3)]
        
        # To support planar slips (stacking faults), we need structured indices
        # Generate indices first
        ijk = np.array(list(itertools.product(*ranges)))
        
        # --- NEW: Planar Fault Injection ---
        phase_cfg = self.phase_cfgs[grain["base_phase_id"]]
        p_fault = phase_cfg.perturbations.planar_fault_prob
        
        if p_fault > 0:
            # Random normal for the slip plane (e.g., in integer coordinates)
            # Simple implementation: Slip along Z axis layers
            z_indices = ijk[:, 2]
            # Unique layers
            layers = np.unique(z_indices)
            shift_accumulator = np.zeros(3)
            shifts = np.zeros_like(ijk, dtype=float)
            
            for layer in layers:
                if self.rng.random() < p_fault:
                    # Introduce a shift (Shockley partial-like or random)
                    # Random vector roughly half nearest neighbor
                    shift_vec = (self.rng.random(3) - 0.5) * self.global_cfg.avg_nn_dist * 0.5
                    shift_accumulator += shift_vec
                
                mask = (z_indices == layer)
                shifts[mask] = shift_accumulator
            
            # Convert integer lattice points to cartesian
            # Points = (ijk * lattice) + shifts
            cartesian_lattice = ijk @ lattice_vecs + (shifts @ lattice_vecs) # Apply shift in lattice basis
        else:
            cartesian_lattice = ijk @ lattice_vecs

        # Add motif
        # Expand dimensions: (N_cells, 1, 3) + (1, N_motif, 3)
        full_cloud = cartesian_lattice[:, None, :] + (motif @ lattice_vecs)[None, :, :]
        full_cloud = full_cloud.reshape(-1, 3)
        
        # Rotate and translate to world space
        world_cloud = (full_cloud @ rotation.T) + seed
        
        # Quick Bounding Box Crop (Optimization)
        L = self.global_cfg.L
        mask = (world_cloud[:,0] > -2) & (world_cloud[:,0] < L+2) & \
               (world_cloud[:,1] > -2) & (world_cloud[:,1] < L+2) & \
               (world_cloud[:,2] > -2) & (world_cloud[:,2] < L+2)
               
        return world_cloud[mask]

    def _generate_amorphous_cloud(self, grain: Dict, recipe: Dict) -> np.ndarray:
            """
            Generates a realistic liquid structure.
            
            Instead of random points (Ideal Gas), we generate a 'hot' lattice.
            1. Generate a perfect lattice at the target liquid density.
            2. Apply large random displacements to break symmetry (scramble it).
            3. The subsequent ASE relaxation step (in step 4) will settle this 
            into a realistic liquid structure with proper Radial Distribution (RDF).
            """
            # 1. Determine liquid density
            # Liquid density is usually ~85-95% of solid density.
            # Let's assume the config 'rho_target' is set correctly, 
            # or derive it from the solid lattice constant.
            
            # We use a temporary FCC grid to ensure packing is efficient
            # (Random packing is very hard to get >60% density without overlaps)
            
            avg_nn = self.global_cfg.avg_nn_dist
            # Expand lattice slightly to match liquid density (lower than solid)
            # A factor of 1.05 expansion roughly gives ~15% volume increase (liquid)
            liquid_expansion = 1.02 
            eff_lattice_const = avg_nn * math.sqrt(2.0) * liquid_expansion
            
            lattice_vecs = np.eye(3) * eff_lattice_const
            motif = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
            
            # 2. Estimate bounds for the amorphous region
            # We generate a box slightly larger than L to avoid edge effects, then crop.
            # (Using the same tiling logic as crystal, but simplified)
            radius = self.global_cfg.L
            n_cells = int(radius / eff_lattice_const) + 2
            ranges = [range(-1, int(radius/eff_lattice_const)+2) for _ in range(3)]
            ijk = np.array(list(itertools.product(*ranges)))
            
            cartesian = ijk @ lattice_vecs
            full_cloud = cartesian[:, None, :] + (motif @ lattice_vecs)[None, :, :]
            full_cloud = full_cloud.reshape(-1, 3)
            
            # 3. SCRAMBLE: Apply large random displacements
            # This breaks the crystal symmetry. 
            # Max displacement ~ 40% of NN dist ensures disorder without fusion.
            scramble_mag = 0.4 * avg_nn 
            displacements = self.rng.uniform(-scramble_mag, scramble_mag, size=full_cloud.shape)
            amorphous_cloud = full_cloud + displacements
            
            # 4. Crop to Box
            L = self.global_cfg.L
            mask = (amorphous_cloud[:,0] > 0) & (amorphous_cloud[:,0] < L) & \
                (amorphous_cloud[:,1] > 0) & (amorphous_cloud[:,1] < L) & \
                (amorphous_cloud[:,2] > 0) & (amorphous_cloud[:,2] < L)
                
            return amorphous_cloud[mask]

    def _generate_nucleated_liquid(self, grain: Dict, recipe: Dict) -> np.ndarray:
        """
        Generates a scrambled liquid background, then inserts small crystalline 
        seeds (nuclei) into it. 
        
        Critical for training ML models to recognize the onset of crystallization.
        """
        # 1. Generate the base Liquid (Scrambled Lattice)
        # We temporarily change the recipe type to amorphous to reuse that logic
        # (Assuming you implemented the _generate_amorphous_cloud from the previous step)
        liquid_points = self._generate_amorphous_cloud(grain, recipe)
        
        # 2. Parse Nuclei Configuration
        # Example config in YAML:
        # structural_params:
        #   nucleus_count: 3
        #   nucleus_radius_min: 4.0
        #   nucleus_radius_max: 8.0
        #   nucleus_phase: "crystal_fcc" 
        
        params = recipe.get("structural_params", {})
        count = int(params.get("nucleus_count", 1))
        r_min = float(params.get("nucleus_radius_min", 3.0 * self.global_cfg.avg_nn_dist))
        r_max = float(params.get("nucleus_radius_max", 6.0 * self.global_cfg.avg_nn_dist))
        nuc_phase_name = params.get("nucleus_phase", "crystal_fcc") # Phase to insert
        
        if nuc_phase_name not in self.reference_structures:
            print(f"Warning: Nucleus phase {nuc_phase_name} not found. Returning pure liquid.")
            return liquid_points

        nuc_recipe = self.reference_structures[nuc_phase_name]
        
        final_points = [liquid_points]
        
        # 3. Insert Nuclei
        for _ in range(count):
            # A. Pick a random position in the grain/box
            # (For single box generation, anywhere in L is fine)
            center = self.rng.uniform(0, self.global_cfg.L, 3)
            radius = self.rng.uniform(r_min, r_max)
            
            # B. Carve hole in liquid
            # Identify liquid atoms within radius + small buffer
            # Buffer prevents atoms being too close to the crystal surface immediately
            current_liquid = np.concatenate(final_points)
            dists = np.linalg.norm(current_liquid - center, axis=1)
            mask_keep = dists > (radius + 0.5) # 0.5A buffer
            
            # Update liquid list (remove carved atoms)
            final_points = [current_liquid[mask_keep]]
            
            # C. Generate Crystal Sphere
            # We reuse the crystal generator but construct a dummy grain for it
            # so we can give it a random rotation
            dummy_grain = {
                "seed_position": center,
                "base_rotation": random_rotation_matrix(self.rng),
                "elastic_strain": np.eye(3), # No strain for nucleus usually
                # Tag phase so crystal generation can pull perturbation params (e.g., planar faults)
                "base_phase_id": nuc_phase_name,
            }
            
            # Generate full lattice
            crystal_cloud = self._generate_crystal_cloud(dummy_grain, nuc_recipe)
            
            # Crop to sphere
            d_cryst = np.linalg.norm(crystal_cloud - center, axis=1)
            sphere_points = crystal_cloud[d_cryst <= radius]
            
            final_points.append(sphere_points)
            
            # Add metadata about this nucleus (optional, but useful for training labels)
            self.metadata.setdefault("nuclei", []).append({
                "center": center.tolist(),
                "radius": radius,
                "phase": nuc_phase_name,
                "orientation": dummy_grain["base_rotation"].tolist()
            })

        return np.vstack(final_points)

    # ------------------------------------------------------------------ #
    def _filter_to_voronoi_cell(self, points: np.ndarray, grain_id: int) -> np.ndarray:
        if len(points) == 0: return points
        
        # Calculate distances to all seeds
        # This can be memory intensive for huge N, so we do it in chunks if needed.
        # Here, simple broadcasting.
        
        # points: (N, 3)
        # seeds: (M, 3)
        
        # We only care if the current grain_id is the CLOSEST seed.
        
        my_seed = self.seed_positions[grain_id]
        other_seeds = np.delete(self.seed_positions, grain_id, axis=0)
        if other_seeds.size == 0:
            # Single grain case: nothing to compare against
            return points
        
        # Distance to my seed
        d_me_sq = np.sum((points - my_seed)**2, axis=1)
        
        # Check against others
        # For strict Voronoi: keep if d_me < d_others
        # Optimization: Find the nearest OTHER seed distance
        
        keep_mask = np.ones(len(points), dtype=bool)
        
        # Chunked check to save memory
        chunk_size = 1000
        for i in range(0, len(points), chunk_size):
            end = min(i + chunk_size, len(points))
            chunk = points[i:end]
            
            # Dist matrix to others: (Chunk, M-1)
            d_others_sq = np.min(np.sum((chunk[:, None, :] - other_seeds[None, :, :])**2, axis=2), axis=1)
            
            keep_mask[i:end] = d_me_sq[i:end] <= d_others_sq
            
        return points[keep_mask]

    def _prune_overlaps(self) -> None:
        """
        Remove atoms that are unphysically close using a KDTree.
        This fixes the high-energy overlaps at grain boundaries.
        """
        if not self.atoms: return
        
        L_box = self.global_cfg.L
        positions = np.array([a["position"] for a in self.atoms])
        # Wrap to periodic box before building periodic KDTree (avoids out-of-box errors)
        positions = positions % L_box
        for idx, atom in enumerate(self.atoms):
            atom["position"] = positions[idx]
        min_dist = self.global_cfg.min_physical_dist
        
        self._progress(f"Pruning overlaps < {min_dist} A...")
        
        # Use KDTree for periodic boundary aware distance check
        # (Note: standard cKDTree handles boxsize for torus topology)
        tree = cKDTree(positions, boxsize=[L_box]*3)
        
        pairs = tree.query_pairs(r=min_dist)
        
        to_remove = set()
        
        # Heuristic: When two atoms overlap, remove one.
        # If one is inside a crystal and one is amorphous, maybe remove amorphous?
        # For now, random removal (remove the higher index) ensures stability.
        for i, j in pairs:
            if i in to_remove or j in to_remove:
                continue
            # Simple heuristic: remove j
            to_remove.add(j)
            
        # Rebuild list
        self.atoms = [a for idx, a in enumerate(self.atoms) if idx not in to_remove]
        self._progress(f"Removed {len(to_remove)} overlapping atoms. Current count: {len(self.atoms)}")

    # ------------------------------------------------------------------ #
    # Step 4 – Perturbations & Relaxation
    # ------------------------------------------------------------------ #

    def apply_perturbations(self) -> None:
        # 1. Thermal Jitter (Fast)
        self._progress("Applying thermal jitter...")
        for atom in self.atoms:
            phase = self.phase_cfgs[atom["phase_id"]]
            sigma = phase.perturbations.sigma_thermal
            if sigma > 0:
                atom["position"] += self.rng.normal(0, sigma, 3)

        # 2. Bubbles (Voids) - Optional
        # (Keeping your logic simplified here)
        
        # 3. PHYSICS RELAXATION (The heavy lifter)
        if ASE_AVAILABLE:
            self.relax_structure()
        else:
            self._progress("Skipping MD relaxation (ASE not installed).")

    def relax_structure(self) -> None:
        """
        Use Molecular Dynamics (Statics) to minimize the energy of the system.
        This creates realistic grain boundary structures (dislocations, etc).
        """
        self._progress("Running Physics-Based Relaxation (EMT Potential)...")
        
        positions = [a["position"] for a in self.atoms]
        # Map phase to symbol
        symbols = [self.phase_cfgs[a["phase_id"]].chemical_symbol for a in self.atoms]
        
        # Create ASE Atoms object
        atoms_ase = Atoms(symbols=symbols, positions=positions, 
                          cell=[self.global_cfg.L]*3, pbc=True)
        
        # Attach Calculator
        # EMT is a fast effective medium theory potential, good for FCC metals
        atoms_ase.calc = EMT()
        
        # Optimize
        # Fmax = 0.05 eV/A is a standard convergence criterion for structure relaxation
        opt = FIRE(atoms_ase, logfile=None) 
        try:
            opt.run(fmax=0.1, steps=200) # 200 steps max to keep generation fast
        except Exception as e:
            print(f"Relaxation warning: {e}")
            
        # Update positions
        new_positions = atoms_ase.get_positions(wrap=True) # Ensure inside box
        forces = atoms_ase.get_forces()
        potential_energies = atoms_ase.get_potential_energies()
        
        for i, atom in enumerate(self.atoms):
            atom["position"] = new_positions[i]
            # We can store physics data for the ML model too!
            atom["potential_energy"] = float(potential_energies[i])
            atom["force_mag"] = float(np.linalg.norm(forces[i]))

        self._progress("Relaxation complete.")

    # ------------------------------------------------------------------ #
    # Step 5 – Saving
    # ------------------------------------------------------------------ #

    def save_outputs(self) -> None:
        self._progress("Step 5/6: Saving outputs")
        output_dir = self.global_cfg.data_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct Final List for Storage AND Visualization
        final_atoms = []
        for i, atom in enumerate(self.atoms):
            if not atom["alive"]: continue
            
            # Construct record matching what visualization.py expects
            record = {
                "final_index": i, # Viz uses 'final_index'
                "position": atom["position"],
                "phase_id": atom["phase_id"], # Viz uses 'phase_id'
                "grain_id": atom["grain_id"],
                "orientation": atom["orientation"], # Crucial for local viz
                "potential_energy": atom.get("potential_energy", 0.0)
            }
            final_atoms.append(record)

        self._final_atoms_cache = final_atoms

        # Save Atoms (NPY for ML - clean array)
        pos_array = np.array([a["position"] for a in final_atoms], dtype=np.float32)
        np.save(output_dir / "atoms.npy", pos_array)
        
        # Save Metadata
        meta = {
            "global": {
                "L": self.global_cfg.L,
                "grain_count": self.global_cfg.grain_count
            },
            "grains": [
                {
                    "grain_id": g["grain_id"],
                    "base_phase_id": g["base_phase_id"],
                    "seed_position": g["seed_position"].tolist(),
                    "base_rotation": g["base_rotation"].tolist()
                }
                for g in self.grains
            ]
        }
        self.metadata = meta
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        self._progress(f"Saved {len(final_atoms)} atoms to {output_dir}")

    # ------------------------------------------------------------------ #
    # Step 6 – Visualization
    # ------------------------------------------------------------------ #
    def create_visualizations(self) -> None:
        """
        Generate diagnostic figures using the visualization util.
        Safe-guards against missing cache so generation never fails silently.
        """
        atoms = getattr(self, "_final_atoms_cache", None)
        if atoms is None:
            return
        try:
            generate_visualizations(
                global_cfg=self.global_cfg,
                grains=self.grains,
                atoms=atoms,
                metadata=self.metadata,
                rng=self.rng,
                output_dir=self.global_cfg.data_path,
            )
        except Exception as e:
            print(f"Visualization failed: {e}")

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self.build_reference_structures()
        self.sample_grains()
        self.populate_atoms()
        self.apply_perturbations()
        self.save_outputs()
        
        if not self.skip_visualization:
            self._progress("Step 6/6: Generating visualisations")
            self.create_visualizations()
        else:
            self._progress("Step 6/6: Skipping visualisations")
            
        self._progress("Generation complete")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="configs/data/data_synth_Al.yaml")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()

    gen = SyntheticAtomisticDatasetGenerator(
        args.config, 
        progress=not args.quiet,
        skip_visualization=args.skip_viz
    )
    gen.run()

if __name__ == "__main__":
    main()
