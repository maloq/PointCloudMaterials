from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.neighborlist import neighbor_list

from .config import GeneratorConfig
from .simulation import SimulatedSystems, ThermodynamicTrace


@dataclass(frozen=True)
class SystemDiagnostics:
    atom_count: int
    number_density_per_A3: float
    mass_density_g_per_cm3: float
    minimum_pair_distance_A: float
    maximum_force_eV_per_A: float
    mean_sampled_temperature_K: float
    mean_sampled_pressure_GPa: float
    mean_sampled_number_density_per_A3: float
    ptm_structure_fractions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_count": self.atom_count,
            "number_density_per_A3": self.number_density_per_A3,
            "mass_density_g_per_cm3": self.mass_density_g_per_cm3,
            "minimum_pair_distance_A": self.minimum_pair_distance_A,
            "maximum_force_eV_per_A": self.maximum_force_eV_per_A,
            "mean_sampled_temperature_K": self.mean_sampled_temperature_K,
            "mean_sampled_pressure_GPa": self.mean_sampled_pressure_GPa,
            "mean_sampled_number_density_per_A3": self.mean_sampled_number_density_per_A3,
            "ptm_structure_fractions": self.ptm_structure_fractions,
        }


def _tail(values: np.ndarray) -> np.ndarray:
    start = len(values) // 2
    return values[start:]


def _minimum_pair_distance(atoms: Atoms, search_cutoff_A: float) -> float:
    distances = neighbor_list("d", atoms, cutoff=search_cutoff_A)
    if distances.size == 0:
        return float(search_cutoff_A)
    return float(np.min(distances))


def _mass_density(atoms: Atoms) -> float:
    atomic_mass_amu = float(atoms.get_masses().sum())
    # 1 amu / A^3 = 1.66053906660 g / cm^3.
    return atomic_mass_amu / float(atoms.get_volume()) * 1.66053906660


def _ptm_structure_fractions(atoms: Atoms) -> dict[str, float]:
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
        from ovito.pipeline import Pipeline, StaticSource
    except ImportError as exc:
        raise ImportError(
            "Physical endpoint validation requires OVITO's Polyhedral Template Matching "
            "implementation. Install the repository requirements in the pointnet environment."
        ) from exc
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())
    data = pipeline.compute()
    atom_count = len(atoms)
    names = ("OTHER", "FCC", "HCP", "BCC", "ICO")
    return {
        name.lower(): float(
            int(data.attributes[f"PolyhedralTemplateMatching.counts.{name}"]) / atom_count
        )
        for name in names
    }


def diagnose_system(
    atoms: Atoms,
    trace: ThermodynamicTrace,
    config: GeneratorConfig,
    *,
    name: str,
    require_pressure_convergence: bool = True,
) -> SystemDiagnostics:
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    force_norms = np.linalg.norm(forces, axis=1)
    if not np.isfinite(force_norms).all():
        bad_indices = np.flatnonzero(~np.isfinite(force_norms))[:20].tolist()
        raise FloatingPointError(
            f"{name}: non-finite force norms for atom indices {bad_indices}."
        )
    maximum_force = float(np.max(force_norms))
    if maximum_force > config.validation.maximum_force_eV_per_A:
        raise RuntimeError(
            f"{name}: maximum force {maximum_force:.6f} eV/A exceeds "
            f"validation.maximum_force_eV_per_A="
            f"{config.validation.maximum_force_eV_per_A:.6f}. Increase equilibration, "
            "reduce the timestep, or use a potential valid for these configurations."
        )

    search_cutoff = max(5.0, 2.0 * config.validation.minimum_pair_distance_A)
    minimum_distance = _minimum_pair_distance(atoms, search_cutoff)
    if minimum_distance < config.validation.minimum_pair_distance_A:
        raise RuntimeError(
            f"{name}: minimum periodic pair distance {minimum_distance:.6f} A is below "
            f"validation.minimum_pair_distance_A="
            f"{config.validation.minimum_pair_distance_A:.6f} A."
        )

    sampled_pressure = float(np.mean(_tail(trace.pressure_GPa)))
    pressure_error = abs(sampled_pressure - config.dynamics.pressure_GPa)
    if require_pressure_convergence and pressure_error > config.validation.maximum_pressure_error_GPa:
        raise RuntimeError(
            f"{name}: tail-mean pressure is {sampled_pressure:.6f} GPa, "
            f"target is {config.dynamics.pressure_GPa:.6f} GPa, and error "
            f"{pressure_error:.6f} GPa exceeds validation.maximum_pressure_error_GPa="
            f"{config.validation.maximum_pressure_error_GPa:.6f}. Run longer NPT equilibration."
        )
    sampled_temperature = float(np.mean(_tail(trace.temperature_K)))
    temperature_error = abs(sampled_temperature - config.dynamics.target_temperature_K)
    if temperature_error > config.validation.maximum_temperature_error_K:
        raise RuntimeError(
            f"{name}: tail-mean temperature is {sampled_temperature:.3f} K, target is "
            f"{config.dynamics.target_temperature_K:.3f} K, and error "
            f"{temperature_error:.3f} K exceeds validation.maximum_temperature_error_K="
            f"{config.validation.maximum_temperature_error_K:.3f}. Run longer or adjust "
            "the thermostat time."
        )

    atom_count = len(atoms)
    ptm_fractions = _ptm_structure_fractions(atoms)
    return SystemDiagnostics(
        atom_count=atom_count,
        number_density_per_A3=float(atom_count / atoms.get_volume()),
        mass_density_g_per_cm3=_mass_density(atoms),
        minimum_pair_distance_A=minimum_distance,
        maximum_force_eV_per_A=maximum_force,
        mean_sampled_temperature_K=sampled_temperature,
        mean_sampled_pressure_GPa=sampled_pressure,
        mean_sampled_number_density_per_A3=float(
            atom_count / np.mean(_tail(trace.volume_A3))
        ),
        ptm_structure_fractions=ptm_fractions,
    )


def load_reference_densities(cache_dir: Path) -> dict[str, float]:
    manifest_path = cache_dir / "manifest.json"
    low_path = cache_dir / "box_low.npy"
    high_path = cache_dir / "box_high.npy"
    for required_path in (manifest_path, low_path, high_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Reference density cache is incomplete: missing {required_path}."
            )
    import json

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    atom_count = int(manifest["num_atoms"])
    box_low = np.load(low_path, mmap_mode="r")
    box_high = np.load(high_path, mmap_mode="r")
    if box_low.shape != box_high.shape or box_low.ndim != 2 or box_low.shape[1] != 3:
        raise ValueError(
            "Reference cache box arrays must both have shape (frames, 3), got "
            f"box_low={box_low.shape}, box_high={box_high.shape}."
        )
    densities = atom_count / np.prod(box_high - box_low, axis=1)
    edge_count = max(1, len(densities) // 10)
    return {
        "liquid_bulk": float(np.median(densities[:edge_count])),
        "solid_bulk": float(np.median(densities[-edge_count:])),
    }


def validate_reference_densities(
    diagnostics: dict[str, SystemDiagnostics], config: GeneratorConfig
) -> dict[str, float] | None:
    cache_dir = config.validation.reference_density_cache
    tolerance = config.validation.maximum_relative_density_error
    if cache_dir is None:
        return None
    if tolerance is None:
        raise ValueError(
            "validation.maximum_relative_density_error is required when "
            "validation.reference_density_cache is configured."
        )
    reference = load_reference_densities(cache_dir)
    comparisons = {
        "solid_bulk": diagnostics["bulk_solid"].mean_sampled_number_density_per_A3,
        "liquid_bulk": diagnostics["bulk_liquid"].mean_sampled_number_density_per_A3,
    }
    for phase_name, observed in comparisons.items():
        expected = reference[phase_name]
        relative_error = abs(observed - expected) / expected
        if relative_error > tolerance:
            raise RuntimeError(
                f"{phase_name}: simulated number density {observed:.8f} atom/A^3 differs "
                f"from repository MD reference {expected:.8f} atom/A^3 by "
                f"{relative_error:.2%}, above allowed {tolerance:.2%}. The selected "
                "potential/thermodynamic protocol is not validated for this benchmark."
            )
    return reference


def validate_systems(
    systems: SimulatedSystems, config: GeneratorConfig
) -> tuple[dict[str, SystemDiagnostics], dict[str, float] | None]:
    diagnostics = {
        "bulk_solid": diagnose_system(
            systems.solid, systems.solid_trace, config, name="bulk_solid"
        ),
        "bulk_liquid": diagnose_system(
            systems.liquid, systems.liquid_trace, config, name="bulk_liquid"
        ),
        "solid_liquid_interface": diagnose_system(
            systems.interface,
            systems.interface_trace,
            config,
            name="solid_liquid_interface",
            require_pressure_convergence=False,
        ),
    }
    solid_fcc_fraction = diagnostics["bulk_solid"].ptm_structure_fractions["fcc"]
    if solid_fcc_fraction < config.validation.minimum_solid_fcc_fraction:
        raise RuntimeError(
            f"bulk_solid: PTM recognizes only {solid_fcc_fraction:.2%} FCC atoms, below "
            f"validation.minimum_solid_fcc_fraction="
            f"{config.validation.minimum_solid_fcc_fraction:.2%}. The solid endpoint did not "
            "retain the declared FCC phase."
        )
    liquid_ptm = diagnostics["bulk_liquid"].ptm_structure_fractions
    liquid_crystalline_fraction = liquid_ptm["fcc"] + liquid_ptm["hcp"] + liquid_ptm["bcc"]
    if liquid_crystalline_fraction > config.validation.maximum_liquid_crystalline_fraction:
        raise RuntimeError(
            f"bulk_liquid: PTM recognizes {liquid_crystalline_fraction:.2%} atoms as "
            "FCC/HCP/BCC, above validation.maximum_liquid_crystalline_fraction="
            f"{config.validation.maximum_liquid_crystalline_fraction:.2%}. Increase melt time "
            "or temperature; the liquid endpoint is still crystalline."
        )
    interface_ptm = diagnostics["solid_liquid_interface"].ptm_structure_fractions
    interface_crystalline_fraction = (
        interface_ptm["fcc"] + interface_ptm["hcp"] + interface_ptm["bcc"]
    )
    interface_minimum = config.validation.minimum_interface_crystalline_fraction
    interface_maximum = config.validation.maximum_interface_crystalline_fraction
    if not interface_minimum <= interface_crystalline_fraction <= interface_maximum:
        raise RuntimeError(
            "solid_liquid_interface: PTM recognizes "
            f"{interface_crystalline_fraction:.2%} atoms as FCC/HCP/BCC, outside required "
            f"mixed-state interval [{interface_minimum:.2%}, {interface_maximum:.2%}]. "
            "Adjust interface_evolution_steps, melt_steps, or the slab size; the snapshot "
            "must contain both crystalline and non-crystalline populations."
        )
    reference = validate_reference_densities(diagnostics, config)
    return diagnostics, reference
