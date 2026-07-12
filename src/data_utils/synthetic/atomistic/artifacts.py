from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import write

from .config import GeneratorConfig
from .simulation import SimulatedSystems, ThermodynamicTrace
from .validation import SystemDiagnostics


PHASE_NAMES = ("solid_bulk", "liquid_bulk", "interface")
PHASE_TO_ID = {name: phase_id for phase_id, name in enumerate(PHASE_NAMES)}
ATOM_DTYPE = np.dtype(
    [
        ("position", np.float32, (3,)),
        ("phase_id", np.int16),
        ("grain_id", np.int32),
        ("orientation", np.float32, (9,)),
    ]
)


@dataclass(frozen=True)
class EnvironmentLabels:
    phase_names: np.ndarray
    grain_ids: np.ndarray
    grain_atom_indices: dict[int, np.ndarray]
    intermediate_atom_indices: np.ndarray
    slab_bounds_fractional: tuple[float, float] | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_bulk(atom_count: int, phase_name: str, grain_id: int) -> EnvironmentLabels:
    if phase_name not in PHASE_TO_ID or phase_name == "interface":
        raise ValueError(f"Bulk phase must be solid_bulk or liquid_bulk, got {phase_name!r}.")
    indices = np.arange(atom_count, dtype=np.int64)
    return EnvironmentLabels(
        phase_names=np.full(atom_count, phase_name, dtype="U16"),
        grain_ids=np.full(atom_count, grain_id, dtype=np.int32),
        grain_atom_indices={grain_id: indices},
        intermediate_atom_indices=np.empty(0, dtype=np.int64),
        slab_bounds_fractional=None,
    )


def _periodic_fractional_distance(values: np.ndarray, boundary: float) -> np.ndarray:
    delta = np.abs(values - boundary)
    return np.minimum(delta, 1.0 - delta)


def label_interface(
    atoms: Atoms,
    slab_bounds_fractional: tuple[float, float],
    interface_half_width_A: float,
) -> EnvironmentLabels:
    lower, upper = slab_bounds_fractional
    scaled_z = atoms.get_scaled_positions(wrap=True)[:, 2]
    cell_height = float(np.linalg.norm(np.asarray(atoms.cell)[2]))
    interface_mask = (
        np.minimum(
            _periodic_fractional_distance(scaled_z, lower),
            _periodic_fractional_distance(scaled_z, upper),
        )
        * cell_height
        <= interface_half_width_A
    )
    liquid_origin = (scaled_z >= lower) & (scaled_z < upper)
    phase_names = np.where(liquid_origin, "liquid_bulk", "solid_bulk").astype("U16")
    phase_names[interface_mask] = "interface"
    counts = {
        phase_name: int(np.count_nonzero(phase_names == phase_name))
        for phase_name in PHASE_NAMES
    }
    empty_phases = [name for name, count in counts.items() if count == 0]
    if empty_phases:
        raise ValueError(
            "Interface geometry produced empty labeled regions. "
            f"empty_phases={empty_phases}, counts={counts}, cell_height_A={cell_height:.6f}, "
            f"slab_bounds_fractional={slab_bounds_fractional}, "
            f"interface_half_width_A={interface_half_width_A}."
        )
    grain_ids = np.where(liquid_origin, 1, 0).astype(np.int32)
    return EnvironmentLabels(
        phase_names=phase_names,
        grain_ids=grain_ids,
        grain_atom_indices={
            0: np.flatnonzero(~liquid_origin).astype(np.int64),
            1: np.flatnonzero(liquid_origin).astype(np.int64),
        },
        intermediate_atom_indices=np.flatnonzero(interface_mask).astype(np.int64),
        slab_bounds_fractional=slab_bounds_fractional,
    )


def _atom_table(atoms: Atoms, labels: EnvironmentLabels) -> np.ndarray:
    table = np.empty(len(atoms), dtype=ATOM_DTYPE)
    table["position"] = np.asarray(atoms.get_positions(wrap=True), dtype=np.float32)
    table["phase_id"] = np.fromiter(
        (PHASE_TO_ID[str(name)] for name in labels.phase_names),
        dtype=np.int16,
        count=len(atoms),
    )
    table["grain_id"] = labels.grain_ids
    identity = np.eye(3, dtype=np.float32).reshape(9)
    table["orientation"] = identity
    return table


def _trace_metadata(trace: ThermodynamicTrace, atom_count: int) -> dict[str, Any]:
    return {
        "temperature_K": trace.temperature_K.tolist(),
        "pressure_GPa": trace.pressure_GPa.tolist(),
        "volume_A3": trace.volume_A3.tolist(),
        "number_density_per_A3": (atom_count / trace.volume_A3).tolist(),
        "potential_energy_eV_per_atom": trace.potential_energy_eV_per_atom.tolist(),
    }


def _phase_statistics(labels: EnvironmentLabels) -> dict[str, dict[str, float | int]]:
    atom_count = len(labels.phase_names)
    return {
        phase_name: {
            "n_atoms": int(np.count_nonzero(labels.phase_names == phase_name)),
            "fraction": float(np.count_nonzero(labels.phase_names == phase_name) / atom_count),
        }
        for phase_name in PHASE_NAMES
    }


def _grain_records(labels: EnvironmentLabels) -> list[dict[str, Any]]:
    records = []
    for grain_id, atom_indices in sorted(labels.grain_atom_indices.items()):
        phase_name = "solid_bulk" if grain_id == 0 else "liquid_bulk"
        records.append(
            {
                "grain_id": grain_id,
                "base_phase_id": phase_name,
                "seed_position": [0.0, 0.0, 0.0],
                "orientation_matrix": np.eye(3, dtype=np.float64).tolist(),
                "orientation_quaternion": [1.0, 0.0, 0.0, 0.0],
                "n_atoms": int(len(atom_indices)),
                "atom_indices": atom_indices.tolist(),
                "neighbors": [1 - grain_id] if len(labels.grain_atom_indices) == 2 else [],
            }
        )
    return records


def _intermediate_records(labels: EnvironmentLabels) -> list[dict[str, Any]]:
    if labels.intermediate_atom_indices.size == 0:
        return []
    return [
        {
            "region_id": 0,
            "grain_A_id": 0,
            "grain_B_id": 1,
            "intermediate_phase_id": "interface",
            "atom_indices": labels.intermediate_atom_indices.tolist(),
            "definition": {
                "kind": "distance_to_prepared_solid_liquid_boundary",
                "slab_bounds_fractional": list(labels.slab_bounds_fractional or ()),
            },
        }
    ]


def _write_environment(
    directory: Path,
    *,
    name: str,
    atoms: Atoms,
    labels: EnvironmentLabels,
    trace: ThermodynamicTrace,
    diagnostics: SystemDiagnostics,
    config: GeneratorConfig,
    potential_sha256: str,
) -> None:
    directory.mkdir(parents=True)
    atom_table = _atom_table(atoms, labels)
    positions = atom_table["position"]
    np.save(directory / "atoms.npy", positions)
    np.save(directory / "atoms_full.npy", atom_table)
    if config.output.save_extxyz:
        extxyz_atoms = atoms.copy()
        extxyz_atoms.arrays["phase_id"] = atom_table["phase_id"]
        extxyz_atoms.arrays["grain_id"] = atom_table["grain_id"]
        write(directory / "structure.extxyz", extxyz_atoms)

    box_vectors = np.asarray(atoms.cell, dtype=np.float64)
    phase_statistics = _phase_statistics(labels)
    metadata = {
        "schema_version": 3,
        "environment_name": name,
        "global": {
            "box_size": float(np.cbrt(atoms.get_volume())),
            "box_vectors_A": box_vectors.tolist(),
            "volume_A3": float(atoms.get_volume()),
            "N_final": len(atoms),
            "rho_actual": float(len(atoms) / atoms.get_volume()),
            "temperature_K": config.dynamics.target_temperature_K,
            "pressure_GPa": config.dynamics.pressure_GPa,
            "random_seed": config.random_seed,
            "grain_count": len(labels.grain_atom_indices),
            "phases": list(PHASE_NAMES),
        },
        "physics": {
            "chemical_symbol": config.system.chemical_symbol,
            "potential_path": str(config.potential.model_path),
            "potential_sha256": potential_sha256,
            "ensemble": "isothermal-isobaric (MTK) after explicit melt/quench preparation",
            "phase_density_policy": (
                "Density is measured from NPT volume samples. No phase density target is used."
            ),
            "label_policy": (
                "Labels encode preparation provenance and distance to the constructed interface; "
                "they are not produced by CNA, bond-order parameters, or template matching."
            ),
        },
        "diagnostics": diagnostics.to_dict(),
        "thermodynamic_trace": _trace_metadata(trace, len(atoms)),
        "grains": _grain_records(labels),
        "intermediate_regions": _intermediate_records(labels),
        "phase_statistics": phase_statistics,
    }
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (directory / "phase_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "name_to_id": PHASE_TO_ID,
                "id_to_name": {str(value): key for key, value in PHASE_TO_ID.items()},
            },
            handle,
            indent=2,
        )


def write_dataset(
    systems: SimulatedSystems,
    diagnostics: dict[str, SystemDiagnostics],
    reference_densities: dict[str, float] | None,
    config: GeneratorConfig,
) -> tuple[Path, ...]:
    output_root = config.output.root_dir
    if output_root.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_root}. Set output.overwrite=true "
            "only when replacement is intended."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    potential_sha256 = sha256_file(config.potential.model_path)
    environments = {
        "bulk_solid": (
            systems.solid,
            label_bulk(len(systems.solid), "solid_bulk", 0),
            systems.solid_trace,
        ),
        "bulk_liquid": (
            systems.liquid,
            label_bulk(len(systems.liquid), "liquid_bulk", 1),
            systems.liquid_trace,
        ),
        "solid_liquid_interface": (
            systems.interface,
            label_interface(
                systems.interface,
                systems.liquid_slab_bounds_fractional,
                config.system.interface_half_width_A,
            ),
            systems.interface_trace,
        ),
    }
    try:
        for name, (atoms, labels, trace) in environments.items():
            _write_environment(
                staging_root / name,
                name=name,
                atoms=atoms,
                labels=labels,
                trace=trace,
                diagnostics=diagnostics[name],
                config=config,
                potential_sha256=potential_sha256,
            )
        manifest = {
            "schema_version": 3,
            "dataset_name": config.dataset_name,
            "environment_dirs": list(environments),
            "config": config.to_dict(),
            "potential_sha256": potential_sha256,
            "repository_reference_number_densities_per_A3": reference_densities,
            "scientific_scope": {
                "supported_claim": (
                    "Comparison of learned local environments across physically simulated bulk "
                    "solid, metastable/supercooled liquid, and solid-liquid interfacial contexts."
                ),
                "unsupported_claim": (
                    "Identification of equilibrium thermodynamic phases from cluster identity alone."
                ),
            },
        }
        with (staging_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.replace(output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return tuple(output_root / name for name in environments)
