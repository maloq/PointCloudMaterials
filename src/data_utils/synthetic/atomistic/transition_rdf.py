from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .simulation import ThermodynamicTrace
from .transition_analysis import (
    analyze_phase_rdf,
    phase_rdf_metadata,
    write_phase_rdf_archive,
    write_phase_rdf_overview,
    write_phase_rdf_visualization,
)
from .transition_config import TransitionBranchConfig, TransitionConfig


def load_transition_trace(path: Path) -> ThermodynamicTrace:
    with np.load(path) as trajectory:
        return ThermodynamicTrace(
            step=trajectory["step"].copy(),
            temperature_K=trajectory["temperature_K"].copy(),
            pressure_GPa=trajectory["pressure_GPa"].copy(),
            volume_A3=trajectory["volume_A3"].copy(),
            potential_energy_eV_per_atom=trajectory[
                "potential_energy_eV_per_atom"
            ].copy(),
            positions_A=trajectory["positions_A"].copy(),
            cell_vectors_A=trajectory["cell_vectors_A"].copy(),
        )


def _write_branch_rdf(
    branch_dir: Path,
    *,
    branch: TransitionBranchConfig,
    config: TransitionConfig,
    progress: Callable[[str], None],
) -> Path:
    trajectory_path = branch_dir / "trajectory.npz"
    atom_table_path = branch_dir / "atoms_full.npy"
    metadata_path = branch_dir / "metadata.json"
    for path in (trajectory_path, atom_table_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot add phase RDF: required artifact is missing: {path}.")

    trace = load_transition_trace(trajectory_path)
    atom_table = np.load(atom_table_path, mmap_mode="r")
    prepared_phase_ids = np.asarray(atom_table["phase_id"], dtype=np.int64)
    if len(prepared_phase_ids) != trace.positions_A.shape[1]:
        raise RuntimeError(
            f"{branch_dir}: atom table has {len(prepared_phase_ids)} rows but trajectory "
            f"frames have {trace.positions_A.shape[1]} atoms."
        )

    analysis = analyze_phase_rdf(
        trace,
        chemical_symbol=config.generator.system.chemical_symbol,
        prepared_phase_ids=prepared_phase_ids,
        timestep_fs=config.generator.dynamics.timestep_fs,
        cutoff_A=config.analysis.rdf_cutoff_A,
        bins=config.analysis.rdf_bins,
        branch_name=branch.name,
        progress=progress,
    )
    rdf_path = branch_dir / "phase_rdf.npz"
    temporary_rdf_path = rdf_path.with_suffix(".npz.tmp")
    write_phase_rdf_archive(temporary_rdf_path, analysis)
    temporary_rdf_path.replace(rdf_path)

    visualization_dir = branch_dir / "visualizations"
    visualization_dir.mkdir(exist_ok=True)
    visualization_path = visualization_dir / "phase_rdf.png"
    write_phase_rdf_visualization(
        visualization_path,
        analysis=analysis,
        branch=branch,
    )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["rdf"] = phase_rdf_metadata(
        analysis,
        cutoff_A=config.analysis.rdf_cutoff_A,
        bins=config.analysis.rdf_bins,
    )
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    temporary_metadata_path.replace(metadata_path)
    return visualization_path


def add_phase_rdf_to_transition_dataset(
    config: TransitionConfig,
    *,
    progress: Callable[[str], None] = print,
) -> None:
    output_root = config.output.root_dir
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Cannot add phase RDF before transition generation completes: {manifest_path}."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["dataset_name"] != config.dataset_name:
        raise RuntimeError(
            f"{manifest_path}: dataset_name={manifest['dataset_name']!r} does not match "
            f"configuration dataset_name={config.dataset_name!r}."
        )

    branch_images: dict[str, Path] = {}
    for branch in (config.crystallization, config.melting):
        branch_images[branch.name] = _write_branch_rdf(
            output_root / branch.name,
            branch=branch,
            config=config,
            progress=progress,
        )
    write_phase_rdf_overview(
        output_root / "phase_rdf_overview.png",
        branch_images,
    )
    progress(f"Wrote per-phase RDF analysis to {output_root}")
