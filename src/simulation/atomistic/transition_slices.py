from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .transition_analysis import (
    write_structure_slice_overview,
    write_structure_slice_visualization,
)
from .transition_config import TransitionConfig
from .transition_rdf import load_transition_trace


def add_structure_slices_to_transition_datasets(
    config: TransitionConfig,
    dataset_roots: tuple[Path, ...],
    *,
    progress: Callable[[str], None] = print,
) -> None:
    for dataset_root in dataset_roots:
        manifest_path = dataset_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot create structure slices before transition generation completes: "
                f"{manifest_path}."
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest["dataset_name"] != config.dataset_name:
            raise RuntimeError(
                f"{manifest_path}: dataset_name={manifest['dataset_name']!r} does not match "
                f"configuration dataset_name={config.dataset_name!r}."
            )

        branch_images: dict[str, Path] = {}
        for branch in config.temperature_runs:
            for replica_index, _random_seed in enumerate(config.random_seeds):
                run_name = f"{branch.name}/replica_{replica_index:03d}"
                branch_dir = dataset_root / run_name
                metadata_path = branch_dir / "metadata.json"
                trajectory_path = branch_dir / "trajectory.npz"
                for path in (metadata_path, trajectory_path):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Cannot create structure slice: required artifact is missing: "
                            f"{path}."
                        )
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                slab_bounds = tuple(metadata["source"]["slab_bounds_fractional"])
                trace = load_transition_trace(trajectory_path)
                visualization_dir = branch_dir / "visualizations"
                visualization_dir.mkdir(exist_ok=True)
                image_path = visualization_dir / "structure_slice.png"
                progress(f"{dataset_root.name}/{run_name}: rendering structure slices")
                write_structure_slice_visualization(
                    image_path,
                    trace=trace,
                    chemical_symbol=config.generator.system.chemical_symbol,
                    timestep_fs=config.generator.dynamics.timestep_fs,
                    reference_planes_fractional=(
                        float(slab_bounds[0]),
                        float(slab_bounds[1]),
                    ),
                    simulation_name=run_name,
                    temperature_K=branch.temperature_K,
                    ptm_rmsd_cutoff=config.analysis.ptm_rmsd_cutoff,
                )
                if replica_index == 0:
                    branch_images[branch.name] = image_path
        write_structure_slice_overview(
            dataset_root / "structure_slice_overview.png",
            branch_images,
        )
        progress(f"Wrote transition structure slices to {dataset_root}")
