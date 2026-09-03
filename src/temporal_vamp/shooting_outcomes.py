"""Physical endpoint labels for fixed-duration crystallization shooting branches."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.shooting_dataset import (
    ShootingCampaignSnapshot,
    resolve_shooting_trajectory_path,
    shooting_snapshot_sha256,
)


@dataclass(frozen=True)
class ShootingEndpointOutcomes:
    path: Path
    document: dict[str, Any]
    committed_crystal: np.ndarray
    committed_liquid: np.ndarray
    censored: np.ndarray
    crystalline_fraction: np.ndarray
    largest_cluster_atoms: np.ndarray
    branch_parent_index: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingEndpointOutcomes":
        root = Path(path).expanduser().resolve()
        document_path = root / "outcomes.json"
        arrays_path = root / "outcomes.npz"
        if not document_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(
                f"Shooting endpoint outcomes require {document_path} and {arrays_path}."
            )
        with document_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        with np.load(arrays_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        branch_count = int(document["branch_count"])
        persistence_frames = int(document["spec"]["persistence_frames"])
        expected_shapes = {
            "committed_crystal": (branch_count,),
            "committed_liquid": (branch_count,),
            "censored": (branch_count,),
            "crystalline_fraction": (branch_count, persistence_frames),
            "largest_cluster_atoms": (branch_count, persistence_frames),
            "branch_parent_index": (branch_count,),
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise RuntimeError(
                    f"Endpoint outcome shape changed for {name}: "
                    f"expected={expected}, observed={arrays[name].shape}, root={root}."
                )
        return cls(path=root, document=document, **arrays)


def classify_endpoint_frames(
    crystalline_fraction: np.ndarray,
    largest_cluster_atoms: np.ndarray,
    *,
    crystal_cluster_threshold_atoms: int,
    maximum_liquid_crystalline_fraction: float,
) -> tuple[bool, bool, bool]:
    crystal = bool(np.all(largest_cluster_atoms >= int(crystal_cluster_threshold_atoms)))
    liquid = bool(
        np.all(largest_cluster_atoms < int(crystal_cluster_threshold_atoms))
        and np.all(crystalline_fraction < float(maximum_liquid_crystalline_fraction))
    )
    return crystal, liquid, not (crystal or liquid)


def _branch_path(snapshot: ShootingCampaignSnapshot, branch: dict[str, Any]) -> Path:
    root = (
        snapshot.root
        if len(snapshot.campaign_roots) == 1
        else Path(str(branch["campaign_root"]))
    )
    return resolve_shooting_trajectory_path(root, branch)


def _analyze_endpoint_task(
    task: tuple[int, str, tuple[int, ...], float, int, float]
) -> tuple[int, np.ndarray, np.ndarray, bool, bool, bool]:
    branch_index, trajectory_path, timesteps, ptm_rmsd_cutoff, threshold, liquid_limit = task
    from ase import Atoms
    from ovito.io.ase import ase_to_ovito
    from ovito.modifiers import ClusterAnalysisModifier, PolyhedralTemplateMatchingModifier

    trajectory = ShootingBinaryTrajectory.load(trajectory_path)
    frames = trajectory.load_position_frames(timesteps)
    crystalline_fraction = np.empty(len(timesteps), dtype=np.float64)
    largest_cluster_atoms = np.empty(len(timesteps), dtype=np.int64)
    numbers = np.full(trajectory.atom_count, 13, dtype=np.int32)
    for frame_index, timestep in enumerate(timesteps):
        frame = frames[timestep]
        atoms = Atoms(
            numbers=numbers,
            positions=np.asarray(frame.positions, dtype=np.float64),
            cell=np.diag(np.asarray(frame.box_lengths, dtype=np.float64)),
            pbc=True,
        )
        data = ase_to_ovito(atoms)
        ptm = PolyhedralTemplateMatchingModifier()
        ptm.rmsd_cutoff = float(ptm_rmsd_cutoff)
        data.apply(ptm)
        structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        crystalline = np.isin(structure_types, np.asarray([1, 2, 3], dtype=np.int32))
        crystalline_fraction[frame_index] = float(crystalline.mean())
        data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
        data.apply(
            ClusterAnalysisModifier(
                cutoff=3.5,
                only_selected=True,
                sort_by_size=True,
            )
        )
        largest_cluster_atoms[frame_index] = int(
            data.attributes["ClusterAnalysis.largest_size"]
        )
    crystal, liquid, censored = classify_endpoint_frames(
        crystalline_fraction,
        largest_cluster_atoms,
        crystal_cluster_threshold_atoms=int(threshold),
        maximum_liquid_crystalline_fraction=float(liquid_limit),
    )
    return (
        branch_index,
        crystalline_fraction,
        largest_cluster_atoms,
        crystal,
        liquid,
        censored,
    )


def analyze_shooting_endpoint_outcomes(
    snapshot: ShootingCampaignSnapshot,
    *,
    output_path: str | Path,
    persistence_frames: int,
    ptm_rmsd_cutoff: float,
    crystal_cluster_threshold_atoms: int,
    maximum_liquid_crystalline_fraction: float,
    workers: int,
) -> ShootingEndpointOutcomes:
    target = Path(output_path).expanduser().resolve()
    sample_interval_steps = int(snapshot.manifest["protocol"]["sample_interval_steps"])
    run_steps = int(snapshot.manifest["protocol"]["run_steps"])
    frame_count = int(persistence_frames)
    timesteps = tuple(
        range(
            run_steps - (frame_count - 1) * sample_interval_steps,
            run_steps + 1,
            sample_interval_steps,
        )
    )
    spec = {
        "snapshot_sha256": shooting_snapshot_sha256(snapshot),
        "endpoint_timesteps": list(timesteps),
        "persistence_frames": frame_count,
        "ptm_rmsd_cutoff": float(ptm_rmsd_cutoff),
        "crystal_cluster_threshold_atoms": int(crystal_cluster_threshold_atoms),
        "maximum_liquid_crystalline_fraction": float(
            maximum_liquid_crystalline_fraction
        ),
        "definition": (
            "finite-horizon committed crystal if the repository PTM largest-cluster "
            "threshold is met in every endpoint persistence frame; committed liquid if "
            "all endpoint frames remain below both liquid thresholds; otherwise censored"
        ),
    }
    document_path = target / "outcomes.json"
    if document_path.is_file():
        cached = ShootingEndpointOutcomes.load(target)
        if cached.document["spec"] != spec:
            raise RuntimeError(
                f"Shooting endpoint outcome specification changed at {target}."
            )
        return cached
    if target.exists():
        raise RuntimeError(f"Incomplete shooting endpoint outcome directory exists: {target}.")

    tasks = [
        (
            branch_index,
            str(_branch_path(snapshot, branch)),
            timesteps,
            float(ptm_rmsd_cutoff),
            int(crystal_cluster_threshold_atoms),
            float(maximum_liquid_crystalline_fraction),
        )
        for branch_index, branch in enumerate(snapshot.branches)
    ]
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        results = list(executor.map(_analyze_endpoint_task, tasks, chunksize=1))
    results.sort(key=lambda value: value[0])
    branch_count = len(snapshot.branches)
    crystalline_fraction = np.stack([value[1] for value in results])
    largest_cluster_atoms = np.stack([value[2] for value in results])
    committed_crystal = np.asarray([value[3] for value in results], dtype=bool)
    committed_liquid = np.asarray([value[4] for value in results], dtype=bool)
    censored = np.asarray([value[5] for value in results], dtype=bool)
    parent_index_by_id = {
        str(parent["parent_id"]): index for index, parent in enumerate(snapshot.parents)
    }
    branch_parent_index = np.asarray(
        [parent_index_by_id[str(branch["parent_id"])] for branch in snapshot.branches],
        dtype=np.int32,
    )
    parent_counts = []
    for parent_index, parent in enumerate(snapshot.parents):
        selected = branch_parent_index == parent_index
        parent_counts.append(
            {
                "parent_index": parent_index,
                "parent_id": str(parent["parent_id"]),
                "source_run_id": str(parent["source_run_id"]),
                "source_split": str(parent["source_split"]),
                "temperature_K": float(parent["temperature_K"]),
                "phase": str(parent["phase"]),
                "branches": int(selected.sum()),
                "committed_crystal": int(committed_crystal[selected].sum()),
                "committed_liquid": int(committed_liquid[selected].sum()),
                "censored": int(censored[selected].sum()),
                "finite_horizon_crystal_probability": float(
                    committed_crystal[selected].mean()
                ),
            }
        )
    document = {
        "state": "complete",
        "spec": spec,
        "branch_count": branch_count,
        "counts": {
            "committed_crystal": int(committed_crystal.sum()),
            "committed_liquid": int(committed_liquid.sum()),
            "censored": int(censored.sum()),
        },
        "parents": parent_counts,
    }
    target.mkdir(parents=True)
    np.savez(
        target / "outcomes.npz",
        committed_crystal=committed_crystal,
        committed_liquid=committed_liquid,
        censored=censored,
        crystalline_fraction=crystalline_fraction,
        largest_cluster_atoms=largest_cluster_atoms,
        branch_parent_index=branch_parent_index,
    )
    with document_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return ShootingEndpointOutcomes.load(target)


def plot_shooting_endpoint_outcomes(
    outcomes: ShootingEndpointOutcomes, path: str | Path
) -> None:
    parents = outcomes.document["parents"]
    x = np.arange(len(parents), dtype=np.float64)
    branch_count = np.asarray([value["branches"] for value in parents], dtype=np.float64)
    crystal = np.asarray(
        [value["committed_crystal"] for value in parents], dtype=np.float64
    ) / branch_count
    liquid = np.asarray(
        [value["committed_liquid"] for value in parents], dtype=np.float64
    ) / branch_count
    censored = np.asarray([value["censored"] for value in parents], dtype=np.float64) / branch_count
    labels = [
        f"{int(value['temperature_K'])}K\n{value['phase'].replace('pre_nucleation_', '-')}"
        for value in parents
    ]
    fig, ax = plt.subplots(figsize=(15.0, 5.0))
    ax.bar(x, crystal, color="#2a9d8f", label="committed crystal")
    ax.bar(x, liquid, bottom=crystal, color="#457b9d", label="committed liquid")
    ax.bar(
        x,
        censored,
        bottom=crystal + liquid,
        color="#b8b8b8",
        label="censored",
    )
    ax.set(
        xlim=(-0.7, len(parents) - 0.3),
        ylim=(0.0, 1.0),
        xticks=x,
        xticklabels=labels,
        xlabel="shooting parent (ordered by temperature and source run)",
        ylabel="fraction of 11 sibling branches",
    )
    ax.tick_params(axis="x", labelsize=6)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=180)
    plt.close(fig)


__all__ = [
    "ShootingEndpointOutcomes",
    "analyze_shooting_endpoint_outcomes",
    "classify_endpoint_frames",
    "plot_shooting_endpoint_outcomes",
]
