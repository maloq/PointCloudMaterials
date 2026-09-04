#!/usr/bin/env python3
"""Write exact per-branch documentation for the 18 completed 24 ps simulations."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_NAME = "pointcloudmaterials.temporal_lammps_trajectory"
BRANCH_README = "README.md"
CAMPAIGN_README = "README.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON document is missing: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(document).__name__}.")
    return document


def _write_text_atomic(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary documentation path: {temporary}.")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _repository_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _gib(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.3f} GiB ({byte_count:,} bytes)"


def _duration(seconds: float) -> str:
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d} h {minutes:02d} min {seconds:02d} s"


def _box_bounds(bounds: list[list[float]]) -> str:
    axes = "xyz"
    return ", ".join(
        f"{axis}=[{float(axis_bounds[0]):.9f}, {float(axis_bounds[1]):.9f}] Å"
        for axis, axis_bounds in zip(axes, bounds, strict=True)
    )


def _parse_thermodynamics(log_path: Path) -> tuple[dict[str, float], dict[str, float], int]:
    columns = (
        "step",
        "time_ps",
        "atoms",
        "temperature_K",
        "pressure_bar",
        "lx_A",
        "ly_A",
        "lz_A",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
    )
    rows: list[dict[str, float]] = []
    dangerous_builds: int | None = None
    in_table = False
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Step") and stripped.split() == [
                "Step",
                "Time",
                "Atoms",
                "Temp",
                "Press",
                "Lx",
                "Ly",
                "Lz",
                "PotEng",
                "KinEng",
                "TotEng",
            ]:
                in_table = True
                continue
            if in_table:
                tokens = stripped.split()
                if len(tokens) == len(columns):
                    try:
                        values = [float(token) for token in tokens]
                    except ValueError:
                        in_table = False
                    else:
                        rows.append(dict(zip(columns, values, strict=True)))
                        continue
                else:
                    in_table = False
            if stripped.startswith("Dangerous builds ="):
                dangerous_builds = int(stripped.split("=", maxsplit=1)[1].strip())
    if not rows:
        raise RuntimeError(f"No thermodynamic table was parsed from {log_path}.")
    if dangerous_builds is None:
        raise RuntimeError(f"No dangerous-neighbor-build count was parsed from {log_path}.")
    return rows[0], rows[-1], dangerous_builds


def _potential_table(potential: dict[str, Any]) -> str:
    files = potential["files"]
    hashes = potential["sha256"]
    if len(files) != len(hashes):
        raise RuntimeError(f"Potential file/hash count differs: {potential}.")
    rows = ["| File | SHA-256 |", "|---|---|"]
    rows.extend(
        f"| `potential/{filename}` | `{sha256}` |"
        for filename, sha256 in zip(files, hashes, strict=True)
    )
    return "\n".join(rows)


def _array_table(binary_manifest: dict[str, Any]) -> str:
    descriptions = binary_manifest["arrays"]
    meanings = {
        "positions": "Wrapped Cartesian position relative to `box_low`, in Å",
        "timesteps": "LAMMPS step for each frame",
        "box_low": "Absolute lower box bound for each frame, in Å",
        "box_high": "Absolute upper box bound for each frame, in Å",
        "atom_ids": "Stable LAMMPS atom IDs; exactly `1..N`",
        "atom_types": "Stable LAMMPS atom type for every atom",
    }
    rows = ["| Array file | dtype | shape | Meaning |", "|---|---:|---:|---|"]
    for name in ("positions", "timesteps", "box_low", "box_high", "atom_ids", "atom_types"):
        description = descriptions[name]
        shape_values = [str(value) for value in description["shape"]]
        shape = "(" + ", ".join(shape_values) + ("," if len(shape_values) == 1 else "") + ")"
        rows.append(
            f"| `{description['file']}` | `{description['dtype']}` | `{shape}` | "
            f"{meanings[name]} |"
        )
    return "\n".join(rows)


def _render_branch(
    campaign_root: Path,
    campaign: dict[str, Any],
    material: str,
    snapshot: str,
    archive: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    branch_dir = campaign_root / "branches" / material / snapshot
    metadata = _load_json(branch_dir / "metadata.json")
    conversion = _load_json(branch_dir / "binary_conversion_float16.json")
    binary_root = branch_dir / "trajectory_binary_float16"
    binary_manifest = _load_json(binary_root / "manifest.json")
    if metadata["state"] != "complete" or conversion["state"] != "complete":
        raise RuntimeError(
            f"Branch must have complete simulation and conversion reports: {material}/{snapshot}."
        )
    if binary_manifest["state"] != "complete" or binary_manifest["format"] != FORMAT_NAME:
        raise RuntimeError(f"Binary manifest is not complete/recognized: {binary_root}.")
    if metadata["material"] != material or metadata["snapshot"] != snapshot:
        raise RuntimeError(f"Branch identity mismatch in {branch_dir / 'metadata.json'}.")
    if conversion["material"] != material or conversion["snapshot"] != snapshot:
        raise RuntimeError(f"Branch identity mismatch in {branch_dir / 'binary_conversion_float16.json'}.")

    initial_thermo, final_thermo, dangerous_builds = _parse_thermodynamics(
        branch_dir / "lammps.log"
    )
    protocol = campaign["protocol"]
    potential = campaign["potentials"][material]
    quantization = conversion["position_quantization"]
    geometry = conversion["local_geometry"]
    source_path = _repository_path(metadata["source"])
    thermostat_damping_ps = (
        float(metadata["timestep_fs"]) * int(protocol["thermostat_damping_timesteps"]) / 1000.0
    )
    barostat_damping_ps = (
        float(metadata["timestep_fs"]) * int(protocol["barostat_damping_timesteps"]) / 1000.0
    )
    warning = ""
    if material == "Zr":
        warning = (
            "\n\n> **Zr compatibility limitation:** the original machine-readable Zr potential "
            "was unavailable. This run uses the reconstructed 2NN-MEAM files listed below. "
            "They produce a large negative initial pressure for the archived box, so this "
            "trajectory includes strong barostat relaxation and must be treated as provisional.\n"
        )
    source_size = int(conversion["source"]["size_bytes"])
    binary_size = int(conversion["binary"]["apparent_bytes"])
    frame_count = int(metadata["expected_frames"])
    atom_count = int(metadata["atom_count"])
    generated_at = datetime.now(timezone.utc).isoformat()
    trajectory_location = "`trajectory.lammpstrj`"
    restart_location = "`final.restart.bin`"
    archive_note = ""
    if archive is not None:
        archived = {
            str(entry["filename"]): entry
            for entry in archive["artifacts"]
            if entry["material"] == material and entry["snapshot"] == snapshot
        }
        if set(archived) != {"trajectory.lammpstrj", "final.restart.bin"}:
            raise RuntimeError(
                f"Archive entries are incomplete for {material}/{snapshot}: {archived}."
            )
        trajectory_location = f"`{archived['trajectory.lammpstrj']['archive_path']}`"
        restart_location = f"`{archived['final.restart.bin']['archive_path']}`"
        archive_note = (
            " The large text dump and restart were subsequently moved to the external "
            "archive recorded in the campaign's `lammps_artifacts_archive.json`."
        )

    document = f"""# Simulation and binary data: {material} / {snapshot}

This directory contains one 24 ps molecular-dynamics branch and its verified float16 trajectory.{archive_note} This document was generated from `metadata.json`, `lammps.log`, `binary_conversion_float16.json`, and `trajectory_binary_float16/manifest.json` at {generated_at}.

## Scientific interpretation

The branch starts from the archived coordinate snapshot `{source_path}`. It is a new position-conditioned trajectory: the source supplies positions and the periodic box, but not velocities or Nose–Hoover state. LAMMPS therefore generated new Gaussian velocities using seed `{int(metadata['velocity_seed'])}`. This is **not an exact continuation** of the archived trajectory.{warning}

## Simulation setup

| Field | Value |
|---|---|
| Material / snapshot label | `{material}` / `{snapshot}` |
| Initial-coordinate file | `{source_path}` |
| Source dump timestep | `{int(metadata['source_timestep'])}` |
| Source SHA-256 | `{metadata['source_sha256']}` |
| Source box | {_box_bounds(metadata['source_box_bounds_A'])} |
| Atoms | {atom_count:,} |
| LAMMPS units | `metal` (Å, ps, eV, K, bar) |
| Boundary / atom style | periodic in x/y/z / `atomic` |
| Integrator | isotropic Nose–Hoover NPT |
| Temperature | {float(metadata['temperature_K']):g} K |
| Pressure | {float(protocol['pressure_bar']):g} bar |
| Timestep | {float(metadata['timestep_fs']):g} fs |
| Run length | {int(metadata['duration_steps']):,} steps = {float(metadata['duration_ps']):g} ps |
| Thermostat damping | {int(protocol['thermostat_damping_timesteps'])} steps = {thermostat_damping_ps:g} ps |
| Barostat damping | {int(protocol['barostat_damping_timesteps'])} steps = {barostat_damping_ps:g} ps |
| Drift removal | linear momentum every 100 steps |
| Saved frames | {frame_count} including steps 0 and {int(metadata['duration_steps'])} |
| Sampling interval | {int(metadata['sample_interval_steps'])} steps = {float(metadata['sample_interval_ps']):g} ps |
| Parallel execution | {int(campaign['execution']['mpi_ranks'])} MPI ranks, one branch at a time |
| Simulation elapsed time | {_duration(float(metadata['elapsed_seconds']))} |
| Completed | `{metadata['completed_at_utc']}` |

The exact executable command and all branch-specific values are retained in `metadata.json`; the executable LAMMPS input is `in.lammps`.

### Interatomic potential

Paths in this table are relative to the campaign directory.

{_potential_table(potential)}

Provenance: {potential['provenance']}

### Observed endpoints

| Quantity | Step 0 | Final step {int(final_thermo['step'])} |
|---|---:|---:|
| Time (ps) | {initial_thermo['time_ps']:.6g} | {final_thermo['time_ps']:.6g} |
| Temperature (K) | {initial_thermo['temperature_K']:.9g} | {final_thermo['temperature_K']:.9g} |
| Pressure (bar) | {initial_thermo['pressure_bar']:.9g} | {final_thermo['pressure_bar']:.9g} |
| Box length x (Å) | {initial_thermo['lx_A']:.9g} | {final_thermo['lx_A']:.9g} |
| Box length y (Å) | {initial_thermo['ly_A']:.9g} | {final_thermo['ly_A']:.9g} |
| Box length z (Å) | {initial_thermo['lz_A']:.9g} | {final_thermo['lz_A']:.9g} |

LAMMPS reported `{dangerous_builds}` dangerous neighbor-list builds.

## Files in this branch

| Path | Description |
|---|---|
| `trajectory.lammpstrj` (archived) | Original LAMMPS custom text dump; {_gib(source_size)}; columns `id type x y z`; current location: {trajectory_location} |
| `trajectory_binary_float16/` | Verified, memory-mappable binary trajectory; {_gib(binary_size)} |
| `trajectory_binary_float16/manifest.json` | Format, provenance, shapes, dtypes, SHA-256 checksums, and quantization metrics |
| `binary_conversion_float16.json` | Conversion and sampled local-geometry validation report |
| `final.restart.bin` (archived) | LAMMPS binary restart at {float(metadata['duration_ps']):g} ps; current location: {restart_location} |
| `in.lammps` | Exact LAMMPS input |
| `lammps.log`, `log.lammps`, `stdout.log` | Thermodynamic, initialization, performance, and launcher output |
| `metadata.json` | Authoritative branch metadata and source provenance |

## Binary format

The binary directory uses format `{FORMAT_NAME}`, schema version `{int(binary_manifest['schema_version'])}`. It is a directory of standard NumPy `.npy` arrays, not one opaque packed file. Every array can be opened with `numpy.load(..., mmap_mode="r", allow_pickle=False)` without loading the complete trajectory into RAM.

{_array_table(binary_manifest)}

The position shape is frames × atoms × Cartesian components. Frames retain LAMMPS dump order; atoms are sorted by ID in every frame. Convert a stored step to branch time with `time_ps = timestep * {float(metadata['timestep_fs']):g} / 1000`.

`positions.npy` stores wrapped coordinates relative to each frame's `box_low`. The intended semantic domain is `[0, box_high - box_low)` in each periodic dimension. Raw half-precision rounding can place a boundary-near value exactly at the upper box length, so consumers should use the repository frame loader below; it decodes to float32 and clips such values back into the valid half-open domain. Add `box_low` only when absolute LAMMPS coordinates are needed.

```python
from pathlib import Path

import numpy as np

from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset

root = Path("{_repository_path(binary_root)}")
trajectory = TemporalLAMMPSBinaryTrajectory.load(root)
trajectory.verify_checksums()

positions, box_lengths, timestep = TemporalLAMMPSDumpDataset.load_dump_frame_positions(
    root, frame_index=0
)
absolute_positions = positions + np.asarray(trajectory.box_low[0], dtype=np.float32)
time_ps = timestep * {float(metadata['timestep_fs']):g} / 1000.0
```

### Float16 validation

The converter compared all {int(quantization['value_count']):,} coordinates to the repository's float32 text-decoding semantics before discarding its temporary float32 staging array.

| Metric | Value |
|---|---:|
| Coordinate mean absolute error | {float(quantization['mean_absolute_error_A']):.9g} Å |
| Coordinate RMSE | {float(quantization['rmse_A']):.9g} Å |
| Coordinate maximum absolute error | {float(quantization['maximum_absolute_error_A']):.9g} Å |
| 16-neighbor set retention | {100.0 * float(geometry['neighbor_set_retention_fraction']):.6f}% |
| Same-neighbor distance MAE | {float(geometry['same_neighbor_distance_mean_absolute_error_A']):.9g} Å |
| Same-neighbor distance p99 error | {float(geometry['same_neighbor_distance_p99_absolute_error_A']):.9g} Å |
| Same-neighbor distance maximum error | {float(geometry['same_neighbor_distance_maximum_absolute_error_A']):.9g} Å |

The neighborhood validation used {int(geometry['sampled_centers_per_frame'])} deterministic centers, {int(geometry['neighbors_per_center'])} neighbors per center, and frames {geometry['sampled_frame_indices']}. These numbers measure float16 encoding effects; they are not a validation of the interatomic potential or physical protocol.

The binary occupies {100.0 * binary_size / source_size:.3f}% of the text trajectory's size. SHA-256 values for every array and the semantic pre-quantization float32 stream are recorded in `trajectory_binary_float16/manifest.json`.

## Campaign provenance

The simulation protocol follows Becker et al., *Scientific Reports* **12**, 3195 (2022), [doi:10.1038/s41598-022-06963-5](https://doi.org/10.1038/s41598-022-06963-5), with the exact local implementation and qualifications recorded in the campaign `manifest.json`. The campaign ran from `{campaign['started_at_utc']}` through `{campaign['completed_at_utc']}`.
"""
    summary = {
        "material": material,
        "snapshot": snapshot,
        "atoms": atom_count,
        "runtime_seconds": float(metadata["elapsed_seconds"]),
        "source_bytes": source_size,
        "binary_bytes": binary_size,
        "position_rmse_A": float(quantization["rmse_A"]),
        "neighbor_retention": float(geometry["neighbor_set_retention_fraction"]),
        "final_temperature_K": final_thermo["temperature_K"],
        "final_pressure_bar": final_thermo["pressure_bar"],
        "dangerous_builds": dangerous_builds,
    }
    return document, summary


def _render_campaign(
    campaign: dict[str, Any],
    summaries: list[dict[str, Any]],
    archive: dict[str, Any] | None,
) -> str:
    source_bytes = sum(int(summary["source_bytes"]) for summary in summaries)
    binary_bytes = sum(int(summary["binary_bytes"]) for summary in summaries)
    mean_rmse = float(np.mean([summary["position_rmse_A"] for summary in summaries]))
    minimum_retention = min(summary["neighbor_retention"] for summary in summaries)
    archive_summary = "- Large raw LAMMPS artifacts: still inside their branch directories"
    source_text_summary = f"- Source text: {_gib(source_bytes)} and preserved in each branch"
    if archive is not None:
        archive_summary = (
            f"- Large raw LAMMPS artifacts: {_gib(int(archive['total_bytes']))} moved to "
            f"`{archive['archive_root']}`; see `lammps_artifacts_archive.json`"
        )
        source_text_summary = (
            f"- Source text: {_gib(source_bytes)} and preserved in the external archive"
        )
    rows = [
        "| Branch | Atoms | Runtime | Final T (K) | Final P (bar) | float16 RMSE (Å) | 16-NN retained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        rows.append(
            f"| [{summary['material']}/{summary['snapshot']}](branches/"
            f"{summary['material']}/{summary['snapshot']}/README.md) | "
            f"{summary['atoms']:,} | {_duration(summary['runtime_seconds'])} | "
            f"{summary['final_temperature_K']:.6g} | {summary['final_pressure_bar']:.6g} | "
            f"{summary['position_rmse_A']:.6g} | "
            f"{100.0 * summary['neighbor_retention']:.4f}% |"
        )
    table = "\n".join(rows)
    return f"""# Zr, Al, and Mg: six 24 ps branches per material

This campaign contains 18 completed position-conditioned NPT simulations generated from all six archived initial-coordinate snapshots for Al, Mg, and Zr. Each leaf branch directory has a `README.md` with its exact source, simulation parameters, observed endpoints, file inventory, binary schema, loading example, and float16 validation.

## Campaign summary

- Simulation state: complete ({campaign['completed_branch_count']}/18 branches)
- Duration per branch: {float(campaign['protocol']['duration_ps']):g} ps
- Frames per branch: 241 at {float(campaign['protocol']['sample_interval_ps']):g} ps intervals, including time zero
- Ensemble: isotropic Nose–Hoover NPT at 0 bar
- Temperatures: Al 650 K, Mg 600 K, Zr 1250 K
{source_text_summary}
- Float16 binary: {_gib(binary_bytes)} ({100.0 * binary_bytes / source_bytes:.3f}% of text size)
{archive_summary}
- Mean coordinate RMSE from float16 encoding: {mean_rmse:.9g} Å
- Minimum sampled 16-neighbor set retention: {100.0 * minimum_retention:.6f}%
- Dangerous neighbor builds: {sum(int(summary['dangerous_builds']) for summary in summaries)} across all branches

These are new trajectories initialized from archived positions. Source velocities and thermostat/barostat states were unavailable, so they are not exact continuations. The Zr trajectories are provisional: their reconstructed 2NN-MEAM potential produces a large negative initial pressure and substantial barostat relaxation.

## Branches

{table}

## Authoritative machine-readable records

- `manifest.json`: campaign protocol, source/potential provenance, branches, execution, and completion
- `preflight.json`: short-run validation for Al, Mg, and Zr
- `binary_conversion_float16.json`: aggregate conversion state and metrics
- `branches/<material>/<snapshot>/metadata.json`: simulation record
- `branches/<material>/<snapshot>/trajectory_binary_float16/manifest.json`: binary array contract and checksums

See each linked branch document for the format description and safe loading example.
"""


def write_documentation(campaign_root: Path, *, overwrite: bool) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    campaign = _load_json(campaign_root / "manifest.json")
    conversion = _load_json(campaign_root / "binary_conversion_float16.json")
    if campaign["state"] != "complete" or int(campaign["completed_branch_count"]) != 18:
        raise RuntimeError(f"Simulation campaign is not complete: {campaign_root}.")
    if conversion["state"] != "complete" or int(conversion["branch_count"]) != 18:
        raise RuntimeError(f"Float16 conversion campaign is not complete: {campaign_root}.")
    archive_path = campaign_root / "lammps_artifacts_archive.json"
    archive = _load_json(archive_path) if archive_path.is_file() else None
    if archive is not None:
        if archive["state"] != "complete" or int(archive["moved_artifact_count"]) != 36:
            raise RuntimeError(f"LAMMPS artifact archive is not complete: {archive_path}.")

    documents: list[tuple[Path, str]] = []
    summaries: list[dict[str, Any]] = []
    for branch in campaign["branches"]:
        material = str(branch["material"])
        snapshot = str(branch["snapshot"])
        document, summary = _render_branch(
            campaign_root, campaign, material, snapshot, archive
        )
        documents.append(
            (campaign_root / "branches" / material / snapshot / BRANCH_README, document)
        )
        summaries.append(summary)
    documents.append(
        (campaign_root / CAMPAIGN_README, _render_campaign(campaign, summaries, archive))
    )

    existing = [path for path, _ in documents if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing documentation: {existing}.")
    for path, contents in documents:
        _write_text_atomic(path, contents)
        print(f"[docs] wrote {_repository_path(path)}", flush=True)
    print(f"[docs] complete: branch_documents={len(summaries)}, campaign_document=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace documentation previously generated by this script.",
    )
    args = parser.parse_args()
    write_documentation(args.campaign, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
