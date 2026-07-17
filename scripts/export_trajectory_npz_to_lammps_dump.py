from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a repository atomistic trajectory.npz as an orthorhombic "
            "LAMMPS text dump with stable atom IDs."
        )
    )
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--atom-type",
        type=int,
        default=1,
        help="LAMMPS atom type written for every atom (default: 1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def _load_trajectory(
    trajectory_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved = trajectory_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Trajectory NPZ does not exist: {resolved}")

    with np.load(resolved) as trajectory:
        required = {"step", "positions_A", "cell_vectors_A"}
        missing = sorted(required.difference(trajectory.files))
        if missing:
            raise KeyError(
                f"{resolved} is missing required trajectory arrays: {missing}"
            )
        steps = np.asarray(trajectory["step"], dtype=np.int64)
        positions = np.asarray(trajectory["positions_A"], dtype=np.float32)
        cells = np.asarray(trajectory["cell_vectors_A"], dtype=np.float64)

    if steps.ndim != 1:
        raise ValueError(f"{resolved}: step must have shape (F,), got {steps.shape}.")
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"{resolved}: positions_A must have shape (F, N, 3), got {positions.shape}."
        )
    if cells.shape != (positions.shape[0], 3, 3):
        raise ValueError(
            f"{resolved}: cell_vectors_A must have shape {(positions.shape[0], 3, 3)}, "
            f"got {cells.shape}."
        )
    if steps.shape[0] != positions.shape[0]:
        raise ValueError(
            f"{resolved}: step has {steps.shape[0]} frames but positions_A has "
            f"{positions.shape[0]} frames."
        )
    if positions.shape[1] == 0:
        raise ValueError(f"{resolved}: trajectory contains zero atoms.")
    if np.any(np.diff(steps) <= 0):
        raise ValueError(
            f"{resolved}: step values must be strictly increasing, got {steps.tolist()}."
        )
    if not np.isfinite(positions).all():
        raise ValueError(f"{resolved}: positions_A contains non-finite values.")
    if not np.isfinite(cells).all():
        raise ValueError(f"{resolved}: cell_vectors_A contains non-finite values.")

    off_diagonal = cells.copy()
    off_diagonal[:, np.arange(3), np.arange(3)] = 0.0
    maximum_tilt = float(np.max(np.abs(off_diagonal)))
    if maximum_tilt > 1.0e-10:
        raise NotImplementedError(
            "The temporal LAMMPS loader currently supports orthorhombic boxes only. "
            f"{resolved} has maximum off-diagonal cell component {maximum_tilt:.6g} A."
        )
    lengths = np.diagonal(cells, axis1=1, axis2=2)
    if np.any(lengths <= 0.0):
        bad_frames = np.flatnonzero(np.any(lengths <= 0.0, axis=1)).tolist()
        raise ValueError(
            f"{resolved}: non-positive orthorhombic cell lengths at frames {bad_frames}."
        )
    return steps, positions, lengths


def _write_lammps_dump(
    *,
    output_path: Path,
    steps: np.ndarray,
    positions: np.ndarray,
    box_lengths: np.ndarray,
    atom_type: int,
    force: bool,
) -> None:
    resolved = output_path.expanduser().resolve()
    if resolved.exists() and not force:
        raise FileExistsError(
            f"Output dump already exists: {resolved}. Pass --force to replace it."
        )
    if atom_type <= 0:
        raise ValueError(f"atom_type must be positive, got {atom_type}.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    atom_count = int(positions.shape[1])
    atom_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    atom_types = np.full(atom_count, int(atom_type), dtype=np.int64)
    table = np.empty((atom_count, 5), dtype=np.float64)
    table[:, 0] = atom_ids
    table[:, 1] = atom_types

    try:
        with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
            for frame_index, timestep in enumerate(steps.tolist()):
                lengths = box_lengths[frame_index]
                wrapped = np.mod(
                    positions[frame_index].astype(np.float64, copy=False),
                    lengths[None, :],
                )
                table[:, 2:] = wrapped

                handle.write("ITEM: TIMESTEP\n")
                handle.write(f"{int(timestep)}\n")
                handle.write("ITEM: NUMBER OF ATOMS\n")
                handle.write(f"{atom_count}\n")
                handle.write("ITEM: BOX BOUNDS pp pp pp\n")
                for length in lengths.tolist():
                    handle.write(f"0 {float(length):.12g}\n")
                handle.write("ITEM: ATOMS id type x y z\n")
                np.savetxt(
                    handle,
                    table,
                    fmt=("%d", "%d", "%.9g", "%.9g", "%.9g"),
                )
        os.replace(temporary, resolved)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = _parse_args()
    steps, positions, box_lengths = _load_trajectory(args.trajectory)
    _write_lammps_dump(
        output_path=args.output,
        steps=steps,
        positions=positions,
        box_lengths=box_lengths,
        atom_type=int(args.atom_type),
        force=bool(args.force),
    )
    resolved_output = args.output.expanduser().resolve()
    print(
        "Exported temporal LAMMPS dump: "
        f"path={resolved_output}, frames={positions.shape[0]}, "
        f"atoms={positions.shape[1]}, first_step={int(steps[0])}, "
        f"last_step={int(steps[-1])}, bytes={resolved_output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
