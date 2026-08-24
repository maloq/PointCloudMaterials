#!/usr/bin/env python3
"""Export one repository homogeneous trajectory to tracked-atom LAMMPS text."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TRAJECTORY_KEYS = {
    "step",
    "temperature_K",
    "pressure_GPa",
    "volume_A3",
    "potential_energy_eV_per_atom",
    "positions_A",
    "cell_vectors_A",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def export_homogeneous_trajectory_lammps(
    trajectory_path: Path,
    output_path: Path,
) -> None:
    with np.load(trajectory_path) as stored:
        if set(stored.files) != TRAJECTORY_KEYS:
            raise RuntimeError(
                f"{trajectory_path}: expected arrays={sorted(TRAJECTORY_KEYS)}, "
                f"got={sorted(stored.files)}."
            )
        step = stored["step"]
        positions_A = stored["positions_A"]
        cell_vectors_A = stored["cell_vectors_A"]

    if (
        positions_A.ndim != 3
        or positions_A.shape[2] != 3
        or cell_vectors_A.shape != (positions_A.shape[0], 3, 3)
        or step.shape != (positions_A.shape[0],)
    ):
        raise RuntimeError(
            f"{trajectory_path}: incompatible step={step.shape}, "
            f"positions_A={positions_A.shape}, cell_vectors_A={cell_vectors_A.shape}."
        )
    off_diagonal = cell_vectors_A.copy()
    off_diagonal[:, np.arange(3), np.arange(3)] = 0.0
    if np.any(off_diagonal != 0.0):
        raise RuntimeError(
            f"{trajectory_path}: temporal LAMMPS analysis currently requires orthogonal "
            f"cells; maximum off-diagonal component is "
            f"{float(np.max(np.abs(off_diagonal)))} A."
        )
    cell_lengths_A = cell_vectors_A[:, np.arange(3), np.arange(3)]
    if np.any(cell_lengths_A <= 0.0):
        raise RuntimeError(
            f"{trajectory_path}: cell lengths must be positive, got range="
            f"[{float(cell_lengths_A.min())}, {float(cell_lengths_A.max())}] A."
        )
    if len(step) > 1 and np.any(np.diff(step) <= 0):
        raise RuntimeError(
            f"{trajectory_path}: trajectory steps must increase strictly, got "
            f"{step.tolist()}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    atom_count = positions_A.shape[1]
    atom_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    atom_types = np.ones(atom_count, dtype=np.int32)
    with temporary.open("w", encoding="utf-8") as handle:
        for frame_index, frame_step in enumerate(step):
            lengths = cell_lengths_A[frame_index]
            wrapped = np.mod(positions_A[frame_index], lengths[None, :])
            handle.write(
                "ITEM: TIMESTEP\n"
                f"{int(frame_step)}\n"
                "ITEM: NUMBER OF ATOMS\n"
                f"{atom_count}\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                f"0 {lengths[0]:.12g}\n"
                f"0 {lengths[1]:.12g}\n"
                f"0 {lengths[2]:.12g}\n"
                "ITEM: ATOMS id type x y z\n"
            )
            np.savetxt(
                handle,
                np.column_stack((atom_ids, atom_types, wrapped)),
                fmt=("%d", "%d", "%.9g", "%.9g", "%.9g"),
            )
            print(
                f"{trajectory_path.name}: exported frame {frame_index + 1}/"
                f"{len(step)} at measurement step {int(frame_step)}",
                flush=True,
            )
    temporary.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes).")


def main() -> None:
    args = _arguments()
    export_homogeneous_trajectory_lammps(
        args.trajectory.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
