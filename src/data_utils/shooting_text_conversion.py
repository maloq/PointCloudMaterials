"""Offline parser used only to convert repository LAMMPS shooting dumps.

Training and analysis must use ``shooting_binary``. Keeping the expensive ASCII
reader in this explicitly conversion-only module prevents an accidental fallback
to text parsing when a campaign has not been migrated correctly.
"""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data_utils.shooting_dataset import ShootingFrame


_FRAME_MARKER = b"ITEM: TIMESTEP\n"
_SHOOTING_COLUMNS = ("id", "type", "x", "y", "z", "vx", "vy", "vz")


def _readline_ascii(mapped: mmap.mmap, *, path: Path) -> str:
    raw = mapped.readline()
    if not raw:
        raise RuntimeError(f"Unexpected end of file while reading selected frame from {path}.")
    return raw.decode("ascii").rstrip("\n")


def load_lammps_shooting_frames_for_conversion(
    trajectory_path: str | Path,
    *,
    timesteps: Sequence[int],
    atom_count: int,
) -> dict[int, ShootingFrame]:
    """Parse selected frames from a source dump before one-time binary migration."""

    path = Path(trajectory_path).expanduser().resolve()
    requested = tuple(sorted({int(value) for value in timesteps}))
    if not requested or requested[0] < 0:
        raise ValueError(
            f"Requested shooting timesteps must be nonempty and nonnegative: {requested}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS shooting trajectory is missing: {path}")
    frames: dict[int, ShootingFrame] = {}
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            search_start = 0
            for timestep in requested:
                marker = _FRAME_MARKER + f"{timestep}\n".encode("ascii")
                offset = mapped.find(marker, search_start)
                if offset < 0:
                    raise RuntimeError(
                        f"Requested timestep {timestep} is absent from completed "
                        f"shooting dump {path}."
                    )
                mapped.seek(offset)
                if _readline_ascii(mapped, path=path) != "ITEM: TIMESTEP":
                    raise RuntimeError(f"Invalid timestep marker at byte {offset} in {path}.")
                observed_timestep = int(_readline_ascii(mapped, path=path))
                if observed_timestep != timestep:
                    raise RuntimeError(
                        f"Selected-frame timestep mismatch in {path}: "
                        f"requested={timestep}, observed={observed_timestep}."
                    )
                if _readline_ascii(mapped, path=path) != "ITEM: NUMBER OF ATOMS":
                    raise RuntimeError(
                        f"Missing atom-count header for timestep={timestep} in {path}."
                    )
                observed_atoms = int(_readline_ascii(mapped, path=path))
                if observed_atoms != int(atom_count):
                    raise RuntimeError(
                        f"Atom count mismatch at timestep={timestep} in {path}: "
                        f"expected={atom_count}, observed={observed_atoms}."
                    )
                bounds_header = _readline_ascii(mapped, path=path)
                if bounds_header != "ITEM: BOX BOUNDS pp pp pp":
                    raise RuntimeError(
                        "Shooting conversion requires orthogonal periodic bounds; "
                        f"got {bounds_header!r} at timestep={timestep} in {path}."
                    )
                box_low = np.empty(3, dtype=np.float32)
                box_high = np.empty(3, dtype=np.float32)
                for axis in range(3):
                    values = np.fromstring(_readline_ascii(mapped, path=path), sep=" ")
                    if values.shape != (2,):
                        raise RuntimeError(
                            f"Invalid box-bound line for axis={axis}, "
                            f"timestep={timestep} in {path}."
                        )
                    box_low[axis], box_high[axis] = values
                atom_header = _readline_ascii(mapped, path=path)
                expected_header = "ITEM: ATOMS " + " ".join(_SHOOTING_COLUMNS)
                if atom_header != expected_header:
                    raise RuntimeError(
                        f"Unexpected atom columns at timestep={timestep} in {path}: "
                        f"expected={expected_header!r}, got={atom_header!r}."
                    )
                block_start = mapped.tell()
                next_marker = mapped.find(_FRAME_MARKER, block_start)
                block_end = len(mapped) if next_marker < 0 else next_marker
                table_values = np.fromstring(
                    mapped[block_start:block_end].decode("ascii"),
                    sep=" ",
                    dtype=np.float64,
                )
                expected_values = int(atom_count) * len(_SHOOTING_COLUMNS)
                if table_values.size != expected_values:
                    raise RuntimeError(
                        f"Selected shooting frame has an incomplete atom table: path={path}, "
                        f"timestep={timestep}, expected_values={expected_values}, "
                        f"observed_values={table_values.size}."
                    )
                table = table_values.reshape(int(atom_count), len(_SHOOTING_COLUMNS))
                ids = table[:, 0].astype(np.int64, copy=False)
                order = np.argsort(ids, kind="mergesort")
                ids = ids[order]
                if not np.array_equal(
                    ids, np.arange(1, int(atom_count) + 1, dtype=np.int64)
                ):
                    raise RuntimeError(
                        f"Shooting dump atom IDs are not exactly 1..{atom_count} at "
                        f"timestep={timestep} in {path}."
                    )
                atom_types = table[:, 1].astype(np.int32, copy=False)[order]
                positions = table[:, 2:5].astype(np.float32, copy=False)[order]
                velocities = table[:, 5:8].astype(np.float32, copy=False)[order]
                box_lengths = box_high - box_low
                if np.any(box_lengths <= 0.0):
                    raise RuntimeError(
                        f"Non-positive shooting box length at timestep={timestep} "
                        f"in {path}: {box_lengths.tolist()}."
                    )
                wrapped = np.mod(
                    positions - box_low[None, :], box_lengths[None, :]
                ).astype(np.float32, copy=False)
                wrapped = np.minimum(
                    wrapped,
                    np.nextafter(box_lengths, np.zeros_like(box_lengths))[None, :],
                )
                frames[timestep] = ShootingFrame(
                    timestep=timestep,
                    atom_ids=ids,
                    atom_types=atom_types,
                    positions=wrapped,
                    box_low=box_low,
                    box_high=box_high,
                    velocities=velocities,
                )
                search_start = offset + len(marker)
    return frames


__all__ = ["load_lammps_shooting_frames_for_conversion"]
