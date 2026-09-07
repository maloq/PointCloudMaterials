"""Shared I/O for the repository's 70,304-atom Aluminum campaigns.

These readers intentionally retain the original atom-count and column contract.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

EXPECTED_ATOM_COUNT = 70_304


EXPECTED_POTENTIAL_SHA256 = (
    "60c8a085be79d273324ab421f5b1447578fef55c1acfc6492c0999f15ee8a284"
)


PRESSURE_BAR_TO_GPA = 1.0e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _read_lammps_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: list[int] = []
    positions: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    expected_ids = np.arange(1, EXPECTED_ATOM_COUNT + 1, dtype=np.int64)
    with path.open("r", encoding="utf-8") as handle:
        while True:
            marker = handle.readline()
            if not marker:
                break
            if marker != "ITEM: TIMESTEP\n":
                raise RuntimeError(
                    f"{path}: expected 'ITEM: TIMESTEP', got {marker.rstrip()!r}."
                )
            step = int(handle.readline())
            if handle.readline() != "ITEM: NUMBER OF ATOMS\n":
                raise RuntimeError(f"{path}: missing atom-count header at step {step}.")
            atom_count = int(handle.readline())
            if atom_count != EXPECTED_ATOM_COUNT:
                raise RuntimeError(
                    f"{path}: step {step} has {atom_count} atoms, expected "
                    f"{EXPECTED_ATOM_COUNT}."
                )
            bounds_header = handle.readline()
            if bounds_header != "ITEM: BOX BOUNDS pp pp pp\n":
                raise RuntimeError(
                    f"{path}: unsupported box header at step {step}: "
                    f"{bounds_header.rstrip()!r}."
                )
            bounds = np.asarray(
                [[float(value) for value in handle.readline().split()] for _ in range(3)],
                dtype=np.float64,
            )
            atom_header = handle.readline()
            if atom_header != "ITEM: ATOMS id type x y z\n":
                raise RuntimeError(
                    f"{path}: unsupported atom columns at step {step}: "
                    f"{atom_header.rstrip()!r}."
                )
            table = np.loadtxt(handle, max_rows=atom_count)
            if table.shape != (atom_count, 5):
                raise RuntimeError(
                    f"{path}: step {step} atom table has shape {table.shape}, expected "
                    f"({atom_count}, 5)."
                )
            ids = table[:, 0].astype(np.int64)
            if not np.array_equal(ids, expected_ids):
                raise RuntimeError(
                    f"{path}: atom IDs are not the exact sorted sequence 1..{atom_count} "
                    f"at step {step}."
                )
            box_lengths = bounds[:, 1] - bounds[:, 0]
            frame_positions = table[:, 2:5] - bounds[:, 0][None, :]
            steps.append(step)
            positions.append(frame_positions.astype(np.float32))
            cells.append(np.diag(box_lengths))
    if not steps:
        raise RuntimeError(f"{path}: trajectory contains no frames.")
    return (
        np.asarray(steps, dtype=np.int64),
        np.stack(positions),
        np.stack(cells),
    )


def _read_thermodynamic_log(path: Path) -> dict[int, tuple[float, float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_fields = ("Step", "Temp", "Press", "Volume", "PotEng")
    header_indices = [
        index for index, line in enumerate(lines) if tuple(line.split()) == header_fields
    ]
    if not header_indices:
        raise RuntimeError(
            f"{path}: did not find the configured thermo columns {header_fields}."
        )
    rows: dict[int, tuple[float, float, float, float]] = {}
    for header_index in header_indices:
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if stripped.startswith("Loop time of"):
                break
            fields = stripped.split()
            if len(fields) != 5:
                continue
            try:
                step = int(fields[0])
                values = tuple(float(value) for value in fields[1:])
            except ValueError:
                continue
            previous = rows.get(step)
            if previous is not None and not np.allclose(
                previous, values, rtol=1.0e-12, atol=1.0e-9
            ):
                raise RuntimeError(
                    f"{path}: conflicting thermo samples for step {step}: "
                    f"{previous} and {values}."
                )
            rows[step] = values  # type: ignore[assignment]
    if not rows:
        raise RuntimeError(f"{path}: thermo blocks contain no numeric samples.")
    return rows
