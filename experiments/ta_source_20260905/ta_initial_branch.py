#!/usr/bin/env python3
"""Run and preserve the prepared, single Ta model_1m 24 ps experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.temporal_lammps_binary import write_temporal_lammps_binary
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_trajectory(run_dir: Path, *, frame_count: int, manifest: dict) -> dict:
    """Decode this campaign's sorted id/type/x/y/z dump in one sequential pass."""
    source = run_dir / "trajectory.lammpstrj"
    atom_count = manifest["source"]["atom_count"]
    expected_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    positions = np.empty((frame_count, atom_count, 3), dtype=np.float32)
    timesteps = np.empty(frame_count, dtype=np.int64)
    box_low = np.empty((frame_count, 3), dtype=np.float32)
    box_high = np.empty_like(box_low)
    started = time.monotonic()
    with source.open("r") as handle:
        for frame in range(frame_count):
            header = TemporalLAMMPSDumpDataset._read_frame_header(handle, source_path=source)
            if header is None or header["num_atoms"] != atom_count:
                raise RuntimeError(f"Missing or incorrect atom count at frame {frame}: {source}")
            if tuple(header["atom_columns"]) != ("id", "type", "x", "y", "z"):
                raise RuntimeError(f"Unexpected columns at frame {frame}: {header['atom_columns']}")
            timesteps[frame] = header["timestep"]
            if timesteps[frame] != frame * manifest["protocol"]["dump_every_steps"]:
                raise RuntimeError(f"Unexpected timestep {timesteps[frame]} at frame {frame}: {source}")
            table = np.loadtxt(handle, max_rows=atom_count)
            if table.shape != (atom_count, 5):
                raise RuntimeError(f"Incomplete frame {frame}: shape={table.shape}, source={source}")
            if not np.array_equal(table[:, 0], expected_ids) or not np.all(table[:, 1] == 1):
                raise RuntimeError(f"Ta atom IDs/types changed at frame {frame}: {source}")
            box_low[frame] = header["box_low"]
            box_high[frame] = header["box_high"]
            lengths = box_high[frame] - box_low[frame]
            # Match the repository's float32 periodic-coordinate decoding.
            coords = table[:, 2:5].astype(np.float32)
            wrapped = np.mod(coords - box_low[frame], lengths)
            positions[frame] = np.minimum(wrapped, np.nextafter(lengths, np.zeros(3, dtype=np.float32)))
            if frame % 10 == 0 or frame == frame_count - 1:
                print(f"Decoded frame {frame + 1}/{frame_count}, elapsed {time.monotonic() - started:.1f} s", flush=True)
        if handle.read().strip():
            raise RuntimeError(f"Unexpected trailing frames/content after {frame_count} frames: {source}")
    binary = write_temporal_lammps_binary(
        run_dir / "trajectory_binary_float32",
        positions=positions,
        timesteps=timesteps,
        box_low=box_low,
        box_high=box_high,
        atom_ids=expected_ids,
        atom_types=np.ones(atom_count, dtype=np.int32),
        atom_columns=("id", "type", "x", "y", "z"),
        source={"trajectory_lammpstrj": str(source), "campaign_manifest": str(Path(manifest["potential"]["file"]).parents[1] / "manifest.json")},
        provenance={"conversion_script": str(Path(__file__).resolve()), "material": "Ta", "potential_sha256": manifest["potential"]["sha256"]},
    )
    checksums = binary.verify_checksums()
    return {"binary_path": str(binary.root), "frame_count": frame_count, "atom_count": atom_count,
            "storage_dtype": "float32", "conversion_seconds": time.monotonic() - started,
            "checksums": checksums, "raw_text_preserved": True}


def run(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    branch = root / manifest["branch_path"]
    if (root / "status.json").exists():
        raise FileExistsError(f"Refusing to duplicate or overwrite a previous run: {root / 'status.json'}")
    preflight = json.loads((root / "preflight.json").read_text())
    if preflight["state"] != "passed":
        raise RuntimeError(f"Ta preflight has not passed: {root / 'preflight.json'}")
    for filename, expected in ((manifest["source"]["path"], manifest["source"]["sha256"]),
                               (manifest["potential"]["file"], manifest["potential"]["sha256"])):
        path = Path(filename)
        if sha256(path) != expected:
            raise RuntimeError(f"Prepared simulation input changed: {path}")
    environment = os.environ.copy()
    environment.update({"LD_LIBRARY_PATH": str(Path(sys.prefix) / "lib"), "OMP_NUM_THREADS": "1",
                        "OMP_DYNAMIC": "FALSE", "MPIR_CVAR_CH4_NETMOD": "ofi", "FI_PROVIDER": "tcp"})
    command = [manifest["execution"]["mpiexec"], "-n", str(manifest["protocol"]["mpi_ranks"]),
               "-bind-to", "core", manifest["execution"]["lammps"], "-in", "in.lammps", "-log", "log.lammps"]
    status = {"state": "running", "stage": "dynamics", "started_at_utc": utc_now(),
              "pid": os.getpid(), "host": socket.gethostname(), "command": command,
              "branch_path": str(branch), "preflight": preflight}
    write_json(root / "status.json", status)
    started = time.monotonic()
    try:
        with (branch / "stdout.log").open("xb") as output:
            subprocess.run(command, cwd=branch, env=environment, stdin=subprocess.DEVNULL,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        lines = (branch / "lammps.log").read_text().splitlines()
        if "TA_24PS_BRANCH_COMPLETE" not in lines:
            raise RuntimeError(f"LAMMPS returned without its completion marker: {branch / 'lammps.log'}")
        restart = branch / "final.restart.bin"
        if restart.stat().st_size == 0:
            raise RuntimeError(f"Empty final restart: {restart}")
        status.update(stage="binary_conversion", dynamics_finished_at_utc=utc_now(),
                      dynamics_seconds=time.monotonic() - started)
        write_json(root / "status.json", status)
        conversion = convert_trajectory(branch, frame_count=manifest["protocol"]["frame_count"], manifest=manifest)
        status.update(state="complete", stage="complete", completed_at_utc=utc_now(),
                      elapsed_seconds=time.monotonic() - started, conversion=conversion,
                      final_restart_sha256=sha256(restart))
        write_json(branch / "outcome.json", status)
        write_json(root / "status.json", status)
        manifest.update(state="complete", completed_at_utc=status["completed_at_utc"])
        write_json(root / "manifest.json", manifest)
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as error:
        status.update(state="failed", failed_at_utc=utc_now(), error=repr(error), traceback=traceback.format_exc())
        write_json(root / "status.json", status)
        write_json(branch / "outcome.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--convert-preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.convert_preflight:
        manifest = json.loads((args.campaign_root / "manifest.json").read_text())
        report = convert_trajectory(args.campaign_root / "preflight", frame_count=11, manifest=manifest)
        write_json(args.campaign_root / "preflight" / "binary_conversion.json", report)
        print(json.dumps(report, indent=2), flush=True)
    else:
        run(args.campaign_root.resolve())


if __name__ == "__main__":
    main()
