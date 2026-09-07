# Trajectory format conversion

Use one maintained command from the repository root:

```bash
conda run -n pointnet python scripts/convert_trajectory.py --help
```

This tool wraps the existing repository converters. It is not a general-purpose
LAMMPS importer: the two binary formats have different producers and required
fields. In-place conversion keeps source text by default and publishes the
existing reader metadata. No real datasets are converted by the code cleanup.

## Shooting trajectories (positions and velocities)

```bash
conda run -n pointnet python scripts/convert_trajectory.py shooting \
  --campaign-root /path/to/shooting_campaign --workers 2
conda run -n pointnet python scripts/convert_trajectory.py audit-shooting \
  /path/to/shooting_campaign
```

Converts complete branches selected by `manifest.json` and `outcome.json` into
`trajectory_binary_float32/`. Incomplete branches are counted and left alone.
Checks include campaign timesteps, atom count, source SHA-256, array checksums,
and float32 positions/velocities against the converter's semantic source hashes.
The outcome points the existing reader at the binary even when the dump remains.
Workers are limited to the existing range 1–8; the campaign lock is retained.

To reclaim space after verification, run the same command with `--delete-source`.
Only verified branch-root `trajectory.lammpstrj` files are removed. Restart files
and archived partial trajectories remain. Rerunning verifies and reuses the
existing binary before deleting retained text. `audit-shooting` is read-only;
add `--require-source-deleted` to require reclamation as well as valid conversion.

## Ordinary temporal trajectories (positions)

```bash
conda run -n pointnet python scripts/convert_trajectory.py temporal \
  /path/to/completed_campaign
conda run -n pointnet python scripts/convert_trajectory.py audit-temporal \
  /path/to/completed_campaign
```

Requires a complete campaign `status.json` and each replica's `analysis.json`
with the checksum-bound `trajectory.npz` produced by this repository. That
archive supplies float32 coordinates; the dump supplies timeline and box metadata
and any unarchived prefix. The original orthogonal-cell and position-column
requirements are preserved. This is not a converter for arbitrary external dumps.

Outputs are `trajectory_binary_float32/`, the conversion report, and the existing
analysis artifact record. Checks cover the archive checksum, cadence, cells,
array checksums and stored positions. Add `--delete-source` to remove verified
text dumps; add `--require-source-deleted` to the read-only audit to require it.
The NPZ archive is retained because it is part of the recorded provenance.

## Elemental source and branch trajectories

`python scripts/convert_trajectory.py elemental /path/to/branch` converts the
completed Ti/Ta elemental producer's sorted position dump, using its
`metadata.json` for exact atom count, timeline, and provenance. The campaign
invokes it automatically. It retains text, rejects incomplete/extra frames and
changed IDs/types, verifies source and array hashes, and compares a semantic
coordinate hash after writing the float32 temporal binary. See the
[Ti/Ta protocol](../experiments/ti_ta_crystallization_20260907/README.md).

Use `elemental /path/to/branch --delete-source` to remove the original dump
after these checks pass. Its conversion report retains the source hash and
records reclaimed bytes. Source text is retained by default. An interrupted
scratch/build directory requires explicit inspection and cleanup before retry;
do not rerun completed dynamics to recover a failed format conversion.
The elemental converter now publishes its validated scratch positions array
without making a second full-size copy; checksums and semantic coordinate
verification still run before deletion of any source text.

## Derived training caches

```bash
python scripts/convert_trajectory.py training-cache /path/to/plan.json
```

The explicit JSON plan has `files`, a list of absolute cache paths. Supported
repository products are `clouds.npy`, `benchmark_clouds.npy`, `*.views.npy`, and
`*.context.npy`: centered neighborhood xyz arrays, not full simulation coordinates.
The command rejects non-finite and float16-overflowing values, writes a temporary
array, verifies it against the float16 rounding of the original, records source
and output SHA-256 plus rounding errors, and atomically replaces the original.
Per-file `*.float16.json` records support verified repeated execution. Interrupted
temporary arrays cause an explicit error and need inspection before retry.

Optional `wait_for_processes` entries contain `host`, `pid`, and `start_ticks`
(field 22 of Linux `/proc/PID/stat`, stored as a string). Conversion waits until
those exact processes exit, avoiding accidental waiting on a reused PID. Run the
guarded command on that host, with its output captured; the plan's `.status.json`
reports waiting, progress, completion, or failure. A host reboot or allocation
shutdown can terminate a watcher; inspect status and restart the command if needed.

Current neighborhood producers use float16 storage, and readers decode to float32
for geometry and model input. Float32 targets, normalization statistics, physical
metadata, integer IDs, and canonical simulation trajectories are preserved.
Float16 storage is lossy and is distinct from bf16 computation. For an active
ablation queue, defer conversion of all referenced caches until it finishes so
that runs use the same cached values. Precision migrations are recorded in the
sidecars; an older producer manifest may describe the original float32 creation.

## Separate exports

```bash
conda run -n pointnet python scripts/convert_trajectory.py export-shooting \
  --campaign-root /path/to/campaign --output-root /path/to/export \
  --all-complete --dtype float32 --no-benchmark
conda run -n pointnet python scripts/convert_trajectory.py export-npz-dump \
  /path/to/trajectory.npz /path/to/trajectory.lammpstrj
```

`export-shooting` leaves the source campaign unchanged, supports selected branches,
float32/float16 precision comparisons and optional benchmarks. Its historical
default exports both precisions; use `--dtype float32` for a single representation.
It does not install an in-place reader artifact. `export-npz-dump` exports the
repository NPZ positions to an orthogonal LAMMPS text dump.

## Compatibility and records

Implementation lives in `src/data_utils/conversion/`. The old standalone
`migrate_lammps_*`, audit and export scripts have been replaced by this command.
The old shooting deletion flag `--delete-originals` is now `--delete-source`.
Existing on-disk names such as `binary_migration_float32.json`, lock names,
schema fields and format identifiers are deliberately retained so existing
readers and conversion records remain usable. A negative `freed_apparent_bytes`
means retained text plus new binaries occupy more space than the original text.

## Float16 simulation positions

New elemental trajectories default to float16 positions; use
`elemental BRANCH --storage-dtype float16 --delete-source`. Existing verified
binaries use `temporal-storage BINARY_DIR... --delete-source`. This is lossy
rounding of stored positions, not a change in MD integration precision. Both
commands record maximum and RMS minimum-image coordinate error in angstrom.
Box bounds remain float32; timesteps, IDs and types remain exact. Float16 readers
decode to float32 and wrap into the periodic box. Source files are removed only
after the target has been written, fsynced and checksum verified. Binary migration
keeps the original manifest in provenance and old paths as compatibility symlinks.
The elemental `binary_conversion.json` is updated for campaign resumption.
