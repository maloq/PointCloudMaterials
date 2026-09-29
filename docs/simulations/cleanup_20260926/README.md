# Simulation cleanup — 26 September 2026

Completed at the user's request: delete older coarse Al trajectories and failed
attempts. Removed **432 MD payload files, 38.347 GiB (41.175 GB) allocated**.
This is the sum of removed files' allocated blocks, not a filesystem quota delta;
ongoing simulations continue writing elsewhere. All removed files had one hard link.

| Removed payload | Scope | Allocated GiB |
|---|---|---:|
| Older coarse Al | 35 trajectories, 70,304 atoms, 3 ps cadence; 34 × 600 ps and one × 999 ps | 23.043 |
| Failed/interrupted attempts | Obsolete independent-source partial dumps, memory-source failures and their stopped SCRATCH copies, nested/continuation failures, old Ti/Ta preparation | 15.304 |

The four retired coarse collections are the nine independent runs from September
1, the 18 completed runs in the August 28 four-temperature campaign, the seven
August 27 velocity replicas and the August 27 999 ps run. Exact paths and sizes
are in [collections.csv](collections.csv). The 42 reviewed directories include
25 attempts that already contained only logs/provenance, with no MD payload left.

Plots, metrics, logs, input recipes, potential files and other provenance remain.
The removed payload includes trajectories, native MD restarts and starting
structures **only inside the listed retirement targets**. The 36 binary manifests
were renamed `retired_trajectory_manifest.json` so they cannot advertise deleted
arrays as available trajectories. Each target has `retirement.json` and
`RETIRED.md`; 11 collection IDs remain in the registry as `retired` /
`provenance_only`, preserving historical identity and ancestry.

Accepted trajectories, model checkpoints, predictions and metrics were not
deleted. All 1,800 audited files across the **150 main independent Al sources**
have unchanged inode, length and modification time after cleanup; this is a
filesystem identity check, not a fresh full-array checksum. Successful continuation
controls and the older incomplete 100k Al source were outside this cleanup.
The BCR overfit cache survives, but its retired coarse source can no longer be
used to rebuild it. Historical producer states are retained and do not imply that
the deleted payload is still available.

Before deletion, Slurm jobs and user processes on all 11 currently allocated
nodes were checked: no target was referenced by an open file, working directory
or command. The active Al birth and Ta shooting campaigns were outside the
targets. A full file-stat comparison also preceded deletion. Old failed statuses
and archived interruption records established that these attempts were stopped.

Evidence:

- [Deletion receipt](receipt.json), [enumerated files](planned_files.csv) and
  [executed actions](actions.jsonl).
- [Registry entries before retirement](registry_before.json),
  [Slurm snapshot](active_slurm_before.txt) and
  [allocated-node process audit](active_process_audit.json).
- [Holdings before cleanup](../inventory_20260926/README.md) and
  [holdings after cleanup](../inventory_20260926_after_cleanup/README.md).

## Can the main Al sources be densified from 0.75 ps to 0.1 ps?

Not by recovering frames from their stored observations. All 150 accepted sources
retain a float16 trajectory at 0.75 ps; none retains its original measurement
text dump or intermediate rolling restart files. Position/velocity interpolation
would create synthetic paths, not observed MD dynamics, and must not be used as
ground truth for short-time structure, motion or crystallization prediction.

All 150 still have `prepared_liquid.lammps.data`, `melt_final.restart.bin`,
`final.restart.bin` and `source.in.lammps`; see the
[per-source audit](al_replay_start_audit.csv). The original source input reads the
prepared liquid, creates velocities with its recorded seed, equilibrates for
15 ps and then measures 600 ps. A new dense run can therefore reuse the native
prepared liquid and skip the 300 ps melt. A final restart can extend the future,
but cannot fill gaps in the already completed past.

The integration step is **3 fs**. Fixed output cadence must use whole steps:

| Sampling choice | Integration/output steps | Consequence |
|---|---|---|
| 0.075 ps | Existing 3 fs step, output every 25 steps | 10× denser; every old 0.75 ps observation time remains on the grid |
| 0.099 ps | Existing 3 fs step, output every 33 steps | Close to 0.1 ps, but must be recorded as 0.099 ps |
| Exactly 0.1 ps | For example 2 fs / 50 steps, or 1 fs / 100 steps | Changes the integration protocol; 1.5× or 3× as many integration steps per ps |

Recommended for a matched cadence study: **0.075 ps**, with sampled velocities,
the original 3 fs integration, potential and thermodynamic protocol. This already
matches the cadence of the three completed predictive-memory precision sources.
It raises canonical position/velocity observation storage by approximately 10×;
native preparations and restarts do not scale by that factor.

Reusing the input and seeds does not establish identical dynamics. Exact replay
depends on the executable, numerical settings and parallel execution, and small
roundoff changes can grow. LAMMPS documents the requirements and limitations in
[read_restart](https://docs.lammps.org/read_restart.html). Validate shared-time
observations before calling a rerun a reproduction; otherwise register it as a new
trajectory sharing the original melt ancestry. Preserve the fixed Al64 source
split and benchmark identity; do not overwrite old data or treat daughter runs
as independent lineages. No denser Al run was launched by this cleanup.
