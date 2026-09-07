# Al MEAM simulation context for a new agent

**Recovery update, September 5:** read
[simulation_recovery_20260905.md](simulation_recovery_20260905.md) first.
Low-temperature array 981561 and controller 981562 replace the failed chain;
source 28 is prepared for its normal scheduled retry. Exact restart extensions
remain blocked after a completed boundary diagnostic.

Latest live audit: **2026-09-05 14:44 CEST** — see
[simulation_audit_20260905.md](simulation_audit_20260905.md) for current counts,
the source-controller failure caused by launcher relocation, and recovery.
The detailed September 4 snapshot below is historical; its queue and completion
counts must not be used as live status.

Historical handoff refresh: **2026-09-04 13:54 CEST**

This document is the operational handoff for the Al MEAM source and shooting
simulations. Read it together with the repository `AGENTS.md`. The authoritative
scientific specification is
`docs/new_shooting_campaign_request_20260903.md`; this document explains what
already exists, what is running, how the products differ, and what the next
simulation agent must do.

## Scope and safety rules

- Repository: `/home/infres/vmorozov/PointCloudMaterials`
- Environment: `pointnet`
- Large-data storage: `/home/ids/vmorozov/simulations`
- Do not copy trajectories to the quota-limited `infres` repository filesystem.
- Use CPU Slurm nodes. Do not modify, stop, or consume resources belonging to
  VAMP, predictive-atlas training, interactive accelerator allocations, or any
  other GPU work.
- The normal QOS permits ten submitted jobs, counting array elements and
  controllers. Always inspect the expanded queue with `squeue -r` before a new
  submission.
- Submit CPU work from a login node, not from inside a GPU allocation. Inherited
  `SLURM_*GPU*` variables caused the superseded smoke job `979929` to fail before
  dynamics began.
- Slurm accounting may be unavailable. A job leaving `squeue` is not proof of
  success; filesystem outcomes are authoritative.
- Never overwrite a completed branch. Preserve partial dumps, logs, and status
  files in a clearly named `interrupted_attempt_*` directory before retrying.
- Do not delete audit archives. Do not use partial trajectories as training
  data.

Start every shell session with:

```bash
cd /home/infres/vmorozov/PointCloudMaterials
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
export PYTHONPATH=/home/infres/vmorozov/PointCloudMaterials
```

## Scientific purpose

The old dataset contains useful trajectories but too few independent parent
structures and is dominated by crystallization. The new work targets a
transition-balanced conditional future law using structurally diverse parent
states, independent source lineages, controls, and five temperatures:

`400, 450, 500, 510, and 520 K`.

For each temperature, the production target is:

| Parent role | Optimization | Model selection | Final validation | Total |
|---|---:|---:|---:|---:|
| Transition | 10 | 2 | 4 | 16 |
| Liquid control | 4 | 2 | 2 | 8 |
| Crystal control | 4 | 2 | 2 | 8 |
| Total | 18 | 6 | 8 | 32 |

This is 160 new parents. Each gets 16 futures in an 8 momentum x 2 thermostat
design, or 2,560 production branches. A separate top-up adds four futures to
each of 40 old parents, or 160 branches. Total mandatory new shooting work is
2,720 branches.

The split unit is the **root source lineage**, not an atom, frame, shifted
window, parent, or thermostat replica. Every descendant and every temperature
version of a paired position must inherit the root split. Final-validation
parents must not be chosen from their future shooting outcomes.

## The current 15 ps branch contract

Each new shooting branch must have:

- 70,304 atoms;
- a 3 fs integration timestep;
- fixed 15 ps dynamics, with no basin-triggered early stopping;
- 51 frames at steps `0, 100, ..., 5000`, or every 0.3 ps;
- float32 positions and velocities;
- explicit atom IDs/types and a periodic box for every frame;
- first-passage/PTM progress measured through 15 ps;
- a nonempty, checksummed 15 ps restart;
- `outcome.json` with `"state": "complete"` after strict validation.

The first 12 ps are the main prediction horizon; the final 3 ps permit shifted
window starts. Source histories are stored in float16 to control size, but the
authoritative new shooting contract is float32. Do not silently change either
dtype.

New branches use LAMMPS `fix temp/csld`. Its stochastic state can be written to
the restart. Both the short run and continuation are locked to 24 MPI ranks.
Production remains blocked until a 24 ps uninterrupted trajectory matches the
corresponding 15+9 ps restarted trajectory in the continuation smoke test.
Older `fix langevin` branches are valid uninterrupted paths but cannot satisfy
this exact stochastic-continuation contract.

## Existing datasets and their meaning

### Fixed-horizon 48 ps atlas data

The following immutable roots contain 480 complete paths from 40 parents, 12
futures per parent:

- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830`
  — 320/320 strict-complete.
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831`
  — 40/40 complete.
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831`
  — 80/80 complete.
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901`
  — 40/40 complete.

Use this merged lineage/identity snapshot instead of inventing identities from
legacy local shot indices:

`/home/ids/vmorozov/experiments/predictive_atlas_geoframe_v2_480branches_20260902/dataset_snapshot.json`

It describes 12 parents at 400 K, 12 at 450 K, 16 at 500 K, 20 independent
source histories, 28 training parents, and 12 held-out parents. It is enough for
pipeline development and preliminary results, not the main independent-parent
claim.

### Nested first-passage pilot

- Variable-stop original:
  `/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902`
- Fixed-24 ps compatible reconstruction:
  `/home/ids/vmorozov/simulations/al_meam_nested_shooting_pilot_70304_400-500K_20260902_fixed24ps_float16_compatible`

Both have 144/144 complete outcomes. The original stops on basin arrival and is
an auxiliary committor/survival dataset; do not pretend it supplies absent
6/12/24 ps frames. Use the separately validated compatible root when fixed
horizons are required. Its additional frames are real LAMMPS continuations,
not fabricated copies, but the old Langevin noise stream could not be restored;
the continuation therefore uses a new deterministic, provenance-recorded noise
seed after the saved first-passage restart.

### Historical fixed-48 ps follow-up

- Smoke:
  `/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed48_smoke_1parent_16branches_float32_20260903`
  — 16/16 complete.
- Partial top-up:
  `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_4shot_topup_to16_20260903`
  — 40/160 complete.

These are immutable historical products superseded by the 15 ps request. Do not
resume, delete, merge blindly, or reinterpret the partial fixed-48 ps top-up as
the new campaign.

## Active and prepared 15 ps work

### Smoke test

Root:

`/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904`

All 16/16 mandatory smoke branches and the strict campaign summary are complete.
The measured paths total 1.39 GB and the restarts total 99.3 MB. Branch 0 used
264.52 seconds of dynamics; the sequential job averaged about 326 seconds per
additional complete branch including conversion and validation.

Exact restart continuation is blocked. The short path and an independently
rerun comparator are bitwise identical through timestep 5000, their midpoint
restart files are byte-identical, and both use a 2x3x4 MPI grid. Nevertheless,
`read_restart` diverges at the first stored continuation frame (timestep 5100).
The current evidence is in `continuation_smoke_test_failed.json`; failed test
artifacts are preserved in per-branch `interrupted_attempt_continuation_smoke_*`
archives. Mandatory 15 ps trajectories remain valid, but no result from this
protocol may be advertised as an exact on-demand continuation.

### Restart-suitability audit still required

Do not assume the divergent restarted segment is either useful or useless. A
future agent must test whether such segments are statistically suitable as a
separate stochastic-future dataset conditioned on the saved 15 ps phase-space
state. The authoritative procedure is in Part E of
`docs/new_shooting_campaign_request_20260903.md`.

In brief, first verify exact boundary positions, velocities, IDs, cell, fix
parameters, checksum, rank count, and processor grid. Then compare paired
uninterrupted, restarted, and fresh-thermostat control ensembles across
temperatures and parent roles using thermodynamics, minimum-image displacement
and velocity statistics, PTM progress, basin outcomes, first-passage times, and
the path-kernel/RFF representation. Predeclare equivalence margins. Keep results
separate and label them `restart_divergent_stochastic_future` only if that audit
passes. Exact extensions remain blocked unless the strict trajectory test itself
passes.

### Old-parent 15 ps top-up

Root:

`/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904`

It has an immutable 160-branch manifest and currently has 0/160 outcomes. Its
four new seed pairs per old parent are disjoint from the existing 480 paths and
use merged shot indices 12--15. At the user's explicit request, mandatory short
paths were submitted despite the separately recorded exact-extension failure.
`extension_gate.json` allows only fixed-15 ps paths and forbids exact-extension
claims.

Initial controller `980083` is dependency-pending on `afterany:979717`, the
current L40S allocation. This occupies the tenth QOS slot without consuming CPU.
When the allocation ends, the controller starts at branch 0 and maintains a
one-branch array plus an `afterany` successor. Each branch requests 24 MPI ranks,
24 GB, and one hour; the measured runtime is about 5.4 minutes.

### New-parent production

Reserved root, not yet created:

`/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_160parents_2560branches_float32_20260904`

Do not create or submit this campaign until all five-temperature sources are
complete, the 510/520 K basin and transition interfaces are calibrated, parent
selection is complete without validation leakage, both smoke tests pass, and
the IDS quota has been increased.

## Independent-source prerequisites

### 400/450/500 K

Root:

`/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`

Target: 90 independent histories, 30 per temperature. At the live refresh,
38/90 had complete outcomes. Array `979918` tasks 38--40 were running with a
three-task concurrency limit; controller `979919` had an `afterany` dependency
and should submit the next wave.

### 510/520 K

Root:

`/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`

Target: 60 independent histories, 30 per temperature. At the live refresh,
0/60 were complete. Array `979924` tasks 0--1 was pending with a two-task
concurrency limit; controller `979925` was dependency-pending. The manifest's
1--99 atom candidate interval is deliberately broad and diagnostic. Calibrate
temperature-specific basin-A and transition ranges from completed 510 and 520 K
histories before selecting parents.

In addition, source task 59 is running detached outside Slurm on `lamedell11`.
It is the pristine `source_059_T520_final_validation` lineage at the far end of
the campaign, uses all 48 physical cores through 48 local MPI ranks, and records
`execution_mode: local_mpiexec`. Driver PID at launch was `3095474`; its log is:

`/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903/local_task_0059_lamedell11.driver.log`

Its per-run `status.json` and eventual `outcome.json` remain authoritative. The
local runner is deliberately strict: an existing partial/running status makes a
later duplicate task fail instead of overwriting it.

The source jobs each request one CPU node, 48 MPI ranks, 24 GB, and six hours.
The source controllers request only a short CPU job. Let each `afterany`
controller continue even after an isolated task failure, but verify that it
actually submits a successor.

## Live Slurm snapshot

At 2026-09-04 13:54 CEST, the expanded queue contained all ten allowed jobs:

- running CPU: `979918_38`, `979918_39`, `979918_40`;
- pending CPU controllers/tasks: `979919`, `979924_0`, `979924_1`, `979925`,
  and top-up controller `980083`;
- user accelerator allocations: H100 `979715` pending and L40S `979717`
  running.

The accelerator allocations are out of scope. With ten submitted entries, do
not submit another job until the expanded count drops. This snapshot will age;
always refresh it rather than assuming these IDs remain current.

```bash
squeue -r -u "$USER" \
  -o '%.22i %.10P %.36j %.2t %.10M %.10l %.4D %R'
```

For a campaign, read `slurm/active_submission.json` and the tail of
`slurm/submission_chain.jsonl`. Do not assume the first array/controller IDs are
still current.

## How to decide whether data are usable

For shooting data, a trajectory directory, final restart, or Slurm `COMPLETED`
state is insufficient. A branch is usable only if its canonical `outcome.json`
has `state: complete` and the repository validator accepts every artifact.
Strict 15 ps campaign summarization also proves the exact frame/timestep/atom
contract and writes the indexes.

For source histories, use only run directories whose `outcome.json` reports
`state: complete`. A source outcome also references the float16 trajectory,
crystallization progress, thermodynamics, and final restarts.

Useful read-only counts:

```bash
SMOKE=/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904
SRC_LOW=/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902
SRC_HIGH=/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903

find "$SMOKE/branches" -mindepth 2 -maxdepth 2 -name outcome.json | wc -l
find "$SRC_LOW/runs" -mindepth 2 -maxdepth 2 -name outcome.json | wc -l
find "$SRC_HIGH/runs" -mindepth 2 -maxdepth 2 -name outcome.json | wc -l
```

Counting files is only a progress hint. Inspect their state and run the strict
summarizer at completion. The 15 ps command is:

```bash
python scripts/run_lammps_campaign.py predictive-dynamics-15ps summarize \
  --campaign-root "$SMOKE"
```

It intentionally fails loudly when any branch or invariant is missing. On
success, require both `summary.json` and `status.json` to say `state: complete`,
and inspect `short_15ps_branches.json`.

## Binary layout and loading

A complete 15 ps branch contains:

```text
branches/<branch_id>/
├── outcome.json
├── metadata.json
├── first_passage_progress.npz
├── final.restart.bin
└── trajectory_binary_float32/
    ├── manifest.json
    ├── positions.npy
    ├── velocities.npy
    ├── timesteps.npy
    ├── atom_ids.npy
    ├── atom_types.npy
    ├── box_low.npy
    └── box_high.npy
```

Load and checksum a path with the repository-owned reader:

```python
from src.data_utils.shooting_binary import ShootingBinaryTrajectory

trajectory = ShootingBinaryTrajectory.load(binary_directory)
trajectory.verify_checksums()
positions = trajectory.positions  # memory-mapped array
velocities = trajectory.velocities
timesteps = trajectory.timesteps
```

Use `src/data_utils/shooting_dataset.py` for campaign validation and snapshot
loading. Do not bypass the manifests by globbing `.npy` files into a training
set.

At strict completion the new runner publishes separate indexes:

- `short_15ps_branches.json` for all mandatory paths;
- `conditional_law_parents.json` for parents with 16 validated futures;
- `overlapping_temporal_windows.json` for correlated shifted windows;
- `extended_24ps_branches.json` for selected exact continuations;
- `extended_event_branches.json` for optional later extensions.

An extension never changes the short branch's outcome or completion state.

## Failure and recovery procedure

1. Refresh `squeue -r`, `active_submission.json`, the chain tail, and filesystem
   outcomes together.
2. If an array/controller is still active, do not submit a duplicate.
3. If a controller has `DependencyNeverSatisfied`, is gone without a successor,
   or an array task failed/timed out, identify the first incomplete manifest
   index from outcomes—not from directory presence.
4. A stale `status.json` with `state: running` is not complete. Confirm that no
   matching Slurm task is active.
5. Archive partial branch artifacts before retrying. For a 15 ps shooting
   branch, use:

```bash
python scripts/run_lammps_campaign.py predictive-dynamics-15ps archive-partial \
  --campaign-root /absolute/campaign/root \
  --branch-index FIRST_INCOMPLETE_INDEX \
  --label wave_JOBID_REASON_YYYYMMDDTHHMMSSZ
```

6. Restart from the first incomplete index with the campaign's generated Slurm
   scripts/runner. Preserve the configured source concurrency (3 for 400--500 K,
   2 for 510/520 K) and never overwrite branches already accepted as complete.
7. After a wave, verify that the `afterany` controller submitted its successor.
   If not, report the exact queue/filesystem evidence before intervening.

The generated runners bind immutable manifests with `manifest.sha256`, reject
duplicate active submissions, and fail loudly on partial artifacts. Do not edit
a manifest after branches have started.

## Storage and resource envelope

Measured/planning values for mandatory 15 ps branches are approximately:

- 24 MPI ranks, one CPU node, 24 GB, one-hour per-branch Slurm limit;
- 0.15 node-hours per completed branch;
- 86.90 MB float32 trajectory plus 6.19 MB restart;
- about 270 MB transient text while a branch is active.

The 2,720 mandatory new branches project to about 408 node-hours and 253.21 GB
of final paths/restarts. The 60 high-temperature source histories add roughly
180 node-hours and 41--43 GB. On 2026-09-04, `/home/ids/vmorozov` used about
294 GB of a 512 GB quota. The minimum projected total was near 620 GB before
selected extensions and safety headroom. Obtain at least 750 GB total quota;
1 TB is safer before production.

## Next actions, in order

1. Monitor top-up controller `980083`, then its `active_submission.json`,
   dependency-chained waves, and filesystem outcomes. Do not add a Slurm job
   while the expanded QOS count is ten.
2. Keep exact 15-to-24 ps extension use blocked unless a new protocol passes a
   strict continuation test. Run the documented restart-suitability audit before
   deciding whether divergent restarted segments are usable as a separately
   labelled stochastic-future dataset; do not reinterpret the failed comparator.
3. Continue monitoring both source chains and local high-temperature source
   task 59 on `lamedell11`.
4. Let all 90 low-temperature and 60 high-temperature independent sources
   complete and strictly summarize them.
5. Calibrate 510 and 520 K basin/transition thresholds using completed source
   histories.
6. Recheck IDS usage/quota and report measured smoke runtime/storage.
7. Strictly summarize the 160-branch top-up after all outcomes complete.
8. Select 160 structurally diverse production parents with the exact role,
   split, temperature, history, and paired-temperature contract.
9. Build and checksum the production manifest, report exact counts/resources,
   then submit recoverable CPU waves.

Never skip the continuation gate, source-lineage split, high-temperature
calibration, or quota check merely because the shooting runner is ready.

## Important repository files

- Scientific request: `docs/new_shooting_campaign_request_20260903.md`
- Short operational overview:
  `docs/predictive_dynamics_simulation_campaign_20260903.md`
- 15 ps configuration:
  `configs/simulation/atomistic/al/meam_predictive_dynamics_fixed15_20260904.yaml`
- 15 ps runner:
  `scripts/run_lammps_campaign.py predictive-dynamics-15ps`
- General fixed-horizon runner:
  `scripts/run_lammps_campaign.py predictive-dynamics`
- Independent-source runner:
  `scripts/run_lammps_campaign.py independent-meam-source`
- 510/520 K source specialization:
  `experiments/independent_sources_20260903/independent_meam_510_520K_sources.py`
- Binary reader/converter: `src/data_utils/shooting_binary.py`
- Strict dataset validators/loaders: `src/data_utils/shooting_dataset.py`
