# Predictive-dynamics simulation campaign (revised 2026-09-04)

The detailed operational handoff for a new simulation agent is
`docs/simulation_context_for_agents.md`.

The active request is `docs/new_shooting_campaign_request_20260903.md`. Its
parameter-locked configuration and runner are:

- `configs/simulation/atomistic/al/meam_predictive_dynamics_fixed15_20260904.yaml`
- `scripts/run_lammps_campaign.py predictive-dynamics-15ps`

The older `fixed48_20260903` configuration, runner, smoke, and partial top-up
remain immutable historical products. They must not be overwritten or silently
reinterpreted as the new 15 ps campaign.

## Canonical roots

- Prepared 15 ps smoke, one parent and 16 branches:
  `/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_smoke_1parent_16branches_float32_20260904`
- Prepared 15 ps old-parent top-up, 40 parents and 160 branches:
  `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904`
- Reserved production root, not created before source selection:
  `/home/ids/vmorozov/simulations/al_meam_predictive_dynamics_fixed15_160parents_2560branches_float32_20260904`
- Running 400/450/500 K independent-source prerequisite:
  `/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_400-500K_30perT_float16_20260902`
- Submitted 510/520 K independent-source prerequisite:
  `/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`

Every root has an immutable `manifest.json` bound by `manifest.sha256`. Only a
canonical `outcome.json` with `state: complete` is training data. Partial text
dumps, binary build directories, and stale running statuses are not data.

## Mandatory branch contract

Each new branch has 70,304 atoms, a 3 fs integration step, fixed 15 ps duration,
and output every 100 steps. Its binary contains exactly 51 frames at timesteps
0, 100, ..., 5000. The first 12 ps are the principal prediction interval and the
last 3 ps permit shifted starts. Positions and velocities are float32; IDs,
types, per-frame boxes, initial velocity fields, PTM progress, first-passage
records, and restarts have checksums.

The new protocol uses fixed-cell stochastic canonical dynamics with LAMMPS
`fix temp/csld`. Unlike `fix langevin`, this fix writes its RNG state to the
restart. Exact continuation is still accepted only after a 24-rank uninterrupted
24 ps versus 15+9 ps smoke test passes. Continuation always uses the original
fix ID, thermostat seed and parameters, and the same 24 MPI ranks.

That exact test failed on 2026-09-04 even though the short and uninterrupted
paths match bitwise through timestep 5000, the midpoint restart files match
byte-for-byte, and the runs use the same 2 x 3 x 4 processor grid. Divergence
starts at the first stored post-restart frame, timestep 5100. Therefore the
mandatory 0--15 ps paths are valid, but restarted segments are not exact
continuations and must not be merged into the exact-extension index.

The restarted segments may still be informative as separately labelled
stochastic futures. Before using them, perform the paired distributional
restart-suitability audit specified in Part E of
`docs/new_shooting_campaign_request_20260903.md`: verify the boundary state,
compare uninterrupted/restarted/fresh-thermostat ensembles across temperatures
and parent roles, predeclare equivalence margins, and check thermodynamics, PTM
progress, basin outcomes, first-passage times, and path-kernel/RFF features.
Passing that audit permits only an auxiliary
`restart_divergent_stochastic_future` classification; exact continuation remains
blocked until the trajectory test passes.

The text dump is transient and is deleted only after binary conversion and
checksum verification. The immutable 15 ps trajectory and outcome never depend
on whether an extension is requested.

## Resource estimate

The planning estimate extrapolates measured 24-rank 48 ps jobs to 15 ps and 51
frames:

- about 0.15 node-hours per mandatory branch;
- about 86.90 MB float32 trajectory plus 6.19 MB restart per branch;
- about 270 MB transient text per active branch.

| Work | Branches | Node-hours | Float32 paths | Restarts | Final minimum |
|---|---:|---:|---:|---:|---:|
| smoke | 16 | 2.4 | 1.39 GB | 0.10 GB | 1.49 GB |
| old-parent top-up | 160 | 24.0 | 13.90 GB | 0.99 GB | 14.89 GB |
| new parents | 2,560 | 384.0 | 222.48 GB | 15.84 GB | 238.32 GB |
| requested mandatory work | 2,720 | 408.0 | 236.38 GB | 16.83 GB | 253.21 GB |

These figures exclude 160 parent histories, independently generated sources,
PTM and thermodynamic arrays, selected extensions, indexes, logs, and safety
headroom. The existing 512 GB quota is insufficient for the current holdings
plus full production. The home currently uses 294 GB; the minimum projection
after remaining sources, smoke/top-up, and production is about 620 GB before
selected extensions and headroom. Request at least 750 GB total quota; 1 TB is
safer. Do not create or submit the 2,560-branch production root first.

The 60 independent 510/520 K source histories add about 180 node-hours and
roughly 41--43 GB. Their broad 1--99 atom interval is diagnostic only. Basin-A
and transition bands for both temperatures must be calibrated from completed
metastable-liquid histories before parent selection.

## Required order

1. Complete the five-temperature source campaigns.
2. Calibrate 510 and 520 K basin-A thresholds and transition interfaces.
3. Complete and strictly summarize all 16 mandatory 15 ps smoke branches.
4. Preserve the failed 15-to-24 ps exact-continuation evidence and run the
   documented restart-suitability audit before using any restarted segment.
5. Report observed smoke runtime/storage and recheck IDS quota.
6. Run the 160-branch old-parent top-up.
7. Select production parents using only present and past information, build the
   immutable 160-parent/2,560-branch manifest, and submit recoverable waves.

Strict campaign completion publishes `short_15ps_branches.json`,
`conditional_law_parents.json`, and `overlapping_temporal_windows.json`.
Extensions publish separately to `extended_24ps_branches.json`; they never alter
mandatory short-branch completion.

## Operational snapshot: 2026-09-04 12:00 CEST

- The corrected mandatory smoke branch 0 is strict-complete: 51 frames,
  timestep 5000, 86.90 MB float32 path, verified restart, and deleted transient
  text dump. Its measured dynamics time was 264.52 seconds.
- Slurm job `979937` is running smoke branches 1--15 sequentially in one
  24-rank CPU allocation and will run strict summarization afterward.
- The 510/520 K source wave is submitted as array `979924` with controller
  `979925`; both tasks are pending CPU scheduling. This campaign contains 30
  independent source histories at each temperature.
- The 400/450/500 K source chain continues independently. Neither source
  campaign consumes or modifies accelerator allocations.
- The 160-branch top-up is prepared but deliberately unsubmitted until the
  complete smoke and exact continuation test pass.
