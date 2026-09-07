# Request for the next Al MEAM predictive-dynamics simulation campaign

Last revised: 2026-09-04

This is a copy-ready handoff for the agent responsible for simulations.

## Objective

Generate data for two related tasks:

1. Learning the conditional distribution of future structural paths from a
   parent structure, history, and temperature.
2. Auxiliary temporal and shot-conditioned training from dense, overlapping
   windows along individual trajectories.

Overlapping windows are intentionally retained. They are correlated rather than
independent, but they expose the model to many nearby encodings and partially
shared futures, improving temporal smoothness and robustness. They must carry
lineage/overlap metadata, remain in one split, and be weighted as correlated
examples during training.

The primary maximum prediction horizon is 12 ps, not 24 ps. Every new branch has
a mandatory 15 ps short trajectory: 12 ps for prediction plus a 3 ps buffer for
moving the window start. Selected short branches must be exactly extendable to
24 ps later without rerunning their first 15 ps or resampling their dynamics.

## Simulation-agent request

Repository:

`/home/infres/vmorozov/PointCloudMaterials`

Conda environment:

`pointnet`

Authoritative simulation storage:

`/home/ids/vmorozov/simulations`

Resolved implementation (2026-09-04):

- Configuration:
  `configs/simulation/atomistic/al/meam_predictive_dynamics_fixed15_20260904.yaml`
- Runner:
  `scripts/run_lammps_campaign.py predictive-dynamics-15ps`
- New mandatory branches use LAMMPS `fix temp/csld`, not `fix langevin`.
  LAMMPS does not save the `fix langevin` random-number state, so it cannot
  satisfy this request's exact stochastic continuation contract. `temp/csld`
  saves its RNG state, provided continuation uses the same MPI rank count.
- Mandatory and continuation runs are locked to 24 MPI ranks. Production is
  blocked until the uninterrupted-versus-restarted smoke test passes.
- The 510/520 K independent-source campaign is a separate required prerequisite:
  `/home/ids/vmorozov/simulations/al_meam_independent_sources_70304_510-520K_30perT_float16_20260903`.

The LAMMPS restart semantics are documented at
<https://docs.lammps.org/fix_langevin.html> and
<https://docs.lammps.org/fix_temp_csvr.html>.

Inspect and reuse the existing simulation infrastructure before changing it:

- `scripts/run_lammps_campaign.py meam-nested-shooting`
- `scripts/run_lammps_campaign.py meam-shooting`
- `scripts/run_lammps_campaign.py meam-shooting-followup`
- `scripts/run_lammps_campaign.py nested-fixed-horizon-compatibility`
- `configs/simulation/atomistic/al/meam_nested_shooting_pilot_70304_20260902.yaml`

Do not modify, overwrite, or delete completed campaigns. New results must live in
new canonical IDS roots. Do not copy large trajectories back to the infres
filesystem.

### Part A: top up the existing 40-parent ensemble from 12 to 16 futures

The following completed roots jointly form the existing 40-parent, 480-branch
dataset:

- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_8shots_20260830`
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_1shot_local_20260831`
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_2shots_local_followup_20260831`
- `/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_48ps_40branches_local_20260901`

The authoritative merged snapshot is:

`/home/ids/vmorozov/experiments/predictive_atlas_geoframe_v2_480branches_20260902/dataset_snapshot.json`

It contains:

- 40 immutable parent configurations;
- 12 complete unique-seed futures per parent;
- 12 parents at 400 K, 12 at 450 K, and 16 at 500 K;
- 28 source-training and 12 source-validation parents;
- 20 independent source MD runs;
- 20 parents selected 12 ps before nucleation and 20 selected 3 ps before
  nucleation.

Create a separate follow-up campaign containing exactly four new branches for
each of these 40 parents: 160 new branches total. Reuse the exact parent
positions, atom IDs, cell, temperature, phase label, root source run, and split
from the merged snapshot.

Requirements:

- Every new velocity-seed/thermostat-seed pair must be absent from all existing
  480 branches.
- Assign canonical merged shot indices 12, 13, 14, and 15. Legacy per-campaign
  local shot indices are not globally unique and must not be used as identity.
- Preserve original source lineage and train/validation split.
- Simulate every new branch for a fixed 15 ps, even if a basin is reached earlier.
- After the top-up, every old parent must have 16 futures at horizons up to
  12 ps. The original 12 branches per parent remain available for separate
  24/48 ps analyses; the four top-up branches do not need to reproduce those
  longer horizons.
- Retain an exact continuation-capable restart at 15 ps for every new branch.
- Produce a strict merged index proving exactly 16 unique futures for every old
  parent after the top-up.
- Do not change or reinterpret the existing 480 outcomes.

Suggested new root:

`/home/ids/vmorozov/simulations/al_meam_position_shooting_70304_400-500K_15ps_4shot_topup_to16_20260904`

### Part B: new multiphase parents at five temperatures

Generate parents at:

`400, 450, 500, 510, and 520 K`

At each temperature target:

| Parent role | Optimization | Model selection | Final validation | Total |
|---|---:|---:|---:|---:|
| Transition candidate | 10 | 2 | 4 | 16 |
| Stable liquid control | 4 | 2 | 2 | 8 |
| Stable crystal control | 4 | 2 | 2 | 8 |
| Total | 18 | 6 | 8 | 32 |

The full target is therefore 160 new parents and 2,560 branches at 16 branches
per parent. Together with the old-parent top-up, the requested campaign work is
2,720 new branches. Before the main submission, report the node-hour and storage
estimate. Execution may be staged in recoverable waves, but do not silently
change the final design or counts.

All 2,720 new branches require 15 ps of mandatory integration. This is 40,800
branch-ps, compared with 130,560 branch-ps in the previous 48 ps design:
approximately 69% less main-branch integration. It is also 37.5% less than a
mandatory 24 ps design. Parent-generation trunks and selected on-demand
extensions are additional and must be reported separately in the resource
estimate.

Using measured 24-rank 48 ps jobs and scaling their integration and output to
51 frames, the planning estimate is 0.15 node-hours and 93.09 MB final storage
per mandatory 15 ps branch. The 2,720 mandatory branches therefore require
approximately 408 node-hours and 253.21 GB for float32 paths plus final
restarts. Parent histories, source campaigns, progress arrays, logs, selected
extensions, and safety headroom are additional. The 512 GB IDS quota is not
enough for all current data plus this full production campaign; do not submit
the 2,560-branch production root before checking available quota and obtaining
the required increase. As measured on 2026-09-04, `/home/ids/vmorozov` already
uses 294 GB. Completed mandatory branches plus the remaining source histories
would put the minimum projection near 620 GB before selected extensions and
headroom. Request at least 750 GB total quota; 1 TB is safer.

If suitable independent 510 or 520 K source trajectories do not already exist,
generate them first. Use multiple independent source velocity seeds and record
them exactly as for the existing 400--500 K data.

Transition parents must cover different present structures and recent dynamics:

- different largest-cluster sizes and crystalline fractions;
- positive, near-zero, and negative recent cluster-growth rates;
- interface-like and bulk-like local environments;
- likely growth, dissolution, and ambiguous states.

Select parents using only present and past information. Do not select final-
validation parents using their future shooting outcomes.

Liquid and crystal controls must be equilibrated states with valid histories,
not unrelaxed frame-zero snapshots. Their actual outcomes must still be measured;
do not assume from the parent label that a branch remains in its starting basin.

### Part C: recover history by propagating before selecting the parent

Every parent used by a history model must have at least 12 ps of valid structural
prehistory.

If a useful state has no pre-parent history:

1. Continue it forward under the intended temperature and phase conditions for
   at least 12 ps.
2. Move the parent time to 12 ps or later.
3. Preserve the full preceding trajectory as parent history.
4. Recompute the phase/role at the new parent time.
5. Store history every 0.3 ps and ensure exact frames at -12, -9, -6, -3, and
   0 ps relative to the parent.

The history-generating continuation is an ancestor trajectory, not one of the 16
independent child futures.

### Part D: use shifted and overlapping starts as auxiliary data

Simulate every new shooting branch for 15 ps with uniform output. The first 12 ps
form the principal future interval and the final 3 ps are an augmentation buffer.
Index multiple overlapping temporal windows along every complete trajectory,
including the old 48 ps branches, the existing 24 ps nested branches, and the new
15 ps branches.

The primary conditional-law horizons are 3, 6, and 12 ps. Retain 1.2 and 9 ps as
auxiliary temporal targets. Use a configurable start stride, initially 0.6 ps;
1.2 ps must remain an inexpensive loader-side ablation. With a 15 ps trajectory:

- complete 12 ps windows can start between 0 and 3 ps;
- complete 9 ps windows can start between 0 and 6 ps;
- complete 6 ps windows can start between 0 and 9 ps;
- complete 3 ps windows can start between 0 and 12 ps;
- shorter 1.2 ps windows can use every start through 13.8 ps.

Retain every window satisfying `start_time + horizon <= 15 ps`. The existing
24/48 ps trajectories can supply still more shifted windows with complete 12 ps
futures.

Each window must record:

- root source-run ID;
- ancestor parent and branch ID;
- absolute and relative start time;
- available prehistory interval;
- available future interval;
- IDs of windows with overlapping future frames;
- temperature, phase descriptors, and split;
- whether it has one realized future or a separately generated 16-shot law.

These windows are required training data. They provide nearby variants of the
present encoding and overlapping future paths, which can be used for:

- temporal encoder pretraining;
- smoothness and consistency losses;
- augmentation of branch-level future prediction;
- shot-conditioned prediction using actual initial velocities;
- robustness to small changes in the parent time.

They must not be counted as independent source runs or independent samples of a
conditional law. All windows and descendants from one root lineage must remain
in the same split.

For a selected shifted state to become a new conditional-law parent, launch a
new 16-branch ensemble from that exact state. Do not automatically branch from
every window. Select a manageable, structurally diverse subset using present and
past descriptors, then create 16 futures for each selected state.

Produce two separate indexes:

1. `conditional_law_parents.json`: only parents with 16 validated futures.
2. `overlapping_temporal_windows.json`: all lineage-labelled single-trajectory
   windows, including their overlap groups.

### Part E: exact on-demand extension of short trajectories

The mandatory campaign product ends at 15 ps. A branch is complete when its 15 ps
trajectory passes validation; it must not wait for or require a long extension.

Implement or adapt a separate continuation command that can extend any selected
complete branch from 15 to 24 ps. The continuation must:

- start from the saved 15 ps restart;
- preserve positions, velocities, integrator/fix state, thermostat parameters,
  and the stochastic continuation state required by LAMMPS;
- never resample momenta or restart the branch as a new shot;
- output frames from 15.3 through 24.0 ps at the same 0.3 ps interval;
- avoid duplicating the existing 15.0 ps frame;
- preserve the immutable short trajectory and short outcome;
- write a separate extension outcome and provenance record;
- produce a validated composed 0--24 ps view with 81 frames;
- record why the branch was selected for extension.

For the new 15 ps campaign, use `fix temp/csld` with the same fix ID, parameters,
seed, and 24 MPI ranks on both sides of the restart. Do not claim exact extension
for existing `fix langevin` branches: their thermostat RNG state is absent from
the restart. Those older branches remain valid uninterrupted 24/48 ps reference
trajectories, but they are not inputs to this continuation contract.

Run an uninterrupted 24 ps versus 15+9 ps continuation smoke test and verify that
the continuation is physically and numerically consistent under the actual
LAMMPS restart semantics. If bitwise identity is not expected, document the exact
reason and a quantitative tolerance before launching the campaign.

#### Required restart-suitability audit

The 2026-09-04 smoke test did not establish exact continuation. The independently
rerun short and uninterrupted paths are bitwise identical through timestep 5000,
the short-run and uninterrupted-run timestep-5000 restart files are
byte-identical, and both runs use the same 24 MPI ranks and 2 x 3 x 4 processor
grid. Nevertheless, the trajectory loaded with `read_restart` differs from the
uninterrupted trajectory at the first saved continuation frame, timestep 5100.
The evidence is recorded in the smoke root's
`continuation_smoke_test_failed.json` and preserved
`interrupted_attempt_continuation_smoke_*` directories. The old-parent top-up
root's `extension_gate.json` records the resulting prohibition on exact
extensions.

This failure has two distinct consequences:

- the validated 0--15 ps trajectories remain valid mandatory data;
- restarted segments must not be described or indexed as the exact continuation
  of the realized short trajectory.

Do not discard the restarted segments without evaluating them. They may still
be useful as separately labelled stochastic futures conditioned on the saved
15 ps phase-space state. Before using them for training or evaluation, perform
and record a restart-suitability audit:

1. Verify the boundary state atom-by-atom: IDs and ordering, types, positions,
   velocities, periodic cell, timestep, fix parameters, restart checksum, MPI
   rank count, and processor grid. Any boundary mismatch is a hard failure.
2. Use a paired ensemble spanning every available temperature and parent role,
   rather than judging suitability from one trajectory. Compare uninterrupted
   futures, restarted futures, and independent thermostat-future controls from
   the same saved phase-space states.
3. Compare thermodynamic traces, minimum-image displacement and velocity
   distributions, PTM crystalline fraction, largest crystalline cluster,
   basin-arrival outcome, first-passage time, and the downstream path-kernel/RFF
   representation. Coordinate equality alone is neither sufficient for
   distributional acceptance nor optional for an exact-continuation claim.
4. Measure restart-versus-uninterrupted differences relative to the natural
   fresh-thermostat-future variation at fixed phase-space state. Define the
   equivalence margins and sample size before inspecting the final comparison.
5. Classify the result explicitly: `exact_continuation` only after the strict
   trajectory test passes; `restart_divergent_stochastic_future` only after the
   distributional audit passes; otherwise `rejected_restart_extension`.

Keep any accepted restart-divergent futures in a separate index with their
restart provenance and classification. Never merge them silently into
`extended_24ps_branches.json`, use them to manufacture a fixed-horizon label, or
let their availability alter the immutable 15 ps outcome. Passing the
distributional audit can authorize an auxiliary stochastic-future dataset; it
does not retroactively satisfy the exact-continuation contract.

Extension selection may use model uncertainty, structural ambiguity, censoring,
or scientific interest based only on information available by 15 ps. Also extend
a configured random control subset, so the long-horizon sample is not composed
only of difficult or unusual branches. Record selection probabilities/reasons so
long-horizon analyses can account for selection bias.

Do not extend all 16-shot branches merely to obtain 24 ps labels. For a
preregistered long-horizon conditional-law check, extend all 16 branches of a
small number of selected parents, not an outcome-dependent subset of siblings
within a parent. The existing 480 old 48 ps branches and 144 nested 24 ps branches
remain the primary long-horizon reference.

Support an additional 24-to-48 ps extension for a much smaller event-time subset,
but do not make 48 ps trajectories part of mandatory campaign completion.

Maintain separate strict indexes:

1. `short_15ps_branches.json`: every valid mandatory branch.
2. `extended_24ps_branches.json`: selected branches with validated 0--24 ps data.
3. `extended_event_branches.json`: optional branches continued beyond 24 ps.

### Part F: 16-branch seed design and shot-specific information

For each new parent, use a balanced 8 x 2 design:

- eight independently sampled Maxwell-Boltzmann momentum realizations;
- two independent thermostat-noise realizations for each momentum;
- sixteen branches total.

Store the actual initial velocity field for every branch. This supports a
separate shot-conditioned model while the position/history model continues to
predict the law averaged over random branches.

Record seed integers for provenance, but do not treat them as physical ML
features. Record explicitly which branches share a momentum realization.

### Part G: paired-temperature configurations

Within the parent totals, include at least:

- four transition base configurations;
- two liquid base configurations;
- two crystal base configurations;

whose exact positions are each shot at 400, 450, 500, 510, and 520 K. Generate
16 branches at every temperature for each base configuration.

Record both `source_temperature_K` and `shooting_temperature_K`. Keep every
temperature version of one base configuration in the same split. This paired
subset is required to identify the temperature response of the conditional law
instead of learning only correlations between temperature and which structures
occurred there.

### Part H: trajectory data contract

Mandatory fixed trajectory for every branch:

- exactly 70,304 atoms;
- 3 fs LAMMPS integration timestep;
- fixed 15 ps duration;
- output every 100 steps = 0.3 ps;
- 51 frames including timesteps 0 and 5000;
- no early stopping of the mandatory fixed trajectory;
- positions and velocities in float32 binary arrays;
- atom IDs and atom types stored explicitly;
- periodic box/cell stored for every frame;
- nonempty continuation-capable restart at 15 ps;
- artifact paths, shapes, dtypes, byte sizes, and checksums in metadata.

First-passage outcomes must be measured through the fixed 15 ps run. Longer
first-passage monitoring is optional and should use the separate extension
contract. Detecting a basin must not truncate the mandatory 15 ps training path.
Record crystallization, dissolution, and censoring with exact physical times and
the censoring horizon.

Every parent must retain immutable positions, cell, physical time, source
temperature, shooting temperature, source descriptors, history, root lineage,
and split.

### Part I: leakage and weighting contract

Assign root source lineages to optimization, model selection, or final validation
before generating descendants.

All of the following inherit the root split:

- shifted parent times;
- overlapping windows;
- branches derived from old shooting branches;
- nested children;
- all temperature versions of a paired parent configuration;
- all short and extended segments of one branch.

Never count atom centers, overlapping windows, multiple times from one trunk, or
thermostat replicas sharing one momentum as independent source runs. Validation
and bootstrap units are root source lineages.

Ensure that model-selection and final-validation parents use independent root
sources whenever possible. At each temperature, final validation should contain
at least four transition, two liquid, and two crystal root lineages.

### Part J: completion and operational contract

A branch is usable only when `outcome.json` has `"state": "complete"` and all
artifacts validate. Partial trajectories and stale running status files are not
data. Preserve `interrupted_attempt_*` directories for audit and never overwrite
a completed branch.

Strict acceptance must verify:

- exact parent and branch counts;
- exact 16-future multiplicity for conditional-law parents;
- no duplicate seed pairs;
- expected 51 short-run frames and timestep sequence 0--5000 by 100;
- 70,304 atoms and stable atom-ID ordering;
- finite positions, velocities, and boxes;
- nonempty continuation restart and recorded checksums;
- exact role/temperature/split counts;
- no root lineage crossing splits;
- separate valid indexes for law parents and overlapping windows;
- extension records that never alter mandatory short-run completion;
- `summary.json` and `status.json` both reporting `"state": "complete"`.

Use CPU Slurm nodes and the repository's safe submission/recovery patterns.
Before submitting, inspect `squeue` and ensure no duplicate campaign array or
controller is active. Do not stop or consume resources belonging to VAMP,
predictive-atlas, or accelerator jobs.

Before full submission:

1. Build the immutable manifest.
2. Report exact planned counts, storage, and node-hours.
3. Run one complete 16-branch smoke-test parent.
4. Validate the 15 ps binary, history, velocity fields, continuation restarts,
   lineage, shifted-window index, outcome, and strict summarization.
5. Extend at least one smoke-test branch from 15 to 24 ps and validate the
   uninterrupted-versus-restarted continuation contract.
6. Submit the remaining work in recoverable waves.

Report the canonical roots, smoke-test result, Slurm job IDs, completion counts,
and any blocking inconsistency. Never silently alter the scientific design.

## Intended downstream use

- Existing 40-parent ensemble after top-up: 16-shot broad/pre-nucleation training
  and a high-reliability 3/6/12 ps baseline.
- New transition parents: primary difficult conditional-law training and testing.
- New liquid/crystal parents: endpoint anchoring, stability tests, and separately
  reported controls.
- 510/520 K and paired-temperature parents: continuous temperature-conditioned
  future-law tests.
- Overlapping windows: auxiliary temporal, consistency, and shot-conditioned
  training with lineage-aware weighting.
- Selected exact continuations: targeted 24 ps analysis without paying the long-
  horizon cost for every branch.
- Independent final-validation roots: the only source for final scientific
  claims.
