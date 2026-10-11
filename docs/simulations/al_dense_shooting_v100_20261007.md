# Additional dense Al velocity reshoots on V100

## Missing-dependency repair, October 7 evening

The four birth-focused workers and collector failed at Python import, before
any of the 33 transferred scientific trajectories started. The frozen snapshot
omitted `dense_al_shooting_expand.py`, which supplies the scheduler-state helper
imported by the worker. All seven retained original branches completed before
the repair; none needed an MD rerun.

The user requested repair in this chat. The maintained `repair` action creates
`technical/dependency-repair-01/` under the birth-focused launch. It preserves
the original failed code, logs, submission and manifest; the corrected snapshot
adds the missing module without changing any existing frozen source file.
The new manifest changes only its launch location. Scientific inputs, seeds,
33-source assignments, saved cadence, stopping rule and October 8 lane deadlines
are retained. Existing numerical validation is reused with an explicit hash
transition; the existing short observation/conversion execution checks remain.
The inherited V100 runtime stays PyTorch 2.6/cu118/Triton 3.2, as required by the
validated frozen adapter; repair orchestration uses `pointnet-torch214`.

Before submitting GPUs, the repair imports the actual frozen worker, monitor,
scheduler helper and converter with the production interpreter, and verifies its
complete code inventory. Future `submit` calls perform the same import check,
and future snapshots include the missing dependency.

Submitted at **20:34 Paris**: workers **1027669–1027672**, with assignments
6/9/9/9, and dependent collector **1027673**. The original launch's `repair.json`
points to the current revision; that revision's `submission.json`,
`technical/lane-*.json`, per-source status files and final `status.json` provide
current progress. The old launch's `prepared` status and failed job IDs are
historical, not live collection progress. Birth searches and cumulative-catalogue
publication remain separate work.

[Replacement jobs](al_dense_shooting_v100_20261007/dependency_repair_submission.json)
and [repair provenance](al_dense_shooting_v100_20261007/dependency_repair_provenance.json)
retain the original failed-launch identities and the added module's checksum.
At **20:38 Paris**, workers 1027669–1027671 passed the 61-frame Q/V conversion
checks and entered scientific production. Worker 1027672 was pending the GPU
quota, and collector 1027673 remained dependent on all four workers.
[Startup verification](al_dense_shooting_v100_20261007/dependency_repair_startup.json).

## Birth-focused handoff (October 7 evening)

The user changed the collection objective to observing nuclei first forming
within the recorded trajectory. **33 unstarted branches** now transfer to
`al-birth-focused-v100-20261007`. Three completed and four active branches retain
the original protocol below. Active simulations finish and publish before their
replacement GPU lane starts; their trajectories are not interrupted.

The [new recipe](../../configs/simulation/al_birth_focused_v100_20261007.json)
keeps the same 70,304-atom liquid parents, velocities/seeds, temperatures, roles,
potential, native float64 NPT integration and float32 GPU force arithmetic.
It changes observation and stopping:

- Save Q/V every **0.10 ps from initialization**, including the first 15 ps.
  Exact 0.30 ps views remain stride-three subsets. Timestamps now start at
  velocity initialization; they are not the old post-equilibration clock.
- Check whole-cell PTM (RMSD 0.1, FCC/HCP/BCC, 3.6 Å clustering) every **3 ps**.
  Two consecutive observations must contain matched connected components of
  at least **128 atoms**, sharing at least half the smaller component.
- Retain **12 ps after that trigger**, then finish and publish. The total
  physical-duration caps are unchanged: previous measurement cap plus the
  previously unobserved 15 ps. Caps can truncate follow-up, recorded explicitly.
- Keep no-trigger runs through their duration cap. They are right-censored,
  never labelled permanently liquid. Likewise, unresolved cluster fates at an
  early endpoint remain censored.

The trigger is a **collection heuristic, not a birth label, continuous
persistence certificate, or critical-nucleus estimate**. The later full-cadence
lineage search must distinguish liquid emergence, initial embryos, nearby
interfaces, absorption, fragmentation and coalescence. Failed embryos are still
retained. Complete precursor history and finite-horizon follow-up eligibility
are separate fields; an early birth lacking 3/6 ps of past observations remains
an event, not an eligible full-history prediction sample. The first 15 ps are
initial settling observations, not a claim of stationary supercooled liquid.

A read-only audit of all 20 full-precision prepared liquid configurations found
largest PTM components of **0–1 atoms**, with **zero >=8-atom embryos**. The
monitor repeats this initial-state QC and fails visibly if an embryo is present;
it does not silently replace the seed or restart until a desired outcome occurs.
Initial-state evidence is stored in
`technical/birth-focus-audit/evidence.json` under the original launch and copied
into the new immutable manifest. This does not prove absence of every form of
prestructural order; it audits the declared PTM criterion.

New jobs **1027531–1027534** each request one V100 after their respective old
worker. Assignments are 6/9/9/9 branches. They inherit the original absolute
deadlines: **October 8 13:02 Paris** for lane 0 and **17:39 Paris** for lanes 1–3;
there is no renewed 23-hour allowance. Collector **1027535** replaces the old
pending collector 1027415. Status reservations make old workers exit at the next
unstarted branch, possibly with an expected pre-source `FileExistsError`;
this is the documented queue handoff, not an MD failure.

Force/integrator code is copied unchanged from the numerically validated frozen
launch; the new receipt enumerates the observation/conversion changes. Each new
lane must pass a **6 ps execution audit** (0/3/6 ps monitors, 61 saved Q/V frames,
native restart and verified conversion) before starting scientific trajectories.
Audit trajectories are excluded from the dataset and W&B. Birth extraction and
cumulative catalogue inclusion remain **not yet performed** for this campaign.
The audit checks the execution path; it does not validate the physical identity
of every candidate or establish long-time nucleation-rate equivalence.

Before handoff, a separate 100-step native CPU execution passed the new initial
observation, monitor, capped endpoint, restart and conversion path: three Q/V
frames at 0/0.1/0.2 ps. Its [receipt](al_dense_shooting_v100_20261007/birth_focused_native_audit.json)
is an execution check, not a scientific trajectory. The per-lane 6 ps GPU checks
remain prerequisites when the replacement jobs start.

```bash
# The reserved handoff is already prepared and submitted; do not resubmit.
python -m src.simulation.campaigns.dense_al_birth_queue prepare \
  --config configs/simulation/al_birth_focused_v100_20261007.json
python -m src.simulation.campaigns.dense_al_birth_queue submit \
  --launch /store/PERSO/vmorozov/simulation-launches/al-birth-focused-v100-20261007
```

The new launch retains `handoff.json`, an immutable `manifest.json`, validation
reuse evidence, submission receipts, per-lane progress, execution audits and final
status. Completed children record `replaces_unstarted_run`; transferred branches
must not be counted twice with the original 40-branch plan. Ancestry remains the
same 20 melts. Useful yield is new, confirmed birth lineages and usable precursor
histories per GPU-hour, reported separately from initial-embryo growth, transient
embryos and front events. No increase in birth yield is claimed before analysis.

Other Al holdings—older trajectories, shooting, CPU dense descendants, the
million-atom replay and relaxed views—retain their separate execution and search
status in the shared Al pool. This handoff does not change their jobs or labels.

## Original campaign protocol

The October 7 launch adds outcome-blind velocity reshoots to the dense Al
campaign. It prepares 40 branches from 20 existing 70,304-atom liquid
preparations, with two new Maxwell velocity seeds per parent. Parent selection
uses a seeded ordering of source IDs within temperature and original role;
it does not inspect crystallization outcomes. These are reshoots of the
original liquid configurations, not continuations from saved trajectory frames.

There are eight planned branches at each of 400, 450, 500, 510 and 520 K.
The inherited roles are 22 train, four selection, four calibration and ten test
branches. They share 20 existing melt ancestors and must not be counted as
40 independent preparations or split independently of their parents.

## Execution and physical protocol

The initial launch was limited to **one V100**. A separate
one-hour validation job precedes one production worker capped at 23 hours,
so the combined GPU allocation is at most 24 GPU-hours. Prepared branches are
processed in a frozen random order; a conservative duration estimate prevents
starting a branch that is unlikely to fit. This is a bounded queue, not a
promise that all 40 branches finish. Remaining branches stay unstarted for
an explicitly requested later expansion.

Submitted jobs: **1026772** (V100 validation), **1026773** (one V100 production
worker, dependent on passed validation), and **1026774** (CPU publication/status
collector). At submission, validation was waiting for resources and production
was waiting for validation. The local Volta topology-extension build and all
320 prepared-input hash checks passed; the GPU numerical audit had not run yet.

Both V100S numerical audits subsequently passed (1.471/1.511 seconds per
100 steps at the 400/520 K starts). The first production attempt completed
equilibration but failed at measurement zero because preparation omitted the
empty `technical/stop-monitor` directory. The failed attempt is archived;
all 40 prepared branches now have the required directory and unchanged input
hashes. Replacement worker **1026907** and collector **1026908** reuse the
passed audit and identical frozen simulation code. The preparation producer
now creates this directory explicitly. That replacement requested one GPU worker.
The launch's `technical/restart-missing-monitor/receipt.json` records the repair.

### October 7 expansion to four V100 workers

The user subsequently authorized three additional V100 jobs. Array
**1027414_0–2** adds three workers, one GPU and a 23-hour cap each, alongside
the unchanged running worker **1026907_0**. At submission the new lanes started
on node10, node42 and node42 (separate allocated GPUs). Each new lane receives
10 unstarted branches; the original lane retains its first 10 entries, which
included three completed branches and the active branch at capture. All 40
original branches remain assigned exactly once. Parent selection, velocities,
native inputs, numerical audit and frozen physical code are unchanged.

The original worker already holds its full queue in memory. Its existing
pre-source status guard therefore stops it at the first transferred branch,
after publishing its retained sources. This is an expected `FileExistsError`
queue handoff, potentially a Slurm `FAILED` worker state, not a failed or
interrupted MD trajectory. A permanent reservation prevents concurrent starts;
the new lane processes this boundary branch last and waits for the original
job to end before removing that reservation. All other transferred branches
can run immediately. Budget-limited branches remain explicitly unstarted.

Collector **1027415** waits for both worker arrays and reconciles all 40
branches; the superseded pending collector **1026908** was canceled. No running
job was stopped. This changes only the scheduling of this chat's V100 campaign.
The frozen expansion driver invokes the original numerical/runtime checks,
MD, monitoring, conversion and publication producers.

Recipe: [three-worker expansion](../../configs/simulation/al_dense_shooting_v100_expand_20261007.json).
Live receipts are in
`/store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007/technical/three-more-v100-20261007/`.
The [submission receipt](al_dense_shooting_v100_20261007/expansion_submission.json)
and [exact assignments](al_dense_shooting_v100_20261007/expansion_plan.json)
are also retained with this document.

The potential is the pinned Lee2003 pure-Al MEAM, evaluated by `FastAl` on GPU.
Native LAMMPS integrates float64 positions and velocities under isotropic NPT:
2 fs timestep, 0 bar, 0.3 ps thermostat, 3 ps barostat, and momentum removal
every 0.3 ps. Forces and normal virials use float32 arithmetic; scalar energy
accumulation is float64. The original full-precision liquid file is reused,
new velocities and a fresh NPT bath are initialized, and 15 ps equilibration
precedes measurement time zero. No constraints or artificial nuclei are added.

The existing dense campaign stopping policy is retained: two observations
15 ps apart at or above 50% PTM crystallinity, followed by up to 6 ps of
additional observation, bounded by the existing temperature-dependent caps:

| Temperature (K) | Maximum measurement (ps) |
| --- | --- |
| 400 | 600 |
| 450 | 291 |
| 500 | 411 |
| 510 | 411 |
| 520 | 561 |

Reaching a cap is censoring, not a permanent-liquid label. Positions and
velocities are saved at exact 0.10 ps intervals; exact 0.30 ps views can later
use every third observation. Native restarts preserve integration precision.
The ordinary checked streaming converter verifies float16 positions/velocities,
float32 boxes, exact identities/timesteps and quantization errors before deleting
the text dump. Complete runs and failures with restarts are published to STORE.

## V100 runtime and validation

The user authorized downgrading the older `pointnet` environment. Its dedicated
execution profile now uses PyTorch 2.6.0+cu118, Triton 3.2.0, matching torchvision
0.21.0+cu118 and xformers 0.0.29.post3. `pointnet-torch214` is unchanged. Compiler
and CUDA paths live in ignored `machine.local.yaml` under
`execution.gpu_meam_v100`; CUDA 11.8 and GCC 11 are installed separately on
shared storage because the inspected V100 node has no system CUDA compiler.
The CUDA topology extension explicitly targets `sm_70`.

Modern PyTorch CUDA 13 builds omit Volta, and newer Triton releases target
newer architectures; this is why the older, pinned runtime is explicit.
See the [PyTorch CUDA support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
and [Triton project](https://github.com/triton-lang/triton).

Production depends on successful native-vs-GPU 100-step full-cell NPT audits
at 400 and 520 K. Gates are force RMS <= 0.0005 eV/Å, energy error <= 0.00002
eV/atom, normal pressure error <= 2 bar, final position RMS <= 0.001 Å and
relative volume error <= 0.00001. The audit binds the frozen production code
inventory. It checks numerical execution, not long-time crystallization-rate
equivalence. Failed validation cancels the dependent production job.
Validation/setup runs are excluded from scientific samples and W&B.

## Commands and records

Recipe: [`al_dense_shooting_v100_20261007.json`](../../configs/simulation/al_dense_shooting_v100_20261007.json).
Implementation: [`dense_al_shooting.py`](../../src/simulation/campaigns/dense_al_shooting.py).

```bash
# Preparation/submission each run once; existing manifests are protected.
conda run -n pointnet python -m src.simulation.campaigns.dense_al_shooting prepare \
  --config configs/simulation/al_dense_shooting_v100_20261007.json
conda run -n pointnet python -m src.simulation.campaigns.dense_al_shooting submit \
  --launch /store/PERSO/vmorozov/simulation-launches/al-dense-shooting-v100-20261007
```

The launch directory contains `manifest.json`, frozen code and checksums,
`submission.json`, validation receipts, per-lane progress and a final
`status.json`. The collector runs after the GPU jobs and separately reports
complete, failed and not-started branches. No pre-existing CPU/GPU jobs are
modified by this workflow. The initial device-inspection job 1026707 found a
32-GB V100 and no system CUDA compiler; it is setup evidence, not MD.

The registered collection is `al-dense-shooting-v100-20261007`. Each completed
child retains `source_origin=shooting`,
`sampling_protocol=prepared_liquid_velocity_reshoot_v1`, melt ancestry,
original role and its new velocity seed. The shared search inventory admits
this explicit producer contract without treating it as an older shooting-prefix
or diagnostic file. Birth extraction/search is a later operation: submitted
MD is not yet part of `al-nuclei-birth-010` or `al-nuclei-birth-030`. Earlier
shooting, dense, million-atom and relaxed holdings retain their own coverage;
this launch is an addition to the shared Al pool.
