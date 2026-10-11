# Continuous motion from 0.1 ps Al snapshots

The [active1600-update recipe](../configs/continuous_motion/all010_1600_20261006.json) runs nine
matched fits: fixed0.1ps steps, continuous latent dynamics, and the same
continuous dynamics with simulator derivatives, each at three random seeds.
The [scientific protocol](../experiments/continuous_motion_20261006/README.md)
and [metric definitions](metrics/continuous_motion.md) specify what this pilot
can establish.

## Inputs and execution

The pinned all0.1ps inventory is reused without its nucleus or neighborhood
filters.88 verified-MEAM views have native velocities; four historical views
with unknown potentials are explicitly excluded. Every70304-atom source has
two outcome-blind anchor times and the same64 Al64 center identities. Native
0.01ps sources are sampled at exact0.1ps during fitting; their intervening
saved observations are diagnostics. Million-atom/no-velocity sources and
relaxed views remain in the wider Al pool, outside this matched Q,V assay.

The encoder sees one unrelaxed local80 neighborhood and actual same-atom V.
Native MACE uses one constant atom channel,128 channels and128 structural
coordinates. Its directional derivative along V gives128 motion coordinates.
The predictor sees only these256 coordinates. Temperature, simulation age,
absolute time, species and external halo atoms are absent. Elapsed duration
controls numerical integration and never enters a neural layer.

The physical teacher is the existing FastAl implementation. Each anchor has
one selected center, two thermostat shots and two local tangent directions.
Whole periodic cells evolve for100fs at1fs resolution with fixed-box BAOAB,
source T and1000fs damping. Forces usefloat32; integration, momentum, tangent,
noise and observables usefloat64. This does not reverse initial archive
quantization or reproduce the original NPT thermostat/barostat state. Only
local prequantization response observables are archived, not new full-cell
movies. New observations go to SCRATCH, then verified copies to STORE.

Long0.1/0.5/3/12ps supervision uses already-saved native futures. Derivatives
are propagated only over100fs. There is no backpropagation through12ps of MD.
All arms receive identical short latent endpoint values. Only the derivative
arm receives simulator sensitivities.

Training uses batch256/microbatch256,1600 updates per seed, with online W&B
in `teshbek/PointCloudMaterials`, stable resumable IDs and local receipts.
Full-batch forward/backward used64.9GB in the local development profile.
No scientific W&B run was created for development checks.

## Commands and ownership

Use conda `pointnet-torch214` and the trusted installed e3nn serialization
setting already required by the other native MACE workflows:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.workflow prepare --config configs/continuous_motion/all010_1600_20261006.json
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.workflow launch --config configs/continuous_motion/all010_1600_20261006.json
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.workflow status --config configs/continuous_motion/all010_1600_20261006.json
```

The active training run is `${storage:training_storage}/continuous_motion/all010-20261006-v3`.
Its immutable v2 data plan,176 completed teachers and11200 deduplicated rows
(6080 fitting,1024 inner selection,4096 test) are reused without resimulation.
The original400-update v2 recipe and launch receipts are preserved; its own
workers and queued handoff were replaced after collection, before any fit
started. The earlier v1 directory contains development measurements only. The launch
freezes source, metrics and configuration; every worker imports that copy.
Shared filesystem leases prevent duplicate collection and fitting. The user
explicitly relaxed the original16-hour limit when requesting at least1600
updates and more GPUs. The extension uses a24-hour operational window including
queue wait. Current-node GPU0/1 contribute immediately; two independent two-GPU
jobs request four additional RTX6000PRO/H100 devices. These preserve microbatch256
on80/96GB cards. A separate two-GPU handoff is queued after the shared allocation
ends and exits immediately if the campaign has finished. Neither
the dependency nor the worker changes the shared allocation or other jobs.

`technical/launch.json`, per-worker launch receipts, `budget.json`, hardware
records and logs identify only this chat's jobs. Workers reserve10 minutes
before their allocation ends. Training checkpoints every update; unfinished
fits resume model, optimizer, target encoder and random generators. A bounded
100fs teacher job is atomic; an interrupted teacher is recomputed from its
recorded input/noise seed. Its failure receipt is also archived to STORE.

On an authorized allocation, resume recorded work with the frozen command:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.workflow resume-local --config /ABSOLUTE/RUN/technical/code/config.json
```

Use the frozen source directory as cwd. Resume retains the original deadline
and does not replace a live recorded worker. W&B authentication/network
failures are reported; offline tracking is never substituted.

Progress and completion are recorded separately for observations, teachers,
individual fits and evaluation. Comparison CSVs live under
`analyses/comparison-v1/tables/`; every numerical export includes METRICS.md
and implementation hashes. Final metrics update each training run's existing
W&B ID. Wider Al collection, birth analysis, recovery and old shooting studies
retain their own running/completed/not-analyzed states.

## Separate smaller-GPU requests

The user additionally authorized A40, A100 and L40S, each requested separately
with one GPU. The [execution profile](../configs/continuous_motion/small_gpu_20261006.json)
uses microbatch128 with two-step accumulation, effective batch256 and the same
1600 updates. These workers share the existing nine-fit queue and checkpoints;
they do not add another set of experiments. Original RTX6000PRO/H100 requests
and running256 workers retain their frozen configuration.

The noncollapse penalty still uses all256 observations, through cached
embedding gradients and recomputation. Solver noise is sliced from the same
reference full-batch stream. The manual comparison on actual fitting inputs
matched the reference gradient to relative error1.66e-7 on CPU. This verifies
the accumulation calculation; it is not a claim of bitwise identity across GPUs.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.small_gpu launch --config configs/continuous_motion/small_gpu_20261006.json
```

Separate submission receipts, frozen source and verification hashes live under
`technical/small-gpu-v1/`. Each actual execution segment records its memory
profile and implementation; its final metric export records the producer.
Only one worker holds a given fit lease. The same original v3 operational
deadline applies to these requests.

## Nucleus-birth transfer at 1, 3 and 6 ps

The [birth recipe](../configs/analysis/continuous_motion_birth_20261006.json)
evaluates all nine frozen encoders on exact1/3/6ps pre-emergence observations.
It reuses all92 velocity-bearing source views from the three pinned v3 nucleus
releases, extracts new lead-specific inputs and preserves fixed Al64 ancestry.
Only the encoder teacher collection excludes unknown-potential historical views;
those views remain in the labelled transfer population. No new MD is needed.
The established neighborhood eligibility and physical-event deduplication remain
in force. Final eligible event counts are written after source preparation.

Structural128, structural+motion256 and predicted-future-law48 each receive
nonlinear CatBoost and linear diagnostic readouts. The pooled test combines the
original validation/calibration/test; all readout selection and refitting use
original training ancestors. Predictions do not use time or temperature features.
AP, NLL, Brier and matched-set metrics have whole-ancestor uncertainty; see the
[complete definitions](metrics/continuous_motion_birth.md). The1:4 retrospective
assay is not population nucleation probability.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.birth launch --config configs/analysis/continuous_motion_birth_20261006.json
```

Output is the campaign's `analyses/birth-136-v1`; source arrays live in the
registered `continuous-motion/birth-136-20261006-v1` cache. Four CPU workers
prepare sources; the free local GPU1 runs frozen inference and171 diagnostic
fits. A still-running encoder is read only after its completion receipt appears.
The detached evaluation is capped at two hours and reserves five minutes before
the allocation ends. Exact owner/PID, source, progress and checkpoint bindings
are in `technical/`. Scientific training on GPU0 and other jobs are unchanged.
No new W&B run is created; final results update the existing nine run summaries.

The first birth evaluation completed all92 source preparations but stopped on
October7 at00:01 Paris after36 readout fits. Its full-block stochastic replay
assumed CUDA random draws were identical across A100 and RTX. PyTorch's normal
kernel uses GPU-dependent launch geometry: the first216 of256 saved prediction
rows replayed within6.8e-6, while the remaining40 used different normal draws.
The resumed execution audits the common first64 Philox subsequences with the
same2e-4 tolerance and records the full-block difference separately. Scientific
inference, observations, checkpoints, readout fitting and the original deadline
are unchanged. Failure logs and frozen code remain in `technical/`; the new
bundle and owned launch receipt are under `technical/resume-v2`.

The sealed longer-lead cohort has16 training births from15 ancestors and7 pooled
test births from4 ancestors. Of210 candidate event views,182 have no eligible
sampled center under the common three-lead neighborhood screen and4 lack the
required history/followup.24 accepted views represent23 physical events after
alias handling. These view counts are not independent-nucleus counts.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.continuous_motion.birth resume --config configs/analysis/continuous_motion_birth_20261006.json
```

## Present-selected embryo establishment

The user approved a separate embryo-to-nucleus cohort on October 7. The
[recipe](../configs/analysis/embryo_establishment_20261007.json) samples existing
8–63 atom embryos at fixed 1 ps landmarks, before classifying their futures.
The same current input receives labels for establishment within 1, 3 and 6 ps.
Establishment means a connected 64+ atom interval persisting for 1.5 ps; these
are leads to the interval's start, with persistence confirmed afterward.
The [cohort definitions](metrics/embryo_establishment_cohort.md) give spatial
filters, censoring, exact physical aliases, weights and exported count formulas.

This first release reuses all 92 pinned velocity-bearing source views, including
historical partial archives and exact 0.1 ps views of finer data. Four historical
views retain unknown generating potentials. The 22 newly completed trajectories
are being searched separately and require a subsequent expanded release; they
are not silently included before their labels are complete. Velocity-free
million-atom, other-cadence and relaxed collections remain outside this Q/V assay.

Every input contains one present unrelaxed nearest-80 position patch and native
same-atom velocities in A/ps. No future members choose the center; no temperature,
age, timestamp or future outcome enters Q/V. Current embryo size is recorded for
an explicit later control, not added to encoder tensors. Centers near existing
crystals are excluded. Non-establishing and dissolved embryos provide negatives;
unresolved competing fates are censored and counted, not assigned false negatives.
All horizons share the same observations and at least 7.5 ps of source followup.

The collection preserves original ancestor roles. Later fitting must use only
original training ancestors, with validation/calibration/test pooled for evaluation
as requested. Repeated landmarks, overlapping archives and multiple embryos in
a melt are not independent melts. This cohort asks about embryo establishment;
the strict pre-eight-atom birth benchmark and its completed metrics are unchanged.

```bash
python -m src.research.continuous_motion.embryo_cohort prepare --config configs/analysis/embryo_establishment_20261007.json
python -m src.research.continuous_motion.embryo_cohort launch --config configs/analysis/embryo_establishment_20261007.json
```

The launch freezes code and definitions and detaches eight CPU workers under the
current allocation deadline. It records ownership and PID in
`${storage:training_storage}/continuous_motion/embryo-establishment-20261007-v1/technical/launch.json`.
Source packets are independently checksummed and retained; final counts require
all sources and cross-view deduplication. Tables and their definitions appear
under `analyses/cohort-v1/tables`. Collection needs no new MD or neural training;
it does not create a W&B run. Frozen-encoder scoring is a subsequent analysis.

The first embryo cohort completed with 34,060 deduplicated observations across
84 melt ancestors: 21,651 training and 12,409 pooled-test rows. Pooled test has
1,776 embryo families across 29 ancestors. Its 1/3/6 ps positive-family counts
are 29/54/67 (positive observations 29/109/252). These are establishment labels,
not additional strict pre-eight-atom births. Input and label reconstruction
checks are in the run's `technical/packet-audit.json`; numerical definitions and
complete cohort counts remain in its sealed analysis tables.
