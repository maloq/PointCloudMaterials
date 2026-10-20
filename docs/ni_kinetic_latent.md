# Native Ni kinetic-latent workflow

This workflow compares six measurement encoders with a common probabilistic
latent propagator and decoder. The [scientific protocol](../experiments/ni_kinetic_latent_20261015/README.md)
records literature choices, frozen objectives, actual nucleus prediction and
the separate generated local-ordering task.

Recipe: `configs/ni_latent_sim/kinetic_latent_20261015.json`.
Implementation: `src/research/ni_kinetic_latent/`.
Result root: `${storage:training_storage}/ni_kinetic_latent/native-20261015-v1`.
Derived cache: `${storage:cache}/ni-kinetic-latent/native-20261015-v1`.
Registry entry: `ni-kinetic-measurements-20261015`.

## Inputs and preparation

The native input is the sealed direct-measurement bank from
`${storage:training_storage}/measurement_sim/direct-20261011-v1`.
Its plan SHA256 is `91812d6d7ab012b1a24221cf9b40d90ccdb8105389097b7c242f39adcc2ca5d8`.
All 14,400 windows and 244 measured channels are reused. The new `data.pt` holds
training-fitted affine/asinh/affine-normalized values, channel weights,
normalizers and identity/role records. There are 3,840 fitting windows from 16
ancestors and 10,560 pooled test windows from 44 ancestors. The original roles
remain provenance. Preparation verifies the source cache hash, actual native
timestep arrays, exact window offsets, duplicate identity and cross-role ancestry.
No raw MD, new simulation, interpolation or broad spatial training bank is created.

Birth observations reuse the history bank from
`${storage:training_storage}/ni_phase_history/attention-20261013-v1`, plan SHA256
`122296aa06f638acc8a4224eaf9335b15f51ef11216afed03a455bc4c70756c8`.
The cached measurements have 19,652 unique four-frame histories and 19,800 assay
rows. Declared catalogue ancestry is 28 train / 62 test; represented assay rows
contain 21 train / 42 test ancestors. The encoder uses present measurements
only. The existing history control retains its true support: present neighbors
held fixed backward, unlike independently reselected native training neighbors.

The independent annotation stage produces `labels.pt` with central PTM type,
central crystalline boolean, local crystalline fraction and central tracked
component size at every window observation. It reuses checksum-bound full-cell
PTM extraction or the established cached/missing-label producer. Crystal types
are 1, 2 and 3 at PTM RMSD cutoff 0.1. Local fractions use the existing nearest-80
support within normalized radius 8 Å, including the center. A zero component
size means the atom is absent from a tracked component of at least four atoms;
it is not substituted for a liquid label. Annotations are evaluation targets,
not representation inputs or a new birth filter.

Per-source annotation packets and partial checkpoints permit independent resume.
Final annotation receipts are `cache/labels.json` and
`technical/labels-complete.json`. The completed packet retains all
14,400 × 64 observations. Old PTM caches and their original provenance are
preserved; only missing atomic classifications use the established producer.

## Detached execution

Use conda `pointnet-torch214`. The maintained entry point is:

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.ni_kinetic_latent.queue launch --config configs/ni_latent_sim/kinetic_latent_20261015.json
```

The queue freezes the implementation/configuration and starts concrete detached
workers without requiring a future chat or scheduled review. Four GPU lanes
request compatible RTX6000PRO/A100/A40/L40S types as alternatives. CPU annotation
is independent of representation training, so missing physical annotations do
not prevent scientific fits from starting. Model stages fit the representation,
a fresh frozen-encoder decoder, and the mixture-density propagator; completed
models have independent trajectory and nucleus evaluations.

All work shares one absolute 12-hour deadline. Checkpoints preserve optimizer,
stage and RNG state; bounded retries retain failure receipts and independent
fits proceed. A partial checkpoint is not promoted to the fixed final result.
Any allocation handoff remains within the original deadline. Only receipt-owned
workers and their subprocesses may be controlled; other jobs are untouched.

There are 18 scientific pipelines: six methods × three seeds. Linear encoder
fits have no gradient epochs, but their matched decoder and stochastic dynamics
are trained independently for each seed. Nonlinear encoders, fresh decoders and
dynamics each use 12 full epochs. The explicit batch is 1,024. Neural operations
are float32; VAMP covariance/Cholesky calculations use float64. These are learned
native 0.1 ps transitions, not physical integration steps.

All scientific pipelines log online to `teshbek/PointCloudMaterials`, with
stable `nikinetic-` run IDs. Frozen phase/nucleus probes, preprocessing and
performance checks stay local; final evaluation metrics update the existing
scientific run. An authentication or network failure is recorded rather than
silently switching to offline mode.

## Initial launch receipt

Launched October 15 at 14:17 CEST, with one absolute deadline of October 16 at
02:17 CEST. Local worker PID 1189532 uses the current RTX6000 allocation 1041402.
GPU jobs 1041483/1041484/1041485 started on A40s; independent annotation job
1041482 started on CPU. All four scientific workers had actual checkpoint
progress and online W&B receipts. The first PCA pipeline completed 5,004 updates
and 24 frozen nucleus readouts during the launch audit. This is execution
verification, not a comparative scientific conclusion. The authoritative receipt
is `technical/submission.json`; these are the only new jobs owned by this launch.

## Inspecting progress and results

Read `technical/plan.json`, queue launch/submission receipts and per-lane worker
logs for immutable identities and execution ownership. Each model's
`analyses/<arm>/technical/progress.json` names its stage and update; `last.pt`
and `final.pt` are resumable and final checkpoints. The final `complete.json`
records the checkpoint hash. Annotation writes
`technical/annotation-progress.json` and per-source partial packets.

Metric tables live under `analyses/<arm>/tables/` with `METRICS.md` and frozen
implementation hashes. Figures under `plots/` show train-projected latent order,
predetermined observed/generated order trajectories and local-entry curves.
Generated trajectories are labeled local measurement forecasts. Actual
32/64-atom nucleus readouts use the existing full-cell catalogue separately.
No whole-cell generated nucleus is inferred from independent local samples.

The local preparation receipts `technical/data-preflight.json`,
`annotation-preflight.json` and `annotation-source-preflight.json` document
normalization roundtrip, preserved ancestry and annotation agreement checks.
They are engineering checks, not scientific outcome evidence. Current stage
status and launch timing must be read from live receipts; this workflow document
does not itself assert completion.
