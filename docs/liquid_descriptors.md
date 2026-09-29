# Rich descriptor baseline workflow

Run with conda `pointnet-torch214`:

```bash
python -m src.research.liquid_predictability.descriptor_queue submit --config configs/liquid_predictability/descriptors_al64_20260928.json
```

The active queue uses dedicated Slurm GPUs for CatBoost and CPUs for extraction,
the no-input prior and the MLP. It snapshots source, metric contracts and the named
recipe. The current recipe reuses the eight already-running extraction tasks and
their sealing job, then runs two GPU boosting lanes, one CPU control lane, and a
final comparison. Each GPU lane requests exactly one GPU from RTX6000PRO or H100.
MACE keeps its own GPUs. Preparation tasks interleave sources sorted by cache size.
Ten fits run in total. W&B stays local for these descriptor controls, in accordance
with the tracking policy; no online debug or benchmark runs are created.

Geometry is reused from the sealed dense Al64 cache. Descriptors and progress are
written beneath `${storage:cache}/liquid-predictability/descriptors-al64-20260928`,
not the repository. The study outputs and frozen commands are under
`${storage:analysis}/liquid_predictability/descriptors-al64-gpu-20260928`.
The original extraction launch remains under `descriptors-al64-20260928`.
Source extraction checkpoints every 8192 unique patches. Resume the recorded
`prepare --index N` command using the frozen configuration; completed sources are
verified and skipped. CatBoost snapshots and MLP optimizer/RNG checkpoints allow
fit recovery. Failures have explicit state JSON and tracebacks.

Approximately 7.9 million unique eligible patches are computed once and shared
between every feature subset and model. Source-local patch vectors and compact
context vectors are memory mapped. CPU parallelism is across patches; no full
trajectory or GPU tensor bank is loaded. Final training uses all 183,596 eligible
training rows, preserving 41,418 selection, 34,117 calibration, and 66,839 test rows.
There are 88/15/15/28 eligible sources respectively.

Before submission, save a local real-data numerical check in
`technical/preflight.json`, containing the exact configuration hash and finite
descriptor/model result. Such checks do not open W&B runs or write automated tests.

Each fit stores inputs, feature families, selected iteration, model checkpoint,
learning curves, prediction probabilities and proper scores. The final
`analyses/comparison-v1/README.md` identifies the validation-selected descriptor
and whether it beats the matched no-input prior. CSVs have frozen metric contracts.

[Scientific protocol](../experiments/liquid_predictability_20260928/DESCRIPTORS.md)
and [metric definitions](metrics/liquid_descriptors.md).

Preparation array `1013122` and sealing `1013123` remain on CPU. Pending CPU fit
lanes `1013124` and comparison `1013125` were cancelled at the user's GPU request,
before any scientific descriptor fit started. Their original frozen recipe is
preserved. The GPU launch has a separate output and receipt; feature preparation
and all input/target/split definitions are reused unchanged.

GPU MultiClass cannot use the originally proposed `rsm=0.7`; the active recipe uses
all features within each declared subset. This change is explicit in configuration
and protocol. Resolved CatBoost parameters are recorded with every checkpoint.

GPU replacement launch: CPU prior/MLP job `1013156`,
GPU boosting array `1013157` (two lanes, one GPU each),
and comparison `1013158`. These depend on the existing sealing
job `1013123`; extraction is not repeated. A real-data GPU numerical check passed
on node60 before submission.

## Mean and linear controls

```bash
python -m src.research.liquid_predictability.descriptor_baselines submit --config configs/liquid_predictability/descriptor_baselines_20260928.json
```

This adds a training-mean point predictor, affine ridge regression, and a linear
probability model over the same bins as boosting. It uses the sealed descriptor
cache and a separate CPU job; no original fit is restarted. Transforms fit training
only and regularization is selected on validation likelihood. The mean/ridge have
point-error scores; the linear probability model also has NLL and Brier scores.
The augmented report is `analyses/comparison-with-baselines-v2/README.md` under
the GPU study. The original comparison remains intact. See the
[additional metric definitions](metrics/liquid_descriptor_baselines.md).

Mean/linear launch (2026-09-28): CPU fit `1013226`, augmented
comparison `1013227`. Controls remain locally tracked.
