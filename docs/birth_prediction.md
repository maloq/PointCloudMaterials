# Pre-appearance birth classification workflow

## Full-cell relaxed input comparison

[Recipe](../configs/birth_prediction/relaxed_temporal_20261001.json) and
[protocol](../experiments/birth_prediction_20260930/RELAXED.md) quench all 1,030
unique observed cells from the retained birth cohort. Nearest-80 membership,
labels, source roles and folds stay fixed. Sixteen 16-rank MPI CPU workers
minimize cells with the original Al MEAM potential; failed cells are archived
and block the full-cohort fit. Local float32 clouds are saved before global
float16 conversion. Patches are never minimized in isolation.

```bash
python -m src.research.birth_prediction.relaxed freeze --config configs/birth_prediction/relaxed_temporal_20261001.json
python -m src.research.birth_prediction.relaxed submit --config configs/birth_prediction/relaxed_temporal_20261001.json
```

Submission requires one real converged-cell receipt and verified descriptor
and both frozen-encoder exports on that cell. The frozen queue then
seals every original row, computes descriptors, exports both frozen MACE
encoders, and repeats all 474 temporal/site readout fits. These frozen probes
stay local rather than creating W&B training runs. Jobs execute independently
of the interactive allocation under Slurm dependencies.

Output: `${storage:training_storage}/birth_prediction/relaxed-temporal-20261001`.
Top-level `RESULTS.md` contains raw train/test NLL and Brier with AP diagnostics.
All raw/calibrated scores and generalization gaps are in
`analyses/relaxed-versus-md-v1/tables/train-test-errors.csv`; paired source
uncertainty is in `paired-domain-differences.csv`. Original outputs are read
without changing their exported definitions. See [metrics](metrics/birth_prediction_relaxed.md).

Execution receipts and worker logs live under `technical/`. For a stopped
worker, use the retained frozen `technical/code/config.json` and run the same
`worker --index N` command. Completed cells verify and skip; an unconverged
full-precision dump is reused as an explicit minimization restart. Successful
clouds and cell provenance are published to STORE before SCRATCH cleanup.
After all workers succeed, resume the retained dependent stage scripts. No
missing cells or rows may be dropped to unblock fitting.

## Fixed-endpoint and site-persistence comparison

[Protocol](../experiments/birth_prediction_20260930/TEMPORAL_SITE.md) and
[recipe](../configs/birth_prediction/temporal_site_20260930.json) reuse the complete
parent data and encoder banks for 13 independently fitted observation treatments.
No simulations, PTM exports or encoder retraining are needed. Six readouts plus
the constant prior yield 474 small fits across the fixed test and five original
readout-CV folds. The explicit frame lists separate fixed-endpoint memory from
fixed-four-frame lead comparisons; see [definitions](metrics/birth_prediction_temporal.md).

```bash
python -m src.research.birth_prediction.temporal prepare --config configs/birth_prediction/temporal_site_20260930.json
python -m src.research.birth_prediction.temporal submit --config configs/birth_prediction/temporal_site_20260930.json
```

Preparation validates every numerical input layout on the real retained banks,
the immutable release and fold assignment, and source-disjoint partitions.
Submission freezes source/configuration/metric contracts. One CPU worker and one
one-GPU worker run independently under Slurm, each processing the fixed test and
five CV partitions serially to fit the account's submitted-job quota. Final
collection waits for both workers, then analyzes within-site descriptor changes,
compares predictions and replays locked models.
Frozen probes and descriptor controls stay local under the tracking policy.

Results live outside the repository at
`${storage:training_storage}/birth_prediction/temporal-site-20261001`.
`technical/launch.json` records jobs and scripts. Per-fit contexts record observed
frames and transformations separately from frozen encoder inputs. Final tables,
paired source-bootstrap intervals and PNG/PDF plots are in
`analyses/{fixed_test,readout_cv}-comparison-v1`; locked early/current predictor
replays are in `analyses/{fixed_test,readout_cv}-replay-v1`; direct structural
within-site changes are in `analyses/site-structure-v1`. The root `RESULTS.md`
is written only after fitting and evaluation complete.

The original test is already inspected, so this follow-up remains exploratory.
Site controls characterize the observed 5.25 ps window in a retrospective
case/control population. They do not remove future-based site selection or
establish prospective natural-risk calibration.

## Original assay and earlier follow-ups

[Leakage and important-feature audit](encoder_research/birth_leakage_features.md)
documents retrospective sampling, raw-observation replay and matched structural
feature reliance. All retained predictive scores keep their original definitions.

[Scientific protocol](../experiments/birth_prediction_20260930/README.md) and
[recipe](../configs/birth_prediction/preappearance_20260930.json) specify the
44 local frozen-readout fits. Use conda **pointnet-torch214**.

Reuse the complete original Al64 full-cell audit. No simulation, relaxation or
PTM recomputation is scheduled. Historical source roles and event definitions
are read-only. Derived coordinates/descriptors live in IDS; readable results,
fitted readouts, predictions and frozen code live in STORE outside the repository.

The queue uses eight CPU preparation shards, a support-gated seal, eight CPU
descriptor shards, two sequential one-GPU frozen encoder exports, then CPU linear
controls and one GPU boosted-tree lane in parallel, followed by source-bootstrap
comparison. Each stage requires successful dependencies. A failed source or
insufficient event coverage stops downstream fitting with explicit receipts.

```bash
python -m src.research.birth_prediction.queue bind --config configs/birth_prediction/preappearance_20260930.json
python -m src.research.birth_prediction.queue prepare --config configs/birth_prediction/preappearance_20260930.json --source 860
python -m src.research.birth_prediction.queue submit --config configs/birth_prediction/preappearance_20260930.json
```

Submission requires a matching local scientific preflight receipt: real training
histories, finite rich descriptors, exact frozen-source model reconstruction and
finite scalar exports. These checks create no online runs. All new fits here are
frozen diagnostic readouts or descriptor controls and remain local under current
tracking policy. Encoder training is not resumed, and no W&B evaluation run is
created for each fit.

The rich encoder uses the published epoch14 RH2 snapshot/source. The old VICReg
encoder has different native code and runs in its own interpreter with the
recorded producer prepended to Python imports. Checkpoint/source checksums and
pool/trunk normalization are preserved. Inference defaults to256 patches, native
cuEquivariance and BF16; no inference-time fine-tuning occurs.

Neural states occupy the existing shared six-entry feature cache with protected
leases. Metrics, checkpoints and predictions are never disposable. A durable
`technical/encoder-cache-<name>.json` records the cache identity; if an entry has
been evicted, rerun the original `encode` stage from frozen code before fitting.
Dataset coordinates and descriptor banks remain reproducible source-sharded
artifacts in IDS, rather than repository files.

`technical/launch.json` holds job IDs and frozen code. Stage JSONs distinguish
running, complete and failed. `analyses/coverage-v1/tables/split-coverage.csv`
shows independent birth support before training. Per-source event-coverage
records retain zero-yield events and exclusions. The final comparison report
and endpoint plots appear in `analyses/comparison-v1/`.

Completed source shards and fits resume only with identical identities and
checksums. Descriptor extraction safely recomputes an interrupted source shard;
boosting has native snapshots. For manual stage recovery, use the command in
the saved Slurm script and the frozen `technical/code/config.json`, rather than
mixing new repository code into an existing scientific release.

## Truncation to one frame and readout cross-validation

[Scientific extension](../experiments/birth_prediction_20260930/TRUNCATION_CV.md)
and [recipe](../configs/birth_prediction/drop_to_one_cv_20260930.json) reuse the
same completed data and frozen feature banks. No new simulation, PTM, data
release or encoder training is involved. The four additional endpoints retain
4/3/2/1 observations at leads 3.75/4.50/5.25/6.00 ps. Original fits remain intact.

```bash
python -m src.research.birth_prediction.extension prepare --config configs/birth_prediction/drop_to_one_cv_20260930.json
python -m src.research.birth_prediction.extension submit --config configs/birth_prediction/drop_to_one_cv_20260930.json
```

Preparation verifies the parent config/data/feature identities and freezes five
training-source/ancestry folds. Selection and calibration sources stay fixed;
test remains separate. Encoders saw these training sources during pretraining,
so this is readout CV conditional on fixed features, not encoder CV. The primary
fixed-test comparison is separate. See [definitions](metrics/birth_prediction_extension.md).

Detached Slurm lanes add 44 fixed-test readouts and 440 CV readouts. CPU CV runs
at most two folds concurrently; the GPU CV lane uses one GPU per fold and one
fold concurrently. Scientific frozen readouts remain local under tracking policy.
An independent `plot-existing` job immediately exposes all six current metrics
with source-bootstrap bands while the new fits run. Comparison waits for every
fitting lane; a failed fit stops the final stage, preserving its failure receipt.

The extension output is `${storage:training_storage}/birth_prediction/drop-to-one-cv-20260930`.
Use `technical/launch.json` for job IDs and frozen source; `technical/folds.json`
for assignments and explicit pretraining exposure. Plots and PDF figures live
in `analyses/existing-visualization-v1`, `analyses/fixed-test-comparison-v1` and
`analyses/readout-cv-comparison-v1`. Final `RESULTS.md` links the fixed-test/CV
panels; numerical tables include definitions, provenance and implementation hashes.

## Leakage and feature audit

Use the retained models and dataset, with no refitting or online tracking:

```bash
python -m src.research.birth_prediction.leakage audit --config configs/birth_prediction/leakage_features_20260930.json
python -m src.research.birth_prediction.leakage explain --config configs/birth_prediction/leakage_features_20260930.json
python -m src.research.birth_prediction.leakage supplement --config configs/birth_prediction/leakage_features_20260930.json
python -m src.research.birth_prediction.leakage plot --config configs/birth_prediction/leakage_features_20260930.json
```

`audit` reconstructs every past input from raw trajectories and checks full-sphere
PTM, source ancestry, matched metadata and descriptor replay. `explain` reads the
likelihood-selected CatBoost/linear artifacts, verifies prediction replay and
computes descriptor/temporal attribution plus matched-group permutations. All
eight fixed-test endpoints and full-history/one-frame CV fits are included.
`supplement` adds source-paired matched-ranking uncertainty and calibration spread
from recorded predictions.
`plot` renders existing numerical tables without refitting or recomputing metrics.

The completed bundle is `analyses/leakage-feature-audit-v1` in the extension run.
Its tables retain their exact explanation-source snapshot and metric definitions;
the later numerical-tolerance correction is captured separately in
`technical/audit-source`. Use a **new output analysis revision** in a copied recipe
when recomputing tables after an implementation change. Definitions are in
[birth prediction leakage metrics](metrics/birth_prediction_leakage.md).
