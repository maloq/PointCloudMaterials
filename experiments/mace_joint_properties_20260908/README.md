# Improve MACE properties together — 2026-09-08

> Completed experiment record — implementation retired on 2026-09-09.
> Commands, plans, and implementation paths below describe the original run,
> not the current supported trainer. Reproduction of that protocol requires
> the run's `tracking/*/source.tar.gz` and recorded configs under `output/`.
> Existing results/checkpoints are retained. Use the
> [current 80-atom recipe](../mace_plain80_20260909/README.md) for new training.

Question: can a single exported 256D small-MACE embedding improve topology,
spatial coherence, temporal stability and physical forecasting together?
This follows the [topology-run review](../../output/mace_topology_nuances_20260907/REVIEW_20260908.md).
No teacher, new simulation or temporal-memory architecture is introduced here.

The [September 8 interim review](../../output/mace_joint_properties_20260908/INTERIM_REVIEW_20260908.md)
reports a completed control with modest gains and a newly verified mismatch:
the 6.5 Å compact encoder excludes some atoms used by the 65-atom TDA target.
The queue continues with the originally specified training settings.

## Controlled queue

| Run | Added objectives relative to control |
| --- | --- |
| `control` | Original spatial/temporal VICReg, instantaneous TDA and future-latent prediction |
| `nuisance_future` | Same-neighborhood perturbation invariance and future physical TDA |
| `joint_ranking` | Above, plus topology ordering on same-material density-matched triplets |
| `joint_pcgrad` | Same objectives, with PCGrad on actual shared-encoder gradients every 20 steps |

Every run starts from the same completed LR5 all-objective checkpoint with a
fresh optimizer, identical train-only scalers and identical data/validation seed.
Six full epochs mean **4,224 updates and 6,488,064 anchor exposures per run**;
early stopping is disabled for this matched budget. Batch **1,536 quadruplets**
contains 6,144 physical views. The three nuisance trials add 1,536 perturbed
anchor views per batch; these are augmentations, not additional unique states.
Each physical neighborhood contains **80 atoms**. Encoder/head peak LRs remain
**3e-4 / 3e-3**, with one epoch warmup followed by per-optimizer-step cosine decay
to 1e-6. Online W&B is enabled separately for all four runs.

The existing Al/Mg/Ta cache contains **286,720 training quadruplets**, **993,768
distinct neighborhood states**, and 1,146,880 physical view slots:

| Material | Quadruplets | Distinct neighborhood states |
| --- | ---: | ---: |
| Al | 245,760 | 847,096 |
| Mg | 32,768 | 117,338 |
| Ta | 8,192 | 29,334 |

Material-balanced batches oversample smaller pools. Temporal VICReg uses only
verified 0.1 ps pairs (1,280 per batch); Al shooting pairs retain spatial/TDA and
longer-horizon forecasting supervision. The cached 512-atom patches are sliced
to 80 atoms before model input; there are no hidden extra input atoms.

## Objective details

Original attraction strengths and VICReg variance/covariance terms are retained;
the median-calibrated topology gates from the previous trial are removed.
Nuisance perturbations add independent Gaussian displacement, sigma 0.005 Å,
with the central atom fixed. Their latent MSE weight is 1,000; the initial real
batch contributes about 0.083 to the weighted loss. This specifies a geometric
tolerance, not an empirically established thermal-noise model.

Physical forecasting decodes the existing predicted future embedding through the
same TDA decoder, against fixed future TDA targets, with weight 5. It trains the
encoder, existing forecast head and TDA head together. Instantaneous TDA targets
and their PCA/whitening remain unchanged. The original latent forecast remains.

Topology triplets share an anchor and material, with both alternative densities
within 3%. Target distances use the previous training-audit reliability weights.
Their difference must exceed **0.348335**, four times the square root of the
previous training-only mean squared perturbation error. This is a conservative
heuristic gap, not a confidence interval or physical noise floor. Latent distances
are normalized by differentiable within-material variance; the hinge margin is
0.1 and loss weight 5. The preflight found 440 eligible triplets in a real batch.

Task groups are spatial (including nuisance and future spread), temporal,
topology (TDA plus ordering), and prediction (latent plus physical). All runs
log actual encoder-gradient norms/cosines every 100 steps. `joint_pcgrad` also
projects conflicting encoder gradients every 20 steps; heads always receive
their ordinary summed gradients. This is periodic PCGrad, not PCGrad at every
update. It preserves the large-batch covariance calculation. See the
[original PCGrad paper](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Evaluation and selection

Selection compares all three materials' TDA-head MSE, physical forecast MSE,
spatial/shuffled and temporal/shuffled latent ratios against initialization.
No-regression candidates take priority; within a category, selection maximizes
the worst relative gain, then the mean gain. Effective rank must retain at least
90% of its initial value. If none passes, the nearest candidate is preserved
for analysis with an explicit failed screen; it is not called a successful model.
The legacy export filename remains `best.pt` and the selection details are saved.

After each run, common frozen probes fit 6,144 training anchors and evaluate 768
validation anchors. Additional analysis reports density-matched topology ranking
and absolute errors against a constant baseline, perturbation sensitivity,
physical forecasting by horizon, and paired source-block bootstrap intervals.
The final comparison requires observed gains in every material and every static
frame; passing a validation screen alone is insufficient.

All six static Al frames (772,953 centers) run through the existing full
encoder-only pipeline, using configs derived from the established static recipe.
The archived spatial comparison uses 684,723 shared centers. Old reference
inference caches were deliberately retired; their retained coordinates and
scores are reused only after exact grid/order and shared-count verification.
The new runs keep their own inference caches. Seven imposed clusters remain a
visualization, not phase truth. No PTM labels supervise training.

Limitations: static Al includes ancestors of training continuations. Validation
has seven Al groups overall, but only one Al source with 0.1 ps temporal pairs;
Mg and Ta each have one validation source. Source-bootstrap temporal uncertainty
is therefore not estimable for any material. This is exploratory controlled screening,
not an independent demonstration of liquid-precursor discovery. A history model
is deferred until these single-frame experiments are analyzed.

## Reproduction and execution

[Queue plan](plan.json) and per-run JSON/YAML files specify scientific settings.

```bash
conda run -n pointnet python -m pytest -q tests/test_mace_joint_objective.py
conda run -n pointnet python -m src.analysis.mace_joint \
  --plan experiments/mace_joint_properties_20260908/plan.json --stage preflight
conda run -n pointnet python scripts/experiment_registry.py run \
  --spec experiments/mace_joint_properties_20260908/run_spec.json
```

The queue includes preflight, then train → frozen probes → full static analysis
for each run. It stops loudly on a failed command or incomplete matched training.
It uses the existing allocation **984861 on node53** and reserves 30 minutes
before its September 9, 05:55 Paris expiry. The controller is launched detached
with stdin disconnected; launch identity and live status are in the output.
Do not repeat a fresh launch into an existing training output.

Preflight compares full backpropagation with both cached implementations on real
quadruplets, checks projection/ordering/selection contracts, exercises all four
objectives at full batch and peak LR, audits both existing encoders under the same
training-only perturbations, and evaluates the common before-model. Peak test
memory was approximately 54 GiB on the 96 GB H100. One-step peak-LR tests are finite;
they do not establish long-run stability. Warmup remains enabled in training.

## Outputs and file roles

Results stay under `output/mace_joint_properties_20260908/`: `status.json`,
`RESULTS.md`, `joint_comparison.csv`, individual `runs/` and standard static plots.
Selected model weights and analysis data remain in the repository; replaceable
optimizer checkpoints live under `/tmp/vmorozov_mace_joint_properties_20260908/`.
Home usage before launch was about 75 GiB against the user's 100 GiB quota.

This directory contains experiment records and configuration. Maintained shared
implementation is in `src/training_methods/mace_joint_objective.py` and
`src/analysis/mace_joint.py`, with extensions to the existing MACE trainer,
queue and analysis. Numerical contracts are in `tests/test_mace_joint_objective.py`.
Logs, launch files, metrics and preflight artifacts are generated run outputs.
Execution tracking retains source snapshots, configs and environment provenance.

## Receptive-field audit

`receptive_field_audit.py` is experiment-specific reproducibility code. It samples
the actual training/validation quadruplets, compares TDA target extent with the
adapter's geometric support, and constructs excluded-atom perturbations whose
retained encoder graph/geometry are exactly unchanged but whose TDA changes.
It runs on CPU and does not alter training. Results and interpretation are in the
interim review linked above.

```bash
PYTHONPATH=. conda run -n pointnet python \
  experiments/mace_joint_properties_20260908/receptive_field_audit.py \
  --config experiments/mace_joint_properties_20260908/control.json \
  --output output/mace_joint_properties_20260908/architecture_review_20260908 \
  --anchors-per-material 2048 --perturb-examples-per-material 16
```
