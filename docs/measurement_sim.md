# Ni trajectory prediction directly in fixed measurements

The October 11, 2026 workflow removes the eSEN encoder, the learned 16-dimensional
state compression and the learned observation decoder. A stochastic autoregressive
transformer evolves all 244 observed measurements directly at 5 fs per step.
Four native observations at -0.3, -0.2, -0.1 and 0 ps initialize it. Every predicted
path starts exactly at the last observation. Ordered memory updates every 0.1 ps;
within an interval the current state advances and the preceding three observations
remain fixed. No temperature, source age or absolute time enters the network.

The state is BO40 + TDA113 + smooth density74 + velocity17. The original BO/TDA
target columns are reused. CNA is omitted. These are direct observations of the
original, unrelaxed Ni positions and velocities. Transformer width64 is an internal
computation width, not a compressed state that must be reconstructed by a decoder.
The state update uses a train-fitted diagonal OU prior with learned drift/diffusion
corrections. It is a stochastic predictor in measurement space, not an atomic
coordinate simulator or a proven closed Markov representation.

## Scientific comparison

Six fits cross three objectives with two initializations (20261015/20261016):

| Treatment | Marginal likelihood | Connected path/increments/lag products | Physical response |
|---|---|---|---|
| marginal | yes | evaluation only | evaluation only |
| connected | yes | yes | evaluation only |
| connected_response | yes | yes | yes |

All fits also match physical short-path mean values. Each fit has 1,600 updates,
batch256, four neural paths per example, AdamW maximum learning rate0.001,
100-update warmup and cosine decay. Each batch draws a 1/3/6-ps unroll with
probabilities 0.5/0.25/0.25. Neural matrix work on long rollouts uses BF16;
states and losses remain float32, and short response secants use float32.

The pinned cohort contains 14,400 sequences from 60 existing Ni NPT ancestors:
4,800 original, 4,800 liquid-persistent and 4,800 growing-front sequences.
One central atom ID remains fixed over each sequence. Train/selection/test roles
are unchanged. No broad spatial-pretraining rows or new MD are introduced.
The liquid/front enrichment used future outcomes, and is explicitly not a
representative sample for estimating unconditional crystallization probability.

The physical derivative bank retains 300 full-cell anchors with two short shots
each, sampled every5fs through100fs. Only 91 smooth measurement channels receive
analytic output-response supervision. Full244 input secants include actual
changes in BO/TDA. Step-halving diagnostics expose input nonsmoothness. The fixed-box
BAOAB teacher and its fixed80 identities remain separate from long native NPT
observations that reselect nearest80 neighbors around the same central atom.

Held-out evaluation uses32 neural particles, complete paths and 1/3/6-ps endpoints,
OU/mean/persistence baselines, source-bootstrap intervals and matched-seed response
contrasts. Plots show12 predetermined held-out examples (four per regime), individual
paths, ensemble mean,90% intervals and actual MD measurements. No decoder noise is
added to those plots. Measurement values need not correspond to a realizable atomic
configuration; predictive scores test their behavior, not coordinate reconstruction.
See [exact metric definitions](metrics/measurement_sim.md).

## Operations and reproduction

Recipe: `configs/ni_latent_sim/direct_measurements_20261011.json`.
Dataset registry entry: `ni-direct-measurements-20261011`.
Result root: `${storage:training_storage}/measurement_sim/direct-20261011-v1`.
Disposable data: `${storage:cache}/measurement-sim/direct-20261011-v1`.
All model checkpoints, metrics, plot inputs and provenance remain in the result root.

Preparation uses the existing native source reader and original descriptor producer:

```bash
python -m src.research.measurement_sim.data prepare --config configs/ni_latent_sim/direct_measurements_20261011.json
python -m src.research.measurement_sim.data native --config configs/ni_latent_sim/direct_measurements_20261011.json --workers 12
python -m src.research.measurement_sim.data physical --config configs/ni_latent_sim/direct_measurements_20261011.json
python -m src.research.measurement_sim.data gpu --config configs/ni_latent_sim/direct_measurements_20261011.json
python -m src.research.measurement_sim.workflow launch --config configs/ni_latent_sim/direct_measurements_20261011.json
```

Use conda `pointnet-torch214`. Scientific fits require online W&B in
`teshbek/PointCloudMaterials`, with stable per-fit IDs. The launcher freezes source,
configuration and metric hashes, then starts two detached workers. The first pair
is connected_response/connected at the same seed, followed by the marginal control
and the second seed. Per-fit file locks prevent duplicate execution. Both fitting
and evaluation save resumable progress; a failed stage gets one retry with the
same settings before the worker moves to the next experiment. Errors are preserved,
not converted into alternate objectives, offline tracking or successful receipts.

If allocation time expires with unfinished work, one dependent two-GPU allocation
is requested from the alternative RTX6000PRO/A100/A40/L40S partitions. Existing jobs
are never cancelled or modified. Queue receipts identify all processes started by
this workflow. Training/evaluation/comparison proceed without another chat wake-up.
There are no scheduled chat tasks in this workflow.

## Overnight memory and matched response study

The October12 recipe is `configs/ni_latent_sim/measurements_overnight_20261012.json`.
Its fifteen scientific fits cross three paired seeds (20261017–19) with ordered
4-frame history, ordered14-frame history, past13 mean/std plus present, physical
values alone, and physical values plus derivatives. The native forecast origin
moves +1 ps so all memory treatments have identical5-ps targets. Native fits use
2,400 updates; separate100-fs physical fits use4,000. All use direct244 measurements
and the existing simulator archive. This is a history/exposure/replication expansion,
not an expansion beyond60 independent MD ancestors.

The physical comparison starts from the current measurement only and matches the
teacher's fixed80 identities and fixed-box BAOAB continuation. Two actual physical
shots are scored individually; analytic mean response supervises91 smooth channels.
Native NPT fits receive no physical auxiliary loss in this queue. The CPU queue
runs78 frozen response transport/residual readouts with source-count and within-source
size curves, and three ridge history readouts. It creates no W&B runs; fifteen
scientific fits remain online with stable resumable IDs. See the
[scientific plan](encoder_research/experiment_plan.md#ni-followup-20261012) and
[distinct metric contract](metrics/measurement_followup.md).

```bash
# Requires two allocated GPUs; submits continuation and local diagnostic jobs.
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.measurement_sim.campaign launch --config configs/ni_latent_sim/measurements_overnight_20261012.json
```

Results: `${storage:training_storage}/measurement_sim/overnight-20261012-v1`.
Source measurements are reused by pinned checksum; no duplicate cache is made.
`technical/submission.json` records the CPU job and dependent continuation;
`technical/launch-<allocation>.json` records detached GPU processes. Both training
and evaluation checkpoint; failures stay visible and do not block unrelated fits.
The continuation is submitted at launch, rather than waiting for a chat or timer.
Per-allocation execution segments preserve actual precision, microbatch and resume
update. Figures show predetermined real held-out paths and neural samples, with
metric tables and paired contrasts produced automatically. The complete workflow
runs without a scheduled chat review.

## Density rates, source expansion and smaller steps — October 13

Recipe: `configs/ni_latent_sim/density_rates_20261013.json`; implementation:
`src.research.measurement_sim.rate_campaign`. This is a separate physical100-fs
study, with the same244 future measurements and an optional74-channel initial
condition D_x density[V]. The initial conditioning token persists throughout
the neural trajectory. Both conditions have the same network capacity, and
both recompute their complete inputs under Q/V perturbations. The smaller-step
comparison uses1-fs versus5-fs neural transitions, emitting common5-fs targets.
The physical simulator retains1-fs integration, float64 states/tangents/noise
and float32 FastNi force arithmetic. No relaxed inputs or structural replay.

Data-size banks contain80 states from16 original training sources,320 states
from those same16, and320 states from28 sources after adding12 independent
training sources. The320-state banks match temperature counts exactly, without
feeding temperature to the model. Source/frame/center identities, all original
roles and cached physical checksums remain in the plan. The62 nontraining
sources contribute310 test anchors: all former selection sources are included.
There is no inner split, test-based tuning, checkpoint selection or early stop.
The checkpoint rule is the final1,600th update; local readouts have fixed
hyperparameters. The new train/test rule is the project default in AGENTS.md.

There are36 scientific fits:24 on the diverse320 bank (rates × derivatives ×
5/1-fs step × three seeds), plus12 values/derivative controls on the two smaller
diversity banks at5 fs without rates. Every fit uses batch64/microbatch64 and
four training particles; this explicitly retains the previous physical-task
batch rather than the spatial-encoder default. Evaluation uses32 particles and
three noise keys, pooled test scores plus old/new cohort panels. Eighteen fixed
response readouts and per-shot noise diagnostics remain local. The
[metric contract](metrics/measurement_rates.md) specifies all losses and scores.

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m src.research.measurement_sim.rate_campaign launch --config configs/ni_latent_sim/density_rates_20261013.json
```

Launch requires two allocated GPUs. One worker prioritizes teacher acquisition,
while the other starts training when its bank is sealed; both train once
preparation finishes. Existing teachers are remeasured/reused, new teachers
publish from SCRATCH to STORE, and measurements live on IDS. Per-stage locks,
bounded retries, failure receipts and resumable checkpoints are independent of
chat activity. Two16-hour two-GPU continuation reservations are submitted at
launch from a combined alternative partition list. They resume only unfinished
work and exit when the queue is complete. No unrelated jobs are modified.

Completed results: [density-rates-20261013-v1](../output/measurement_sim/density-rates-20261013-v1/README.md).
All 36 fits and evaluations, 18 readouts and 751 anchor packets finished in about
6.01 hours. All fits reached the fixed 1,600-update checkpoint. Execution receipts
and `technical/completion-review.json` record completion; the
[evidence register](encoder_research/evidence.md#density-rates-20261013) summarizes
the scientific findings and limitations.
