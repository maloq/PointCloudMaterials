# Supervised onset: maximize AP at 3 and 6 ps

Question: can direct crystallization supervision, relaxed observations and a
ranking objective give a more predictive exported state on the larger existing
Al cohort? This is the `crystallization_supervised` branch. The separate
[self-supervised question](../../docs/encoder_research/training_branches.md) is
reserved for a later experiment.

The preceding small screen had only three positive development windows at 3 ps.
Relaxed inputs and fresh readouts were promising, while tensor pooling and wide
context did not improve the short-horizon result consistently. Therefore this
study increases event coverage, trains the actual state for onset, and tests
paired relaxation information before spending more budget on model width.

## Fixed scientific protocol

Reuse the completed `relaxed-encoder-large-test-20260921` paired assay. There are
150 distinct independent melt lineages, split into 90 training, 15 selection,
15 calibration and 30 test sources. All 31,609 rows are causally eligible,
currently at-risk local observations. No new simulation or quench is required.

| Role | Windows | Events by 3 ps | Events by 6 ps |
| --- | ---: | ---: | ---: |
| Train | 10,825 | 79 | 188 |
| Selection | 5,306 | 39 | 79 |
| Calibration | 4,222 | 35 | 85 |
| Historical test | 11,256 | 53 | 135 |

The label is first sustained **original-MD local** crystalline onset, with the
existing three-frame confirmation and eligibility rules. Relaxed crystallinity
is never substituted for the future label. A relaxed view is obtained from the
current observation, without future coordinates. The held-out assay samples
12 ps-spaced anchors but its labels resolve the original **0.75 ps** cadence;
3 and 6 ps are forecast horizons, not input sampling intervals.

Test sources were used in previous studies. They are held out from fitting,
selection and calibration in this run, but are **not a new untouched test**.
No older pretrained encoder is imported: all six fits start from the same seed
and scratch spatial weights, avoiding pretrained ancestry leakage.

## Six supervised encoders

| Arm | Input at inference | Training objective / question |
| --- | --- | --- |
| O-NLL | Observed geometry | Source-balanced first-event hazard likelihood baseline |
| R-NLL | Relaxed geometry | Does relaxation help the directly supervised baseline? |
| O-AP36 | Observed geometry | NLL plus full-population smooth AP, weighted 2:1 for 3/6 ps |
| R-AP36 | Relaxed geometry | Same ranking objective with the strongest prior input candidate |
| OR-AP36 | Observed **and** relaxed geometry | Shared MACE processes paired views; a trainable fusion of hot, cold and their difference yields one state |
| O-distilled | Observed geometry | Same ranking objective plus fitted-state and hazard distillation from the frozen screen R-AP36 teacher |

Every encoder is trained end to end with original-MD onset supervision. Native
MACE uses two spatial blocks, width32, cuEquivariance and float32. Geometry is
converted explicitly from the producer's normalized units back to Angstroms,
then edges are rebuilt at 5 A. The input remains the existing nearest-80 paired
observation cropped at 8 A; it is not a new complete-radius or halo observation.
Scalar pooling and a learned residual produce 128 channels. The paired arm
shares spatial weights and compresses both views into one 128-dimensional state.
Hazard readouts take the exported state and the known temperature and elapsed simulation-age conditions.

No information bottleneck or direct smoothness penalty is applied. Unlike the
previous robust-onset screen, there is no physical-reconstruction gate restricting
which checkpoint can win. This isolates the attainable onset skill under the
supervised task; it does not assert that other physical information is retained.

Training samples are a 50:50 mixture of the source-balanced natural population
and its by-6-ps event subset. Exact `p/q` weights correct the hazard and teacher
losses to the natural source-balanced objective. AP updates use **all 10,825
training rows** with their natural source weights; they do not use enriched-batch
precision. Every 32 optimizer steps, compute smooth AP on cumulative-risk log
odds, with temperature0.5, weights2/3 and1/3, coefficient2, and replay exact latent
gradients through encoder microbatches. This is a source-weighted Smooth-AP
adaptation, not the exact nondifferentiable AP metric or an AUROC loss. The
implementation reuses the tested [earlier definition](../../docs/metrics/robust_onset.md).

The teacher is the screen R-AP36 checkpoint selected on selection AP3. Freeze it
before student training, standardize its state on training sources, and provide
targets only for training rows. The student uses a projection loss and soft
conditional-hazard BCE, each within a combined coefficient0.2. It does not need
a relaxation at inference.

## Selection and comparison

One seed per arm. Select checkpoint by **selection AP3**, breaking exact ties by
AP6; also preserve a separately labeled AP6-selected checkpoint. Six screens get
up to8,192 updates, with a42-minute per-arm safety cap. Promote the two best
screen AP3 models and split remaining training time between them, continuing
their optimizer and random state, up to200,000 total updates each. Learning rate
warmup128; inverse-square-root decay beyond8,192 updates. Report actual update
counts and cap hits, so time-limited fits are not mistaken for matched-update fits.

Fresh frozen linear and128-wide nonlinear readouts are fitted to each AP3-selected
encoder, with their own selection AP3 checkpoint. Fixed observed/relaxed/paired
descriptor and temperature-plus-age controls receive the same source split. A
two-finalist risk mixture searches11 weights on selection sources only; report
it separately as a predictive ensemble, not a new encoder.

Calibration uses a shared increasing affine transformation of cumulative-risk
log odds, fitted only on calibration AP3/AP6 labels. This preserves cumulative
order. AP always uses the original uncalibrated scores. Alarm thresholds are
chosen on calibration negatives for source-weighted FPR<=5%; report actual test
FPR, recall, Brier and AP. Source-bootstrap intervals quantify source uncertainty,
not training-seed uncertainty. Save all rows, predictions, source IDs and hashes.

State spectrum and normalized input-noise response accompany AP3 checkpoints.
Noise RMS is measured relative to each clean patch's mean twelve-neighbor
distance. The paired arm perturbs each supplied view separately. No re-quench is
performed after perturbation. The paired held-out cache has no0.75 ps differences,
so its movement rank/stability is explicitly unavailable; do not substitute12 ps
or interpolate. Dense trajectory diagnostics can be a separate follow-up.

## Reproduction and outputs

Recipe: [ap36_20260924.json](../../configs/supervised_onset/ap36_20260924.json).
Implementation: `src/research/supervised_onset/`.
Commands and detached status: [operations](../../docs/supervised_onset.md).
Results: `output/encoder_supervised/ap36-large-20260924/`;
`tables/comparison.csv` is the eventual comprehensive table, with
`tables/METRICS.md` and exact implementation hashes.

Launched 24 September2026 at23:33 Europe/Paris as detached Slurm step
`1007275.3` on node58. Budget deadline:25 September at08:03 Europe/Paris.
Fourteen tests, every-arm production GPU updates and a complete disposable
end-to-end pipeline passed. The queue completed at05:41 Europe/Paris on25 September; see [results](RESULTS.md).

Input audit on25 September verified all seven condition columns: five temperature indicators, age_ps/600 and its square. See the [context correction](../../output/encoder_supervised/ap36-large-20260924/tables/CONTEXT_ERRATUM_20260925.md). The prior description omitted the age inputs; numerical results are unchanged.
