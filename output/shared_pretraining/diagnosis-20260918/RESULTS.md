# Why the September 18 pretraining campaign failed

The local batch-512 campaign suffered optimization failure: excessive update
sizes, saturated decoder/projector activations and representation collapse.
The causal GATr stage also exposed an unsafe interface between an unbounded
pretrained embedding and a newly initialized prediction head. The loss curves
are not just a W&B plotting problem. This audit uses local saved artifacts;
it does not establish what happened on an independently operated H200.

MACE–VICReg and GATr–VICReg had each completed 5,860 structural updates and 720
causal updates before the stop, and their frozen analyses had completed. JEPA
was checkpointed at 3,495/5,860 updates, about 7.16 epoch equivalents. A status
of `complete` meant the requested update budget finished; it did not certify
representation quality. No stopped campaign was resumed by this analysis.

## Direct evidence

All selection scores below use the same broad-release normalization and
source-balanced **physical + 0.25 instantaneous-TDA error**, lower is better.
The definitions are preserved in the [original metric contract](../../../../docs/metrics/shared_pretraining.md).

| Structural model | Selected update | Best selection score | Last selection score | Checkpoint inspection |
| --- | ---: | ---: | ---: | --- |
| MACE–VICReg | 128 | 0.5111 | 0.6668 | Final physical decoder nearly constant, while projector still varies |
| GATr–VICReg | 512 | 0.5350 | 0.6674 | Final encoder constant on all 60 inspected selection observations |
| GATr–JEPA/SIGReg | 128 | 0.4906 | 0.7118 | Final projector and physical/TDA decoders constant despite very large, varying encoder outputs |
| Al/MEAM training-mean predictor | — | **0.4047** | — | No geometry-dependent learned representation |

The baseline was fitted only to the native-potential Al training observations
and evaluated on the same fifteen selection sources. Even the selected trained
heads failed this cheap information test. For context, the earlier lower-rate,
batch-128 MACE pilot reached **0.2585**. Its GATr counterpart reached **0.4045**,
almost the mean baseline: GATr already had a weak-information problem before
the larger campaign. Those pilots change both rate and batch, so they alone do
not isolate the effect of learning rate.

![Training traces and collapse diagnostics](plots/training_failure.png)

## Learning rate and saturated activations

AdamW peaked at **0.02 for every encoder and head parameter**, compared with
0.0003 in the pilot: a 66.7-fold increase, alongside a fourfold batch increase.
Warmup reached this rate at update 586; serious spikes started much earlier:

- GATr–VICReg: update 60, LR 0.00205, loss **77.2**; maximum loss **247.1**.
- MACE–VICReg: update 175, LR 0.00597, loss **30.1**.
- JEPA: update 69, LR 0.00235, loss **82.7**; later loss **9,432** and an
  unclipped gradient norm of **5.62 million**.

Gradient clipping at 5 was active. It did not prevent large AdamW parameter
updates or subsequent saturation. The inspection used fixed first-four
selection windows from each of fifteen sources, sixty observations total,
with microbatch 8 and the original saved model weights.

A **paired 128-update replay** starts at the same GATr–VICReg update-512
checkpoint, retaining optimizer moments, full batch 512, sample identities and
the same H100. Only the learning-rate schedule amplitude changes:

| Replay, updates 513–640 | Selection score at 640 | Mean coordinate standard deviation, float64 |
| --- | ---: | ---: |
| Original peak 0.02 | **1.3148** | **0.00000** |
| Schedule scaled to peak 0.0003 | **0.5396** | **0.01905** |

The original-rate branch is already exactly constant by update 576; the
lower-rate branch preserves variation throughout. Both start with score 0.5350
and standard deviation 0.01832. The original campaign used RTX6000 for this
interval, so the H100 replay is not a bitwise reconstruction of its trajectory.
Within this paired intervention, the rate change prevents the observed
collapse. This supports a causal role for excessive updates. It does **not**
prove that lower LR alone repairs already saturated heads or makes a fresh
GATr run informative: the lower-rate score did not improve its starting point.
Exact batches and results are in `technical/lr-replay.json`.

![Paired learning-rate intervention](plots/paired_lr_replay.png)

At GATr's selected structural checkpoint, the exported state's mean absolute
coordinate was about **146**. **All** inspected physical and topology head
preactivations were below −30, putting their SiLU activations deep into the
near-zero tail. Their predictions were exactly constant on these observations.
At JEPA's final checkpoint, the exported state had mean absolute magnitude
about **27,905**, but every inspected physical, topology and projector
preactivation was below −30. All three output heads were exactly constant.
MACE's final physical preactivations were below −30 about 77% of the time, and
its physical predictions varied only at a negligible level.

This explains how a representation penalty can coexist with failed physical
decoding: it acts on a separate [projector](../../../../docs/research_glossary.md#projector),
and large latent offsets/weights can saturate the physical decoders. A nonzero
encoder variance alone is not evidence that its physical heads still work.
The final GATr–VICReg **encoder readout** is saturated too: all its inspected
preactivations are below −30, in both the structural and causal checkpoints.
Its last linear layer consequently exports its constant bias. This locates
the complete loss of exported variation inside the encoder itself.

## Why the auxiliary losses did not rescue learning

The implemented [VICReg](../../../../docs/research_glossary.md#vicreg-in-the-earlier-encoder-protocols)
contribution is `0.1 × (25 I + 25 V + C) / 51`. At exactly constant paired
projector outputs, invariance and covariance are zero and the variance penalty
is 0.99. The entire weighted penalty is only **0.04853**. At exact collapse its
gradient is also zero; it is not a mechanism for reviving a saturated network.
This value appears directly in the failed GATr traces.

For [temporal JEPA/SIGReg](../../../../docs/research_glossary.md#lejepa-and-sigreg-in-the-structural-pretraining-proposal),
near-zero next-embedding error can mean both sides became constant. The pinned
upstream SIGReg implementation returns **205.848** for zero batch-512 projector
outputs, matching JEPA's observed long plateau; the weighted loss contribution
is **1.02924**. A fixed-seed Gaussian reference gives about **1.09 raw**, instead
of 205.85. A high flat penalty therefore was evidence of a failed regularizer,
not convergence to a good representation. The exact-zero gradient check is
also zero for this penalty.

The upstream Epps–Pulley statistic includes sample count. Its constant-zero
value is 51.462 at batch 128 and 411.697 at batch 1,024. Retaining the same
coefficient across batches changes its strength for a fixed non-Gaussian
distribution. The formula was implemented as intended, but its batch dependence
needed a controlled calibration before the H200 expansion.

## Causal initialization damaged an already weak parent

The new future MLP consumed the raw exported state without input normalization
or a bounded initial output. Reconstructing the **actual first causal batch**
from saved structural states and the same newly initialized head gives:

| Future-head input/output at initialization | GATr future loss | MACE future loss |
| --- | ---: | ---: |
| Original implementation | **2422.130** | 0.9954 |
| Subtract the training-only latent mean before that same head | **0.9945** | 0.9947 |
| Predict normalized zero initially | 0.9926 | 0.9926 |

The recorded GATr first-step future loss is 2422.130, reproducing the failure.
The centering comparison changes neither targets nor head weights; it isolates
the effect of the large latent offset. It is an initialization counterfactual,
not a validated repair or a new trained model. Starting joint adaptation with
this large new-head loss and warming the entire encoder back to 0.02 was unsafe.

The selected causal GATr checkpoint maps **all 38,400 native observations to one
identical exported vector**, verified directly from saved extracted states.
Its frozen ridge predictions equal the training-mean baseline. This is full
representation collapse, not just a poorly tuned decoder.

## What still contains information

Fresh, frozen ridge readouts can recover some information from the earlier
selected structural checkpoints, despite their failed trained heads. The table
uses the separate native forecast normalization and all thirty held-out test
sources, so its numbers must not be compared numerically with selection scores.

| Selected checkpoint | 9 ps physical MSE | 9 ps TDA MSE |
| --- | ---: | ---: |
| MACE structural | **0.6551** | **0.2294** |
| MACE causal | 0.6707 | 0.2447 |
| GATr structural | 0.7187 | 0.2786 |
| GATr causal | 0.9369 | 0.4683 |
| Training mean | 0.9369 | 0.4683 |
| Persistence | 1.0877 | 0.3459 |

These point estimates justify preserving the early checkpoints. They do not
establish a successful shared-pretraining recipe or a benefit from causal
adaptation. One seed does not quantify training-seed uncertainty.

![Information retained by selected checkpoints](plots/saved_checkpoint_information.png)

## Monitoring and validation gaps

The code checked nonfinite losses and gradients, but a dead network can remain
finite indefinitely. There was no requirement to beat a train-only mean, retain
nonconstant decoder outputs or maintain useful exported-state variation before
continuing the queue. Two-update preflight checks and gradient-equivalence tests
could not validate learning through the warmup. I should have required these
learning checks before allowing the full budgets and dependent causal jobs.

There is also a numerical diagnostics issue: float32 NumPy variance can report
nonzero variation for exactly identical rows. The constant causal GATr state
has a float32 mean-coordinate standard deviation of about **1.33e−6** over the
saved 38,400 rows, but float64 standard deviation and coordinate ranges are
**exactly zero**. The nonlinear-probe normalizer also used float32 reductions.
This does not cause the original training collapse, but it can obscure it and
requires correction. For example, the large-offset structural GATr state's
float32 standard deviation is 0.0356 versus 0.0189 with float64 reductions.

Raw training loss additionally mixes different material/potential groups, and
causal replay steps omit future supervision. This explains some curve noise
and alternating levels, not the spikes, saturated heads and lost information.
Per-group curves and per-objective gradients are needed alongside total loss.

## Before another large queue

1. Use a short matched learning-rate check at the real statistical batch, starting
   near the earlier 0.0003 scale, with lower rates for pretrained encoder
   adaptation than for newly initialized heads. Verify learning through warmup.
   Lowering the rate on an already dead checkpoint is not a sufficient repair.
2. Stabilize the exported-state/head interface and the new future head's initial
   output. The normalization must be fitted only on training observations.
   Monitor activation saturation and actual decoder variation.
3. Calibrate physical/TDA versus representation gradients, including SIGReg's
   batch dependence. Check both exported z and projector q; retain the same
   data/splits while diagnosing these implementation choices.
4. Require held-out selection improvement over the training-mean baseline and
   noncollapsed states/heads before launching causal training or long continuations.
   Use stable float64 diagnostic reductions and checkpoint readout tests.

The original campaign and H200 batch-1,024 recipe should remain paused pending
these checks. This audit did not change production losses, data, checkpoints or
learning rates and did not submit another scientific campaign.

## Audit artifacts and scope

`technical/checkpoints.py`, `anchors.py`, `regularizers.py`, `lr_replay.py` and `lossplots.py` are
disposable reproduction diagnostics for this report. Their JSON outputs retain
checkpoint/data identities, fixed observation indices and calculation details.
`checkpoint-audit.json` records float64 variance, coordinate ranges, unique-row
counts and activation thresholds; participation rank is `(sum s²)² / sum s⁴`
of centered feature singular values and is only a descriptive spectrum statistic.
`anchor-audit.json` verifies the training target normalizers against their actual
producer (maximum differences below 1.5e−14) and reconstructs the causal first
batch. `regularizer-audit.json` records the constant/Gaussian reference checks.
`loss-summary.json` hashes the stopped training traces and saved evaluations.

Thirteen existing tests passed on CPU, including structural/causal cached versus
full-batch gradients, uneven chunks, checkpoint continuation and source splitting.
These tests and normalizer reconstruction found no evidence that gradient caching
or corrupted target normalization caused this failure. They do not certify all
possible GPU numerical behavior; the paired replay uses the same H100 for both
branches. Original stopped artifacts and frozen source snapshots are preserved.
