# Why the corrected VICReg curves are still poor

The corrected runs are not repeating the previous catastrophic saturation, but
their trained physical/TDA heads remain close to constant predictors within Al.
Useful structural information is still present in the encoders: fresh frozen
linear readouts recover it much better. Unrestricted BF16 additionally distorts
small differences between observations and worsens numerical rotation invariance.
These are early-training diagnostics, not completed 12-epoch results.

## Live training and what the curves mean

At the recorded curve capture, MACE had completed 270/2,930 updates (1.11 epoch
equivalents) and GATr 403/2,930 (1.65). Both were running. The scheduled warmup
ends at update 293. MACE's best selection score was 0.41790 at update 256;
GATr's was 0.40681 at update 128. The matching training-only Al mean scores
0.40472, lower being better. GATr's latest score was 0.40939 at update 384.

The score is source-balanced current physical error + 0.25 instantaneous-TDA
error, using 480 observations from fifteen selection sources and the original
train-only target normalization. It excludes VICReg. It is not test performance.

![Training and selection curves](plots/loss_and_selection.png)

The training curve alternates whole batches from different material/potential/
static groups. Those groups have very different target distributions. Some
spikes therefore reflect a change of task population, not an unstable optimizer.
Within native Al, losses have fallen. Unlike the earlier run, the audited losses
are finite and have not exploded into the hundreds or thousands.

However, fitting material means is a substantial shortcut for the physical
anchors: 41–71% of the variance in the four geometry blocks and 32–67% in the
three TDA blocks lies between material/potential/static groups. These fractions
come from the original training target population, with no selection data.
The geometry is cutoff-normalized at the input; reconstruction targets remain
in physical units with global training normalization. VICReg itself still uses
within-group batches, so between-material means cannot satisfy its variance floor.

## The encoders contain more information than the trained heads reveal

I captured fixed later checkpoints: MACE update 224 and GATr update 336. For
each, a ridge regression was fitted to **512 labelled native-Al training
anchors**, using train-only centering/scaling of the frozen embedding. The same
480 selection observations selected regularization from six declared strengths.
No encoder gradients or original checkpoint changes were made.

| Checkpoint | Trained heads, BF16 | Frozen ridge, BF16 features | Frozen ridge, FP32 features |
| --- | ---: | ---: | ---: |
| MACE, update 224 | 0.41374 | 0.23307 | **0.15879** |
| GATr, update 336 | 0.41773 | 0.29024 | **0.25234** |
| Full native-Al training mean | 0.40472 | 0.40472 | 0.40472 |

These selection results diagnose information retention and decoder accessibility;
they do not establish generalization on the held-out test sources. The 512-row
subset's own mean scores 0.40563. All probes use identical training/selection
indices and target definitions. FP32 and BF16 extraction use the **same weights**,
which were learned in BF16 mixed precision; this is not a retraining comparison.

Thus a poor trained-head curve is not evidence that the encoder has lost all
structural information. A train-centered linear readout succeeds where the
jointly trained heads do not. The current per-observation LayerNorm controls
amplitude but does not remove a large common feature component across observations
or standardize the small informative directions for the decoder.

## Precision and rotation invariance

The audit uses sixty fixed Al selection observations, four from each source.
It compares the same coordinates under an identity control, an exact 90-degree
axis rotation and three seeded proper random rotations. Rotations are computed
in float64 and inputs returned to float32; identities, masks, supports and MACE
edge connectivity stay fixed. The table reports the worst of the three random
rotations, measured as RMS change across exported coordinates and observations.
The exported state has RMS approximately one.

| Later checkpoint | FP32 rotation RMS | BF16 rotation RMS | BF16 with FP32 readout/heads | BF16 rotation error / between-observation RMS |
| --- | ---: | ---: | ---: | ---: |
| MACE, update 224 | 1.37e-7 | 2.05e-3 | 1.59e-4 | 0.236 |
| GATr, update 336 | 1.22e-7 | 2.47e-3 | 7.65e-4 | 1.024 |

The last column uses each checkpoint's **BF16** centered between-observation RMS
as denominator. Relative to its FP32 variation, the corresponding errors are
about 0.25 for MACE and 2.15 for GATr. Numerical invariance must therefore be
assessed against useful signal, not just the unit-sized latent offset.

Both architectures behave rotationally invariant to approximately FP32 rounding
accuracy in this audit. BF16 retains only approximate invariance, and its errors
are too large for these low-contrast states, particularly GATr. This does not
show a mathematical symmetry defect in GATr or MACE; it shows inadequate
finite-precision fidelity for the current learned representations.

The readout/head intervention changes computation precision only. For MACE it
reduces rotation error about 13x at update 224 and reduces centered FP32/BF16
state disagreement from 27% to 2.2% of the FP32 signal. For GATr it reduces
rotation error only about 3.2x; significant error already enters the backbone.
Earlier captured checkpoints (MACE 128, GATr 128/208) show the same qualitative
issue. The problem is not an artifact of selecting only an unusually weak late state.

On the later sixty-observation audit, FP32 physical predictions have mean
coordinate standard deviation 0.00277 for MACE and 0.000162 for GATr, versus
0.421 in the standardized physical targets. They are near-constant, though not
identical. The FP32 projector's variance is overwhelmingly concentrated in one
direction: participation ratios are approximately 1.00013 and 1.00001.
Participation ratio `(sum s^2)^2 / sum s^4` is a variance-concentration measure,
not exact algebraic rank; the ridge results demonstrate useful smaller directions.

## What the previous checks missed

The new normalization prevents the earlier large-offset/SiLU saturation failure,
but a nonzero spread above 1e-6 is not a sufficient information criterion.
BF16 rounding can inflate apparent spread and effective rank. The gradient-cache
tests verified that cached and full BF16 updates agree; they did not establish
that BF16 preserved rotational fidelity or fine structural information.

Nor should a small VICReg *loss value* be interpreted as weak gradient pressure.
On a fixed 64-pair native-Al diagnostic batch, the actual weighted VICReg gradient
with respect to exported states was roughly 5–6.5 times the physical gradient
in the early captured checkpoints. These are state gradients, not encoder
parameter gradients, and do not extrapolate to every batch. The result rules
out blindly increasing VICReg's weight as the obvious repair. Raw projector
variance and its total loss do not establish useful exported-state diversity.

## Recommended next comparison

1. Protect the scalar readout, physical/TDA heads and projector in FP32. For GATr,
   also protect multivector mixing, geometric contractions/products, normalization
   and geometric attention; retain BF16 only in invariant scalar operations first.
2. Validate FP32 versus mixed precision using rotations **and** frozen physical
   readouts. A small difference in total loss alone missed substantial readout
   degradation here. Specify absolute rotation tolerance and an error-to-signal
   limit on each material separately.
3. Run a short physical/TDA-only versus physical/TDA+VICReg comparison before
   another long queue. Use train-centered decoder inputs and monitor within-group
   prediction gains, projector spectrum and per-objective gradients. Keep data,
   seed and update budget matched. Do not change every objective coefficient at once.

The concrete architecture and efficiency proposal is in
[the mixed-precision proposal](../../../experiments/shared_pretraining_20260918/MIXED_PRECISION_PROPOSAL.md).

## Artifacts and scope

`technical/capture.json` identifies the initial immutable checkpoint copies.
`*-precision-audit.json` and `*-later-precision-audit.json` contain hashes,
indices, per-mode spreads, ranks, direct errors and state-gradient diagnostics.
`*-rotation-audit.json` and `*-later-rotation-audit.json` contain all rotation
matrices and errors, including identity/axis controls. `*-probe-audit.json`
records every ridge candidate; paired features/targets/predictions are in NPZ.
`target-audit.json` records group constants and variance decomposition.
`curve-summary.json` and `analysis-end-*` freeze the live curves used above.

Original training weights, source snapshots, recipes and W&B history were not
changed. The small read-only GPU audits temporarily shared the existing
allocations; throughput during them is not an independent performance benchmark.
Both scientific runs were still active at the recorded capture, with their
existing update-640 baseline gate. No new long training experiment was launched.
