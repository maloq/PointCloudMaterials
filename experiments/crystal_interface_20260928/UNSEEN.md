# Interface prediction without direct boundary visibility

**Superseded before training:** the clarified [liquid-only experiment](LIQUID_DISTANCE.md)
reuses this preparation but excludes all established crystal from observations
and keeps crystal-absent cells out of localization fitting/selection.

Question: what distance/direction information remains when no interface atom is
present in any of the 25 encoder patches? Keep the same spatial support and one
distance+direction+VCReg treatment, explicitly retained by the user.

Expand existing trajectories with 256 candidate centers on 64 frames per source
(2,457,600 candidates over the fixed 150 Al sources). Uniformly sample candidates
before checking visibility. Preserve all original benchmark rows, scans and source
roles. Train/validate only on contexts without a visible interface; retain candidate
denominators so selection does not overweight sources with few eligible centers.

One jointly trained encoder/context model, two GPUs, global batch 512 (256 per
GPU), differentiable global VCReg statistics. Train for 16 nominal blocks / 8192
replacement updates, selecting by validation predictive likelihood. Same original
snapshot parent, fresh head/optimizer; no time/temperature/species inputs. The
parent's historical visible-example exposure is recorded, so this is restricted
adaptation rather than training from an entirely visibility-naive initialization.

Evaluate distance likelihood, calibration, directional error, information readouts,
feature ranks, normalized noise response and 0.75-ps response. Scan alarms must
occur before the first visible interface observation, with no concatenation across
discarded positions. Report all-path recall and eligible observation opportunities.
Compare the frozen previous VCReg model on the same original invisible test rows;
the expanded uniform evaluation is an additional track, not a substituted benchmark.

The prior results review motivates a separate overfitting audit: complete selected-
checkpoint train/held-out gaps, learning curves, feature spectra and nearest-training
feature distances, source ancestry separation, and actual tensor inputs. This audit
creates no scientific training or online evaluation runs.

[Recipe](../../configs/crystal_vector/interface_unseen_20260928.json) ·
[Exact definitions](../../docs/metrics/crystal_interface_unseen.md) ·
[Execution](../../docs/crystal_interface_unseen.md).
