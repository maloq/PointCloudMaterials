# Predictive information for local crystallization

Protocol `supervised_onset_information_v4`, branch `crystallization_supervised`.
Prepared after the user withdrew AP-specific tuning on 25 September 2026.
This is a fresh protocol, not continuation of an AP-trained checkpoint.

Question: how much future onset information is accessible from an exported local
state, and how does that depend on observed versus relaxed input and capacity?
The objective is not maximum AP. Supervised and self-supervised encoder branches
remain separate; this recipe concerns the supervised branch only.

Use the existing paired prospective source cohort and original-MD event labels.
Fit the discrete first-event/survival likelihood over bins ending 0.75, 3, 6, 9
and 12 ps. Retain importance-corrected event-enriched minibatches: this changes
sampling variance, not the natural source-weighted likelihood being optimized.
No AP loss, ranking replay, AP checkpoint selection, AP promotion or fitted
ensemble weights are allowed. Select encoder and frozen readout checkpoints
by minimum natural source-weighted selection NLL. Exact ties retain the earlier
checkpoint or declared arm order. The optional two-predictor ensemble has fixed
equal weights and is never labeled a single encoder.

Matched recipes: small, approximately 500k, 1M and 2M native MACE encoders; two
arms each, `O-NLL` observed and `R-NLL` relaxed. Both export 128 dimensions, use
one seed, effective batch 256 and microbatch 256. The matched cache, model sizes,
learning-rate schedule and declared update limits are retained from the capacity
comparison. Fresh outputs are under `encoder_supervised/information-runtime-20260925`.
These recipes have not been submitted by the policy change.

Encoder input: current centered local Al coordinates and center indicator,
maximum 80 atoms, 8 Å support, 5 Å edges, two message-passing layers, no halo,
history or motion. Relaxed input comes from full-current-cell relaxation before
local cropping. Predictor/probe input: exported state only. No temperature,
simulation age or explicit time covariates; no training teacher in these arms.
Descriptor controls use their declared observed/relaxed geometry descriptors.
Full input ledgers are mandatory.

Report first-event NLL, raw and calibrated horizon log loss/Brier, calibration,
and AP3/AP6 with source uncertainty. Three ps is the main scientific horizon;
six ps is secondary. AP is diagnostic, not a selector. Compare joint heads with
fresh frozen linear and nonlinear probes and constant/descriptor controls.
Evaluate benefit relative to those matched baselines; none of these scores alone
measures all information retained by the state. Source-level splits and the
untouched test role remain mandatory.

Retain dataset/fitting/test spectra and input-noise response normalized by local
spacing. The existing paired cache has 12 ps anchor cadence, so 0.75 ps movement
and movement rank remain explicitly unavailable; do not substitute another lag.
Additional present-information and state-sufficiency probes are future research
extensions, not implemented results of this protocol change.

Reproduction and execution: [operations](../../docs/supervised_capacity.md),
[campaign recipe](../../configs/supervised_onset/information_20260925/campaign.json),
[research policy](../../docs/encoder_research/training_branches.md).

Runtime: native cuEquivariance layout and fused convolution, full-graph dynamic
compilation, immutable geometry with bounded reusable index plans. The model
factory builds spatial layers directly; identical numeric seeds do not imply
identical historical initializations. Runtime comparisons use matched copied
weights; scientific fits use fresh identities. See the [runtime record](../../docs/encoder_research/runtime_refactor_20260925.md).
