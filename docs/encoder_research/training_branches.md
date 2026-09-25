# Two encoder-training branches

We now keep two scientific questions separate. These are experiment branches,
with separate recipes, output roots and table labels; they do not require separate
Git branches.

| Branch | Encoder optimization and selection | Question |
| --- | --- | --- |
| `crystallization_supervised` | Original-MD future onset labels may train the encoder and select its checkpoint on selection sources. | Which predictive information can the encoder retain with direct task supervision? |
| `self_supervised` | No crystallization/onset labels in encoder losses, sampling, promotion or checkpoint selection. Use declared self-supervised validation objectives. | How much predictive information emerges without crystallization supervision? |

Frozen post-hoc readouts in either branch may learn onset labels from training
sources and select their own parameters on selection sources. Always distinguish
joint encoder/head scores from fresh frozen linear and nonlinear probes. Calling
a *frozen probe* supervised does not make its encoder supervised; selecting an
otherwise self-supervised encoder by onset AP **does** violate the strict second
branch. Repeated test-driven architecture selection also makes an old test set
development evidence rather than fresh confirmation.

**Standing policy, 25 September 2026:** study predictive information, not maximum
AP. Never use AP-specific losses, full-population ranking replay, AP-driven
checkpoint selection, promotion, hyperparameter search or fitted mixture weights.
Supervised encoders and their frozen probes train/select by declared predictive
likelihood objectives. Self-supervised encoders use their own label-free criteria.
AP3/AP6 remain evaluation diagnostics. Do not resume old AP-trained checkpoints
as if they were likelihood-only training.

Compare matched linear/stronger probes, natural-population log loss and Brier
scores, calibration, input-only controls and incremental predictive benefit.
Giving a matched predictor the original observation as well as the embedding
can test whether available predictive information was discarded. A log-loss
reduction relative to a baseline measures predictive benefit within those model
families; it is not automatically an estimate of mutual information.
Present-information retention, temporal stability, spectra and noise response
remain separate requirements. Neither AP nor NLL alone establishes state
sufficiency or an information-theoretic maximum.

For both branches use the same declared prospective targets, source ancestry,
natural-risk evaluation population, physical horizons and readout controls.
**3 ps is the main scientific horizon; 6 ps is secondary.** AP at both horizons
is diagnostic, never the optimization/selection goal. Report 12 ps for context.
Compare observed-input models separately from models that require a same-frame
relaxation at inference. Do not call a two-model risk ensemble a single encoder.

The active recipe is the [predictive-information study](../../experiments/supervised_information_20260925/README.md). The earlier AP studies are historical.
No self-supervised fits are submitted by that queue. Historical reports retain
their actual losses and selectors; they are not retroactively reclassified by
the new branch names.
