# Separate encoder, head, Fourier target and variance effects

The first local shooting baseline improved clear-liquid prediction versus the
prior, but its newly trained representation did not outperform frozen references
with linear readouts. Unconstrained moments implied negative variance in about21%
of evaluated coordinates. These findings motivate three controlled questions:

1. Does a matched nonlinear head explain the apparent advantage over frozen
   encoders? Add the user's MM-TDA-BLOCK-DIRECT-FULL reference with its true z256
   architecture and compare linear and 128-hidden nonlinear heads.
2. Does Fourier supervision improve prediction or information retained beyond
   first/second moments? Cross full274 with moments18 using the SAME moment-based
   validation selector and fixed target transform.
3. Does enforcing nonnegative marginal variance help or hurt predictive accuracy?
   Cross a free head with a softplus-variance head while keeping the moment targets.

Each combination uses three seeds. New MACE128 encoders train from scratch;
VICReg, Epi and MM-TDA encoders remain frozen. MM-TDA keeps its larger historical
capacity and broader pretraining population. This is an informative reference,
not a matched pretraining or capacity ablation. Archived ancestry limitations
remain explicit. Source roles and all Al480 rows are unchanged.

Comparisons report common moment errors, available full-feature errors, physical
variance violations, full-target frozen readouts of each selected representation,
and predeclared paired source intervals. Moments-only heads do not receive invented
Fourier predictions. Positive variances alone do not guarantee a valid joint law.

Recipe: [followup_20261001.json](../../configs/predictive_baseline/followup_20261001.json).
Reproduction: `python -m src.research.predictive_followup.lanes submit --config configs/predictive_baseline/followup_20261001.json`
after the [documented preparation/gates](../../docs/predictive_baseline.md#matched-head-target-and-mm-tda-follow-up).
Results: [heads-targets-mmtda-20261001](../../output/predictive_baseline/heads-targets-mmtda-20261001/README.md).
[Exact definitions](../../docs/metrics/predictive_followup.md).

The original full-feature-selector results are historical references. The
follow-up reruns all four joint variants under the declared common selector;
no old checkpoints or exported calculations are redefined.
