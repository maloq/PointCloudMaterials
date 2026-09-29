# How far from a crystal can its approach be detected?

[Completed results and post-hoc analysis](RESULTS.md): spatial context improves
distance prediction; warning distance depends strongly on the alarm's proximity
radius, and most warnings already include crystal in the contextual input.

[Literature comparison](../../docs/encoder_research/spatial_warning_literature_20260926.md)
identifies close Al interface studies and separates prior recognition results
from calibrated spatial warning and nucleation-precursor questions.

The user selected fixed snapshots on 26 September: measure warning distance,
with temporal warning time a separate question. Reuse full-cell crystal ancestry,
all Al64 sample IDs/source roles and a frozen observed encoder. Add matched probe
scans through calibration/test snapshots. No new simulation or encoder fitting.

| Fit | Observation | Question |
| --- | --- | --- |
| Geometry MLP | Local radial/count/bond-order descriptors, <8 Å | What do simple geometric readouts retain? |
| Frozen MACE + MLP | Focal z128, <8 Å | Does the learned state carry proximity information? |
| Symmetric invariant | Shared patch z128 at 0/10/20 Å | What does wider context add? |
| Vector messages | Same plus vector fields | Does directional organization help? |
| Harmonic hierarchy | Same plus tensor/higher bond fields | Does hierarchical orientation help? |

One seed, batch/microbatch 256, sixteen epochs, validation distance-NLL selection
after epoch twelve. The common encoder is the completed observed,
scratch-initialized temporal-onset-NLL model from the batch-1024 study; it remains
frozen. Geometry MLP does not consume the encoder. This is supervised transfer.

Predict nearest confirmed crystal-atom distance in six bins, with edges
4/8/12/20/32 Å. Report NLL, Brier, calibration and diagnostic AP. Calibrate alarms
to 5% empirical path-level false alarms using separate far-away scans. Evaluate
first warning distance, recall at 8/12/20/32 Å including misses, source intervals,
and whether actual input neighborhoods already contain crystal atoms.

Wider context can see a crystal before the focal patch reaches it. Detection
while all consumed atoms remain outside reference crystals would suggest
additional liquid-structure information and requires the far controls.
Studies in silicon/copper and NiAl motivate this question but do not establish
its answer in our Al data:
[interface-induced ordering study](https://www.nature.com/articles/s41467-020-16892-4),
[NiAl preordering study](https://www.nature.com/articles/s41467-022-32241-z).

Routes intentionally approach known crystals: this is a controlled diagnostic,
not an autonomous navigation policy. No true direction, distance, PTM/ancestry,
temperature or time is a model input. No AP tuning or data resplitting.

[Recipe](../../configs/analysis/spatial_approach_20260926.json) ·
[Metric definitions](../../docs/metrics/spatial_approach.md) ·
[Execution](../../docs/spatial_approach.md).
