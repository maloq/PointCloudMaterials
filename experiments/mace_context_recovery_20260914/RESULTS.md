# Completed shared-backbone recovery experiment

Combining smooth-inner and tracked-center features recovers much of the lost
local bond-order, density and distance information while retaining continuity at
the tested neighbor-membership boundary. The additional physical supervision does
not substantially close the instantaneous-TDA gap in this 12-epoch experiment.
The combined embedding has more temporal variation than the inner block alone.

Both detached runs completed all 12 epochs and their selected-checkpoint
evaluations. Runtime including evaluation was 35.5 minutes for the SSL control
and 39.4 minutes for physical supervision. Checkpoint checksums, selected epochs,
implementation/input hashes and successful final collection were verified.

All rows below use the same independently fitted linear ridge readout protocol.
TDA values are dimensionless balanced mean squared errors (lower is better).
Physical-change values are percent error reduction versus zero-change persistence
at 0.75 ps (higher is better), using embeddings at both observed endpoints. They
are reconstruction scores, not forecasts.

| Representation | Instantaneous TDA error | Relaxed TDA error | q6 change reduction | Density change reduction | Mean-distance change reduction |
|---|---:|---:|---:|---:|---:|
| Original 80-atom encoder, earlier pilot | 0.017150 | 0.035616 | 11.17% | 26.60% | 82.26% |
| Smooth inner, earlier pilot | 0.041476 | 0.030984 | 3.26% | 3.29% | 18.45% |
| Shared inner + center, selected control (epoch 0) | 0.040124 | 0.031535 | 20.81% | 61.07% | 96.98% |
| Shared inner + center, physical supervision (epoch 12) | 0.040117 | 0.031440 | 21.14% | 60.84% | 96.94% |

Both shared models output 512 features, unlike the earlier 256-feature models.
The direct comparison of supervised and control shared models has matched width
and initialization. The two earlier rows provide application baselines, not
matched training-duration or checkpoint-selection controls.

Compared with its selected shared control, physical supervision reduces linear
readout instantaneous-TDA error only **0.0174%**, with a paired six-source 95%
bootstrap interval of **-0.321% to +0.365%** reduction. Relaxed-TDA error decreases
**0.300%** (interval **-0.087% to +0.674%**). Both intervals include zero. q6-change
error reduction rises only 0.33 percentage points, while density and distance
scores change slightly in the other direction. Most of the local-information
recovery therefore comes from exposing the center feature, not from these extra
encoder updates. The supervised model's instantaneous-TDA error remains **2.34
times** the original encoder's; its relaxed-TDA error is **11.73% lower**.

Checkpoint selection matters. Both variants use the same physical-head validation
criterion and include their common initialization as epoch zero. None of the
SSL-only continuation checkpoints beat its initial validation score, so its
reported model is the initial encoder plus warmed-up physical heads. The
supervised variant selects epoch 12, improving aggregate validation loss from
0.094113 to 0.093284 (0.88%). This is not evidence that a trained SSL control
improved. The final epoch of that control is preserved separately in `last.pt`.

The trained nonlinear heads give a modestly different tradeoff from refitted
ridge. Physical supervision improves their instantaneous-TDA MSE from 0.048872
to 0.047188 (3.44% reduction, 95% interval 1.95–4.89%) and relaxed-TDA MSE from
0.033198 to 0.032241. Their q6-change reduction declines from 25.39% to 24.16%,
while density improves from 72.72% to 75.15%. Their TDA errors remain higher than
the separately refitted linear readouts. Improved task-head performance alone
does not establish a correspondingly large improvement in encoded information.

The boundary test still shows a continuous response. For the supervised shared
model, the squared crossing response divided by natural 0.75 ps latent change is
2.39e-4, 2.47e-6, 2.46e-8 and 2.57e-10 as epsilon decreases through
0.1/0.01/0.001/0.0001 Angstrom. This approximately quadratic decrease contrasts
with the original encoder's finite plateau near 0.03671. The decoded hot-TDA
crossing response also decreases, reaching 6.32e-9 of ordinary hot-TDA change for
ridge and 1.83e-9 for the trained heads at the smallest epsilon.

Temporal variation must be distinguished from that continuity result:

| Representation/block | Normalized squared change at 0.75 ps | At 12 ps |
|---|---:|---:|
| Original encoder, earlier pilot | 0.048409 | 0.108189 |
| Smooth-inner encoder, earlier pilot | 0.027008 | 0.075617 |
| Supervised shared model: inner block | 0.026259 | 0.075822 |
| Supervised shared model: center block | 0.295416 | 0.470684 |
| Supervised shared model: both blocks, equally scaled | 0.160837 | 0.273253 |

The inner block retains low relative temporal variation, but the complete
representation has **3.32 times** the original encoder's normalized squared change
at 0.75 ps when both blocks have equal training variance. Each block is scaled
using its own mean training feature variance; this prevents arbitrary raw feature
scales from determining the comparison. Center information is more time-varying,
even though the tested boundary discontinuity remains removed. The combined
model should not be described as uniformly smoother than the original.

This is one initialization and one joint-loss recipe on the already examined
six-source test cohort. The result does not establish that other supervision
weights, training schedules or richer readouts cannot help. The instantaneous
hard-80 TDA target's own boundary discontinuity also remains. Forecasting and
fresh-source confirmation have not been evaluated by this experiment.

[Full comparison table](../../output/mace_context_recovery/forecast-seed20260910-20260914/tables/comparison.csv),
[machine scores and paired intervals](../../output/mace_context_recovery/forecast-seed20260910-20260914/technical/summary.json),
[metric definitions](../../docs/metrics/mace_context_recovery.md), and
[earlier cached-probe findings](CACHED_RESULTS.md).
