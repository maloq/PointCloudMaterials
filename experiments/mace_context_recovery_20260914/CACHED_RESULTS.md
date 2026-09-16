# Completed cached-feature recovery comparison

The nonlinear readout does not recover the lost instantaneous-TDA information
from smooth-inner embeddings in this tested probe family. Combining inner and
center features does recover local physical changes, while leaving a substantial
instantaneous-TDA gap. This motivates the matched encoder-training experiment.

All eight representations completed both ridge and residual nonlinear readout
evaluation. The original six single-representation ridge results reproduce the
prior pilot scores numerically. All selected ridge penalties are interior to the
grid. Nonlinear heads select regularization, initialization and epoch using
validation sources, including the original ridge as the zero-correction candidate.
The cohort comprises 18/6/6 sources, 5,760 anchors and 144 test temporal tracks.

| Retained trained representation | Readout | Instantaneous TDA MSE | Relaxed TDA MSE | q6 increment error reduction at 0.75 ps |
|---|---|---:|---:|---:|
| Original | Ridge | 0.017150 | 0.035616 | 11.17% |
| Original | Nonlinear | 0.017008 | 0.034684 | 10.42% |
| Smooth inner | Ridge | 0.041476 | 0.030984 | 3.26% |
| Smooth inner | Nonlinear | 0.041476 | 0.030049 | 2.16% |
| Inner + center | Ridge | 0.040212 | 0.031326 | 21.92% |
| Inner + center | Nonlinear | 0.040177 | 0.030925 | 20.85% |

The nonlinear trained fusion also reconstructs density and mean-distance changes
with 68.20% and 97.61% error reduction versus persistence. These are decoded
changes using embeddings at both observed times, not forecast performance.

Trained fusion combines the separately optimized pilot encoders and has 512
features. The frozen fusion shares one backbone and likewise recovers local
information, with ridge instantaneous/relaxed errors 0.040124 / 0.031535. The
next joint experiment tests a shared trained backbone explicitly rather than
silently treating the separately trained fusion as that model.

No encoder changed in these probes. Nonlinear corrections can amplify feature
changes, so all four crossing parameters are scored both in feature space and
after decoding; those curves remain in the complete machine summary. Readout
nonlinearity is not proven incapable of improving further by this finite sweep.

These results use the already examined exploratory cohort. The two MLP seeds
are validation candidates rather than a training-seed uncertainty study. Paired
six-source bootstrap intervals and every retained observable appear in the
[full output](../../output/mace_context_recovery/forecast-seed20260910-20260914/README.md).
