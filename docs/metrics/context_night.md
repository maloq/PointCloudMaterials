# Overnight observed-context trajectory predictions

Long-horizon calculations are unchanged from `crystallization_paths_refinement.md`:
32 physical/embedding states at3ps spacing through96ps, original MD event CDF at
0.75ps resolution, source-weighted MSE/CRPS, event likelihood, integrated Brier,
calibrated alarm and timing scores. Forecast evaluation never feeds future targets
back into the predictor. Autoregressive teacher forcing is training-only.

The added `short_horizon` block uses CDF indices1/4/8/12/16 (0.75/3/6/9/12ps),
converted to five conditional hazard logits. True delays >12ps are censored at
12ps. `crystallization_information.md` defines the resulting source-weighted NLL,
AP, Brier, AUROC,5%-calibration-FPR threshold, recall, precision and conditional
timing MAE. Miss counts accompany MAE; overlapping event windows are not independent
events. Short and96ps NLL are different quantities and cannot be compared directly.

Observed added inputs: current local geometry/order93, their changes over3/6/12ps
(279 components), four smooth shell counts/mean radii in7–17Å and17–25Å; optionally
128 features from a frozen newly trained current-snapshot encoder. All arms have
504 auxiliary slots and the same MLP. Disabled inputs are zeroed after source-
weighted training-only standardization; no labels/PTM/future geometry enter it.
Descriptor-only controls replace every learned context feature with its calibrated
training mean, making its normalized value zero; observed positions/time remain.

All predicted trajectories retain the same original MACE latent targets, physical
packet, order targets and original MD crystallinity. The new-encoder bridge is an
additional current observation, not a replacement for all old neighboring/history
encodings. This keeps target-space comparisons fixed and avoids a misleading
cross-encoder latent MSE comparison. Promotions minimize development integrated
Brier within10% of the family's best development physical MSE; never test metrics.
