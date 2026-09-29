# Interim Ta evaluation after a user-requested stop

This is a frozen-checkpoint diagnostic, not a completed twelve-epoch fit or a
checkpoint selected by held-out outcomes. The user requested stopping Ta training
on 2026-09-27. Evaluate its latest durable checkpoint, recording its exact epoch,
against the unchanged CD-MACE128-D6-075nominal parent on the predeclared selection
and test branches. No optimizer runs and no new W&B run is created.

The population, targets, weighting and numerical definitions are unchanged from
[material fine-tuning](distance_encoder_material_finetune.md). Ta has 61,440
selection and 184,320 test windows over four different velocity branches of ONE
known starting structure. This is not independent-preparation generalization.
Every six-observation sequence uses exact 0.70-ps spacing and a 3.5-ps span.

`distance.csv` reports censored distance NLL, capped-mean RMSE, capped-median MAE,
censored fraction and CDF Brier scores, with equal trajectory weights. All
distance units are Al-equivalent Angstrom; the censoring/point-error cap is 64.
`confidence-reliability.csv` reports coverage, mean probability and observed
precision at strict probability thresholds .5/.75/.95 for the declared distance
radii. Empty selections remain undefined. These are pointwise distance measures,
not a Ta spatial warning or future nucleation assay.

Prediction arrays retain source, atom and raw frame identities; checkpoint hashes
and exact epochs are stored in the evaluation receipt. Final twelve-epoch exports
and selectors are untouched. Interim scores update the original training run
under `interim_material_distance/`, separately from final results. One training
seed and one preparation family do not support intrinsic material-difficulty or
statistical-significance claims.
