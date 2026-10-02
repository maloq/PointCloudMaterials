# Merged evaluation with model selection confined to training

User-requested revision, 2 October 2026: combine the former selection,
calibration and test populations into one evaluation population. Keep all
original training sources and all original birth/control examples.

New split: 965 training histories and 510 evaluation histories, with 20% cases
in both. Evaluation contains 36 births in 28 contributing sources. Internal
five-fold source CV on the original training population chooses regularization
or boosting length by NLL; final readouts fit all training rows. Probability
calibration is removed. All parent split definitions and results remain intact.

Retest the original/relaxed descriptor-family, temporal, truncation and frozen
MACE comparisons. Deduplicate identical numerical treatments. The expanded
recipe contains 324 final readouts with one fitting seed. Compare train and
merged-test error, source uncertainty, original-role strata and feature reliance.

The enlarged test is previously inspected and cannot become confirmatory by
merging roles. Rich-MACE's historical structural checkpoint selection also used
former validation sources, which is recorded explicitly. Readout selection in
the new study uses none of the merged-test labels.

Reproduce with `python -m src.research.birth_prediction.merged submit --config configs/birth_prediction/merged_retest_20261002.json`.
See [workflow](../../docs/birth_prediction.md#merged-evaluation-retest) and
[metric definitions](../../docs/metrics/birth_merged_test.md).
