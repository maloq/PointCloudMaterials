# Physical descriptor families in selected birth-site prediction

Question: which geometric, bond-order, common-neighbor and topological information
supports prediction, and does its contribution change after relaxation?

The predeclared comparison uses the latest crystal-free snapshot (0.75 ps before
first local PTM appearance), original and relaxed inputs, logistic regression and
the retained depth-4 boosted model. For each family, fit it alone and omit it
while refitting the other families. Reuse the exact completed full-feature models.
Measure their reliance by exchanging whole feature groups inside matched
source/time case/control sets. Secondary permutations resolve l=6 order and
H0/H1/H2 topology at 32 and 80 points.

The original rows, event weights, source roles, five source folds, fitting seed,
likelihood selector and calibration are fixed. 192 subset refits complement 24
existing full-feature models. Raw and calibrated NLL/Brier, diagnostic AP/AUROC,
matched discrimination, train error and paired source intervals are reported.

A family can support prediction alone yet be replaceable when correlated
families remain. A permutation effect with little loss after refitting indicates
model reliance that alternative available features can compensate. Differences
between relaxed and original inputs are paired observational comparisons, not
evidence that an identified descriptor causes nucleation.

This preserves the retrospective selected-site question. It does not implement
prospective sampling, new relaxation controls, new encoders or new simulations.

Recipe: `configs/birth_prediction/feature_families_20261002.json`.
Reproduce with `python -m src.research.birth_prediction.feature_families submit --config configs/birth_prediction/feature_families_20261002.json`.
See [workflow](../../docs/birth_prediction.md#descriptor-family-information)
and [definitions](../../docs/metrics/birth_feature_families.md).
