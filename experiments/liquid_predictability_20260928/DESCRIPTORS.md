# Can fixed structural descriptors recover the missing distance information?

Add rich, fixed geometry representations to the crystal-free liquid distance
assay. This tests whether the earlier small descriptor set or learned MACE
representation missed available structural information. It does not establish
an information-theoretic upper bound when these models fail.

Reuse every source role, sample and proposal weight from the current assay.
No additional data or simulations. No crystal labels enter descriptors. The same
25 observed patches provide radial/angular/shape invariants, q/w bond order and
coherence, CNA bond fingerprints, and H0/H1/H2 alpha persistence at two local sizes.
Use shell moments and invariant spatial gradients/quadrupoles to summarize the
context; all 3,536 values are fixed before examining model scores.

| Arm | Features | Predictor |
|---|---|---|
| prior | None | Weighted training distance distribution |
| geometry_catboost | Radial, pair, angular, shape | CatBoost depth 6 |
| bond_order_catboost | Bond order and coherence | CatBoost depth 6 |
| cna_catboost | Common-neighbor fingerprints | CatBoost depth 6 |
| tda_catboost | Persistence images/curves/moments | CatBoost depth 6 |
| without_tda_catboost | Geometry + bond order + CNA | CatBoost depth 6 |
| tda_order_catboost | TDA + bond order | CatBoost depth 6 |
| all_catboost | All descriptors | CatBoost depth 6 |
| all_catboost_shallow | All descriptors | CatBoost depth 4 |
| all_mlp | All descriptors | Two hidden layers, width 256 |

One seed. CatBoost has at most 1000 trees, patience 100, learning rate .04,
weighted likelihood, and no class rebalance. The MLP has dropout .15, weight decay
.001, 12 blocks × 256 updates and batch 512; transforms fit training data only.
These are fixed descriptor controls, not encoder pretraining treatments.

Before any scientific fit started, the user changed boosting to GPU. Every tree
fit now uses one dedicated GPU; extraction, prior and MLP remain on CPU. The
initial CPU-only fit submission was cancelled. GPU MultiClass does not support
the proposed `rsm=0.7`, so each arm uses all features in its declared subset.
Plain boosting and Bayesian bootstrap (temperature 1) are explicit. The original
CPU recipe is preserved in its launch snapshot; inputs, splits, likelihood,
model candidates and validation selection remain unchanged. See the
[CatBoost GPU parameter restrictions](https://catboost.ai/docs/en/references/training-parameters/common).

Select by source-held-out validation distance likelihood. Test remains outside
selection. Every new model predicts the same piecewise-uniform distance density
with a right-censored tail at 64 Å. Report NLL, capped-distance RMSE, Brier and
reliability, paired source uncertainty, distance subgroups, and observations beyond
the entire spatial envelope. Report every candidate and the prior alongside the
validation winner. This likelihood family differs from the earlier lognormal
mixture, so the new no-input prior is the primary representation control.

The descriptor library follows the established [GUDHI alpha-complex convention](https://gudhi.github.io/alphacomplex/)
and [CNA bond-neighborhood approach](https://www.ovito.org/manual/reference/pipelines/modifiers/common_neighbor_analysis.html).
The custom CNA output consists of continuous bond fractions and moments, not OVITO
phase labels. [CatBoost MultiClass](https://catboost.ai/docs/en/concepts/loss-functions-multiclassification)
provides weighted categorical likelihood for the distance-bin distribution.

[Recipe](../../configs/liquid_predictability/descriptors_al64_20260928.json) ·
[Execution](../../docs/liquid_descriptors.md) ·
[Metric definitions](../../docs/metrics/liquid_descriptors.md).

Additional controls requested after the descriptor queue started: the weighted
training mean and affine ridge distance regression, plus a linear-logit distance
distribution for likelihood comparison. All use the same samples and weights.
Ridge regularization is selected by validation Gaussian likelihood on capped
distance; its density score is not compared to the original censored likelihood.
The linear-logit model uses that same censored histogram likelihood as boosting.
Mean/ridge appear as point-error references; the probability model participates
in validation-likelihood selection. The original report is preserved and an
augmented comparison is exported separately. [Control recipe](../../configs/liquid_predictability/descriptor_baselines_20260928.json).
