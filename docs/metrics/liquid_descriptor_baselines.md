# Mean and linear controls for the descriptor comparison

This additive analysis imports the original GPU descriptor predictions without
refitting them. It uses exactly the same sealed row IDs, conditional weights,
source roles, 3,536 geometry-only descriptors and distance target. It does not
introduce conditions, time, temperature, phase labels or material inputs.

**Training mean:** every row receives the weighted training mean of min(D,64 Å).
This minimizes the training squared error (equivalently Gaussian negative log
likelihood with fixed variance). There is no fitted feature transform or validation
selection for this model. It is different from the existing histogram prior,
whose mean uses bin midpoints.

**Affine ridge:** standardize every feature using weighted training means and
population standard deviations, with a 1e-4 scale floor. Fit an intercept and
coefficients by minimizing sum_i w_i(y_i-b-x_i beta)^2 + alpha ||beta||², where
y=min(D,64), weights sum to one and standardized features have weighted mean zero.
The intercept is the weighted training target mean. Compute one weighted Gram
matrix and solve each alpha using its eigendecomposition; numerical negative
eigenvalues are clipped to zero. Predeclared alpha candidates are .01/.1/1/10/100.
Select the minimum full-validation Gaussian NLL with variance fixed to the
training residual variance of the constant-mean predictor. This is equivalent to
minimum validation MSE. Predictions remain affine and are not clipped.

Mean and ridge are **point-only diagnostics**. Report capped-target RMSE and MAE;
censored distance NLL and Brier fields are undefined, represented by blank CSV
cells. Their Gaussian selection objective is on capped distance and is not mixed
with the censored density likelihood used by boosting. They do not participate
in the density-likelihood winner selection.

**Linear distribution:** L2 multinomial logistic regression, affine logits with
an intercept and no hidden layers, using the same nine finite distance intervals
and censored tail as boosting. Fit with weighted training likelihood; weights sum
to one, so scikit-learn's inverse regularization C has that specific normalization.
Predeclared C values .01/.1/1/10 correspond to L2 coefficients 100/10/1/.1 in the
average log-loss objective, with the usual half-squared norm penalty convention.
Use L-BFGS, max_iter=2000 and tolerance 1e-5; convergence warnings fail the fit.
Feature standardization is training-only. Select C by full-validation censored
distance NLL, never calibration/test. Predictions use stable softmax and the exact
parent `quantities` function, including clipping, bin widths, midpoint means and
20/32/48 Å probabilities. Export NLL, RMSE, MAE and Brier scores.

The combined table rereads hash-verified predictions for every original model and
the three additions, matching the complete original row array exactly. All scores
use the inherited weights, normalized within the selected role. Positive relative
RMSE reduction is 1-RMSE(model)/RMSE(training mean). Positive NLL gain is
NLL(original histogram prior)-NLL(model). All point predictions target min(D,64).

Paired bootstrap uncertainty resamples entire independent sources, preserving
their conditional weight mass, for 2,000 draws with seed 20260928. Ordinary 95%
intervals and a multiplicity-adjusted one-sided RMSE-benefit upper bound cover
all twelve comparisons to the mean. Two-sided Bonferroni NLL intervals cover ten
probabilistic comparisons to the histogram prior. Point-only baselines do not
inflate or enter NLL comparisons. Hyperparameters are selected on validation;
intervals on the reused test cohort do not include training-seed uncertainty.

The additions were requested after the original descriptor queue began. Preserve
the original ten-model comparison and its definitions. Export the augmented table
to `analyses/comparison-with-baselines-v2`, with separate frozen definitions.

Derived sensitivity/relaxation protocols may reuse these fitted-model calculations
on explicitly sealed derived cohorts. Their prediction-context records name the
input domain and original versus synthetic versus relaxed-label target. Pairing,
synthetic generators and cold membership rules are defined in liquid_controls.md;
historical exported definitions remain frozen.

A derived cohort may have no training example in a declared distance bin. The
linear multinomial fit uses observed training classes; absent classes have zero
predicted mass and the shared 1e-12 scoring floor. Record absent bins explicitly.
Binary fits are expanded to the declared output space without changing logits.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.


## Explicit task-head refactor

The training/model refactor separates typed patch and spatial-context trunks from
task heads and expands training statements. Mathematical objectives, populations,
weights and selectors retain their definitions. Joint/rich-patch initialization
and state names are preserved; distance/control fresh initialization changes
when unused head construction is removed and receives a versioned architecture
identity. Historical continuations use their frozen sources. W&B wall-time stays
local and fixed baselines stay in summary; metric calculations are unchanged.
See [implementation and compatibility evidence](../code_cleanup_implementation.md#training-and-model-follow-up).

Paired comparison mechanics are implemented in `src/research/liquid_predictability/comparisons.py`: exact prediction-row alignment, weighted whole-source totals, seeded source resampling and confidence intervals. Population selection, target censoring, weights, point-estimate reductions and multiple-comparison rules retain the definitions above. Duplicate or missing prediction IDs are errors.
