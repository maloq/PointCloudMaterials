# Interface visibility and overfitting audit

The completed six-model audit does not show clear late-training overfitting or
cross-role source leakage. It does show weak localization when the interface is
absent from all input patches, including on training data. The dense, visibility-
restricted VCReg experiment tests that specific weakness; it has no results yet.

The audit reuses all six selected checkpoints' saved predictions, verifying their
hashes. It does not retrain, choose new checkpoints, or modify historical results.
[Definitions](../../docs/metrics/crystal_overfit.md) specify populations and weights.

## Selected interface + direction + VCReg model

The full validation predictive objective decreased from **4.7269** at block 1 to
**4.0991** at block 16, the selected checkpoint. The last block's logged training
minibatch objective was **3.8890**; that sampled training trace is not an exhaustive
training score. The selected checkpoint's distance-marginal scores are:

| Population | Role | Rows | Distance NLL ↓ | Capped-distance RMSE, Å ↓ | Brier, distance ≤20 Å ↓ |
|---|---|---:|---:|---:|---:|
| All visibility | Train | 63,251 | 2.2084 | 11.81 | 0.03199 |
| All visibility | Selection | 29,667 | 2.3275 | 12.01 | 0.03542 |
| All visibility | Test | 63,515 | 2.1140 | 12.13 | 0.02982 |
| Interface invisible | Train | 40,351 | 2.0307 | 13.40 | 0.01064 |
| Interface invisible | Selection | 18,481 | 2.2528 | 13.60 | 0.00980 |
| Interface invisible | Test | 42,160 | 1.9037 | 13.72 | 0.00778 |

The reference population gives half mass to fixed at-risk centers and half to
uniform centers, with equal source mass within each half, then conditions on the
named visibility group. Source distributions and censored fractions differ across
roles. A lower test NLL does not prove the absence of overfitting. The historical
crystal-set study also lacks uniform test interiors, so its mixed train/test scores
must not be interpreted as a matched generalization gap.

For invisible test contexts, the weighted prevalence of distance ≤20 Å is only
**0.001067**. An always-zero proximity probability has Brier score equal to that
prevalence, better than the model's **0.00778**. This is a strong warning about
overconfident rare-positive predictions, not evidence of useful early detection.
Direction NLL is **2.5569** on invisible training rows and **2.5604** on invisible
test rows with valid direction. A uniform spherical direction has NLL log(4π) ≈
2.5310: directional information is weak on both sides of the split.

## Feature and input checks

| Export | Train d95 | Test d95 | Train effective rank | Test effective rank |
|---|---:|---:|---:|---:|
| Local scalar embedding, 128 channels | 12 | 13 | 6.99 | 7.89 |
| Context scalar state, 128 channels | 2 | 2 | 1.99 | 1.93 |
| Local vector channel covariance, 16 channels | 3 | 3 | 2.54 | 2.54 |

d95 is the number of principal covariance directions explaining 95% of variance,
not intrinsic manifold dimension. These are complete fixed/uniform populations,
with empirical row weighting. No channel is constant at the declared 1e-6 standard-
deviation threshold. The context state is highly concentrated on both train and
test; that deserves attention as a representation limitation, but is not a
test-only collapse or proof of memorization.

Median distance to a held-out training-feature reference is similar across train,
selection and test: **4.57 / 4.70 / 4.65** for local scalars and **0.743 / 0.739 /
0.733** for context scalars after training-only channel standardization. This is a
support diagnostic, not a guarantee of generalization.

All six runs retain **150 source IDs and 150 independent-melt ancestors**, with
**zero ancestors shared across split roles**. Actual forward inputs are coordinates,
patch inverse indices and relative patch offsets. Encoder atom attributes are
constant; no time, temperature, source/material ID, phase or target is forwarded.
The visibility mask matches the radius-8 Å atom support of all 25 encoder patches.

## Alarms before any interface visibility

Each original test path is truncated before its first visible-interface query or
crystal entry. An alarm requires two consecutive original observations; discarded
segments are never concatenated. All **495** toward paths remain the denominator.
For the previous interface + direction + VCReg model, thresholding P(d≤20 Å):

| Probability threshold | Paths with alarm / all paths | Misses | Median distance at alarm, Å | Away-path alarms |
|---|---:|---:|---:|---:|
| >0.50 | 8 / 495 | 487 | 41.30 | 7 / 292 |
| >0.75 | 3 / 495 | 492 | 39.23 | 1 / 292 |
| >0.95 | 0 / 495 | 495 | — | 0 / 292 |

The long median distances describe a handful of alarms, not reliable warning
distances. Predicting high probability of an interface within 20 Å while it is
about 40 Å away is also a proximity false positive. These results do not establish
useful prediction beyond visible spatial context.

## Reproduction and artifacts

[Audit workflow](../../docs/crystal_interface_unseen.md) ·
[New experiment protocol](UNSEEN.md) · [Historical six-model results](RESULTS.md).

The complete audit bundle is
`${storage:analysis}/crystal_interface/review-20260928/analyses/overfitting-v1/`:

- [All six models: fit gaps](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/tables/fit-gaps.csv)
- [Feature spectra](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/tables/feature-spectra.csv)
- [Feature-neighbor diagnostics](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/tables/feature-neighbors.csv)
- [Learning curves](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/plots/learning-curves.png)
- [Strict unseen-interface alarms](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/tables/unseen-alarms.csv)
- [Input and ancestry audit](/work/PERSO/vmorozov/analysis/crystal_interface/review-20260928/analyses/overfitting-v1/technical/input-audit.json)

Frozen metric definitions and implementation hashes accompany the tables. This
audit uses one training seed per treatment and does not establish seed uncertainty.
