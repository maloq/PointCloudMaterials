# Line-JEPA Training Metrics

Line-JEPA metrics put the semantic category before the only `/`, because W&B
groups only by the first path component:

```text
{category}/{stage}_{metric}
```

For example, `similarity/val_top1` is validation top-1 target-matching accuracy.
The groups are deliberately separate:

- `prediction`: direct target or residual prediction.
- `similarity`: correct-target identification among hard negatives.
- `context`: matched-context versus shuffled-context controls.
- `manifold`: compactness and frozen-teacher geometry preservation.
- `clustering`: prototype consistency and unsupervised cluster probes.
- `masked_tokens`: GeoFrameTransformer's training-only EMA token-prediction objective.
- `view_consistency`, `regularization`, and `hard_weighting`: optional objectives and diagnostics.

## Current Dashboard

The continuous-manifold frozen-target configuration logs these primary metrics:

| Metric | Direction | Meaning |
|---|---:|---|
| `loss/train`, `loss/val` | lower | Complete weighted training objective. |
| `prediction/*_loss` | lower | Direct prediction loss. |
| `context/val_error` | lower | Matched-context normalized L2 prediction error. |
| `context/val_global_gain` | higher | Gain over contexts shuffled across the complete batch. |
| `context/val_local_gain` | higher | Gain over contexts shuffled within the same source group. |
| `train/vicreg_sim`, `val/vicreg_sim` | lower | Invariance error between two lightly jittered views. |
| `train/vicreg_std`, `val/vicreg_std` | lower | Variance-floor penalty that prevents dimensional collapse. |
| `train/vicreg_cov`, `val/vicreg_cov` | lower | Redundancy penalty between embedding dimensions. |
| `manifold/*_anchor` | lower | Pointwise drift from the frozen pretrained representation. |
| `manifold/*_relation` | lower | Drift of frozen-teacher pairwise geometry. |
| `clustering/val_silhouette` | higher | Separation diagnostic for genuinely disconnected groups. |

`*` means `train`, `val`, or `test`. Shuffle controls are validation/test only
because evaluating two additional predictor passes on every training batch would
be expensive.

The encoder starts from the pretrained VICReg checkpoint, remains frozen for two
epochs, and then adapts at `0.01` times the predictor learning rate. VICReg and
frozen-teacher geometry losses constrain that adaptation. The compact semantic
projector is initialized once from pretrained features by PCA and then fixed.
Hard-negative matching, masked-token training, and balanced prototypes remain
disabled. The checkpoint monitor is `clustering/val_silhouette`, while
`prediction/val_loss` remains the direct predictor-quality metric.

## Reading the Current Run

The optional hard-negative ablation uses 64 hard negatives, so its random
similarity performance is:

```text
similarity/{stage}_loss = log(65) = 4.174
similarity/{stage}_top1 = 1 / 65 = 1.54%
```

A useful continuous-manifold run should show these behaviors:

1. `prediction/{stage}_loss` decreases or remains low.
2. Both context gains are clearly positive. A gain near zero means the
   predictor gives almost the same answer after its context is replaced.
3. VICReg standard-deviation and covariance diagnostics remain non-collapsed.
4. Silhouette remains high for genuinely separated groups.
5. Downstream continuous coordinates vary smoothly across neighboring structures
   while retaining nonzero variance.

`last.ckpt` is also saved independently of the best-silhouette checkpoint.

## Continuous manifolds and soft regimes

Some structural states are distinct but connected by physically meaningful
intermediates. In that case the embedding contains a continuous order parameter,
not separated spherical clusters. A hard clustering algorithm can still divide
the continuum into regimes, but the boundary is a threshold along the manifold
rather than an empty gap.

Silhouette must not be interpreted as requiring every pair of regimes to have a
gap: it rewards gaps and penalizes valid transition samples. It remains useful
for boundaries that really should exist, such as a disconnected C1 manifold
versus the connected C6/C7 manifold. Use C6 and C7 labels as convenient regime
names and quantify their difference with a one-dimensional
discriminant/order-parameter coordinate, diffusion pseudotime, neighborhood
preservation, effective rank, and the overlap or Wasserstein distance between
regime-coordinate distributions. Plot the coordinate continuously instead of
only coloring it with hard cluster IDs.

### Post-training connected-regime analysis

The full post-training pipeline runs `clustering.connected_regimes` on the
primary clustering labels and original invariant latents. It deliberately keeps
connectedness and distinguishability separate:

- symmetric cross-label kNN contact and boundary-sample fractions measure
  whether two regimes touch locally;
- a PCA component selected for the largest standardized regime shift supplies a
  continuous order parameter;
- Cohen's d, Wasserstein distance, energy distance, and histogram overlap measure
  how different the two order-parameter distributions are without asking for an
  empty boundary.

Pairs can be supplied explicitly with zero-based IDs (`C6/C7` is `[5, 6]`) or
discovered automatically from kNN contact. Outputs are written to
`connected_regimes/`: an overview matrix, a four-panel figure and sample CSV for
each selected pair, a pair-metrics CSV, and the complete JSON summary. UMAP
coordinates are used only by the separate visualization pipeline and never by
these connected-regime metrics.

## Compact Semantic Manifold

The GeoFrame encoder exports a 64-dimensional feature, and the clustered and
predicted representation is a normalized 64-dimensional semantic embedding.
On the first training batch, its linear projector is initialized
from the top frozen-teacher PCA directions. Consequently `manifold/val_pca95`
is bounded by 64 by construction.

Unlike the legacy probe, PCA is computed after mean centering without scaling
each coordinate to unit variance. It therefore measures the natural variance
spectrum rather than the correlation rank of standardized coordinates.

### `manifold/{stage}_anchor`

Mean cosine distance between the online compact embedding of an augmented
environment and the frozen-teacher embedding of the same environment. Lower is
better. This preserves individual semantic positions while allowing the
predictor and prototypes to learn.

### `manifold/{stage}_relation`

Mean squared difference between online and frozen-teacher pairwise cosine
similarity matrices. Lower is better. It preserves neighborhoods and cluster
geometry even when individual coordinates rotate or rescale.

### `manifold/{stage}_pca95` and `effective_rank`

`pca95` is the number of raw centered PCA components needed for 95% explained
variance. `effective_rank` is `exp(entropy(explained_variance_fraction))`, a
continuous dimensionality measure. Both are evaluated on the exported compact
embedding. Lower is compact,
but values approaching one are not automatically good: they can indicate rank
collapse unless coordinate variance, neighborhood preservation, and downstream
continuity remain healthy.

## Clustering Metrics

Balanced prototype consistency is retained only as an ablation. It is disabled
in the continuous-manifold configuration because uniform prototype occupancy can
turn a smooth transition into arbitrary discrete bins. Different line directions
are never treated as equivalent views.

### `clustering/{stage}_prototype_loss`

Cross-entropy from the online prototype probabilities to balanced teacher
assignments, averaged over all prototype resolutions. Lower is better.

### `clustering/{stage}_assignment_agreement`

Mean hard-assignment agreement between online and teacher assignments over all
prototype resolutions. Higher is better.

### `clustering/{stage}_usage_entropy`

Entropy of mean online prototype probabilities, normalized by `log(K)` and
averaged over resolutions. `1` means balanced use; low values indicate that only
a few prototypes are active.

The validation probe separately runs KMeans in the natural compact space:

- `clustering/val_silhouette`: compactness versus separation at `probe_k`.
- `clustering/val_stability`: mean ARI across KMeans seeds.
- `clustering/val_balance`: normalized cluster-size entropy.
- `clustering/val_best_k`: best silhouette count in the configured search range.

## Masked-token Metric

`masked_tokens/{stage}_loss` is present only when
`line_jepa_masked_token_coeff > 0`. It measures the unweighted normalized token
prediction loss; the configured coefficient is applied only to the total loss.
The EMA target sees every patch, while the student predictor receives mask tokens
at the sampled patch positions. This is an auxiliary training signal: normal
GeoFrameTransformer encoding and analysis remain fully unmasked.

## Prediction Metrics

### `prediction/{stage}_loss`

The unweighted configured prediction loss between predictor output `p` and
prediction target `z`:

- cosine: `mean(1 - cos(p, z))`;
- MSE;
- Smooth L1.

Hard-example weights are applied here when enabled. The configuration
coefficient is applied only when forming the total objective, so this metric
remains comparable between runs with different coefficients.

For normalized cosine prediction, cosine itself is not logged because it is
exactly `1 - prediction/{stage}_loss` and adds no information.

### `prediction/{stage}_cos`

Logged only when the configured prediction loss is MSE or Smooth L1. In those
cases cosine measures embedding direction and is not a duplicate of the
optimized loss. Higher is better.

### Residual-mode metrics

These appear only for `line_jepa_prediction_target: residual`:

- `prediction/{stage}_residual_error`: mean absolute error of the predicted residual.
- `prediction/{stage}_residual_gain`: relative improvement over predicting a zero
  residual. `1` is perfect, `0` gives no improvement over context alone, and a
  negative value is worse than context alone.

Other residual metrics were removed because residual norm equals baseline
error, relative gain equals `1 - relative_error`, and reconstruction cosine was
strongly redundant with prediction error.

## Similarity Metrics

Similarity training asks whether a prediction identifies its own target among
hard alternatives. It is distinct from direct prediction: prediction minimizes
distance to `z`; similarity forces the answer to depend on the matched context.

### `similarity/{stage}_loss`

Cross-entropy over one positive target and the configured number of hard
negative targets. Lower is better.

Hard negatives are made without material or class labels:

1. Center frozen-teacher targets by the batch mean and L2-normalize them.
2. Compute all target-to-target cosine similarities in the batch.
3. Exclude the target itself.
4. Exclude candidates above
   `line_jepa_context_match_negative_max_target_cosine`; these are treated as
   likely false negatives because their target representations are nearly
   identical.
5. Choose the most similar remaining targets, up to
   `line_jepa_context_match_negative_count`.

### `similarity/{stage}_top1`

Fraction of predictions whose positive target has higher similarity than every
selected hard negative. Higher is better.

Negative count is configuration, not a learned metric, and is therefore no
longer logged. Hardest-negative target cosine was also removed because it mostly
tracks the frozen teacher and batch sampler rather than predictor progress.

## Context Controls

The controls keep the target and positional inputs fixed while replacing the
context embeddings.

### `context/{stage}_error`

Mean L2 error for the real matched context. Lower is better.

### `context/{stage}_global_gain`

Context is shuffled across the complete batch:

```text
(global_shuffled_error - matched_error) / global_shuffled_error
```

### `context/{stage}_local_gain`

The same calculation, but contexts are shuffled only inside the same source
group. This is the stricter diagnostic because broad source identity cannot
explain the gain.

For both gains, higher is better. `0` means matched and shuffled contexts are
equivalent; negative means the matched context is worse.

Shuffled errors and cosine versions are not logged separately. For normalized
targets they are highly redundant, while the gain is the decision metric.

## Optional Groups

### `view_consistency/{stage}_loss`

Weighted target-view consistency loss added to the total objective when
`line_jepa_target_view_sim_coeff > 0`. It compares two augmented views of the
same target. Lower is better.

### `regularization/{stage}_*`

Logged only when the corresponding coefficient is active:

- `regularization/{stage}_sigreg`: sliced Gaussian distribution regularizer.
- `regularization/{stage}_std`: VICReg-style standard-deviation penalty.
- `regularization/{stage}_cov`: VICReg-style off-diagonal covariance penalty.

All are lower-is-better.

### `hard_weighting/{stage}_*`

Logged only when hard prediction weighting changes sample weights:

- `hard_weighting/{stage}_mean_novelty`: mean target-to-context feature distance.
- `hard_weighting/{stage}_threshold`: smallest novelty included in the hard subset.

## Total Loss

The logger reports the complete weighted objective as `loss/{stage}`. For the
current continuous-manifold run, the active terms are:

```text
loss = line_jepa_weight * (
    line_jepa_prediction_coeff * prediction/{stage}_loss
  + line_jepa_semantic_anchor_coeff * manifold/{stage}_anchor
  + line_jepa_semantic_relation_coeff * manifold/{stage}_relation
)
+ vicreg_weight * (
    vicreg_sim_coeff * vicreg_sim
  + vicreg_std_coeff * vicreg_std
  + vicreg_cov_coeff * vicreg_cov
)
```

Optional similarity, prototype, masked-token, view, Line-JEPA regularization,
or SwAV losses are included only in ablation configurations when enabled.
Compare total loss only between runs using the same coefficients.

## Removed Legacy Metrics

| Old metric family | Replacement |
|---|---|
| `line_jepa_pred`, `line_jepa_pred_target_cos` | `prediction/{stage}_loss`; cosine removed when algebraically redundant. |
| `line_jepa_context_match*` | `similarity/{stage}_loss`, `similarity/{stage}_top1`. |
| `line_jepa_control_*` errors and cosines | `context/{stage}_error`, `global_gain`, `local_gain`. |
| `line_jepa_context_mean_target_cos` | Removed; online context features need not remain in the frozen target coordinate system. |
| Seven residual diagnostics | Two metrics: `residual_error`, `residual_gain`. |
| Raw and weighted target-view MSE plus cosine | One contributing metric: `view_consistency/{stage}_loss`. |
| Four hard-weighting statistics | `mean_novelty` and `threshold`. |
