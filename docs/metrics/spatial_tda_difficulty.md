# Spatial TDA difficulty: frozen trained heads

This diagnostic evaluates completed, fixed-final spatial checkpoints with their
jointly trained nonlinear descriptor heads, without fitting a probe. It uses the
original 4,096 validation observations and their saved, training-standardized
targets. Inference reuses each run's frozen model code, recorded execution variant,
BF16 encoder, FP32 head, batch 256, and both original 0.01-A jitter views with
seed `training_seed + batch_start`. All reported errors average the two individual
view errors; predictions are not averaged before scoring. FP64 reductions are
used for the diagnostic tables. The reconstructed weighted loss must agree with
the recorded validation loss within an absolute tolerance of 0.0002.

`components.csv` retains every output, its scientific name, original objective
weight, and active status. Zero-weight outputs are excluded from difficulty
rankings and block means: their training standard deviation was below 0.0001.
Their untrained outputs must not be interpreted as failed learned descriptors.

- `mse`: mean squared error in the original training-standardized coordinates,
  averaged over observations and views. Lower is better.
- `baseline_mse`: mean squared standardized target, the error of predicting the
  **training** mean (zero in these coordinates) everywhere. No validation fit.
- `mean_baseline_skill`: `1 - mse / baseline_mse`. For a block, ratio of summed
  errors and summed baseline errors across its active outputs. One is perfect,
  zero matches the constant training mean, and negative is worse. This is a
  supplementary score; the primary ranking and plot use R2.
- `target_variance`: population variance over validation observations, with
  equal weighting of active coordinates within each block.
- `r2`: one minus summed error divided by summed validation target variance;
  undefined feature values (variance <= 1e-12) are blank. One is perfect, zero
  matches the constant validation mean, and negative is worse. The plot shows
  100 times R2 for readability; these numbers are not classification accuracies.

Eighteen TDA blocks partition n32/n80 x H0/H1/H2 x statistics/spectrum/Betti.
H0 describes finite component-merger intervals; H1 loops; H2 enclosed cavities.
Statistics are normalized count, three log-lifetime summaries, lifetime entropy,
and log-birth/log-death sums. Spectrum means 12 Gaussian death-radius bins for H0
or a 6x6 smoothed birth/lifetime persistence image for H1/H2. Betti curves are
smoothed counts at eight radii from 0.75 to 4.25, normalized by neighborhood size
minus one. The targets use alpha-complex filtration radii, in the collection's
fixed material-normalized length coordinates. Infinite intervals are omitted.

`models.csv` includes reconstructed `descriptor_half_mse = 0.5 * sum(weight * mse)`.
Weights give half the mass to bond order and half to the 18 equally weighted TDA
blocks, equally weighting active coordinates within each block.
`tda_weighted_mse` and `tda_mean_baseline_skill` retain those TDA block weights.
`rank_agreement.csv` is Spearman correlation of the 18 block R2 rankings.

This is a descriptive single-seed comparison on 4,096 Al validation patches from
15 distinct sources/ancestors, following mixed-material spatial training.
The population is audited by mapping the saved panel row IDs to the original
descriptor manifest's selection shards. This is not a nucleation metric,
an independent test-set claim, or a
measurement of convergence speed. Row dependence is not treated as independent
replication; no significance intervals are asserted. All checkpoints, saved
predictions, panel inputs and frozen producer paths are recorded in `technical/`.
