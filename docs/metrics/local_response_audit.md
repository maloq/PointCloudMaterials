# Disturbance, target scale and competing-gradient audit

Read-only diagnostics on the completed local80 experiment with binding
852f2b168521b5a2ea567aadb7f5a4f4ef826fe52b2c2dc7a9936cae6198c029.
Scale, noise and gradient calculations use only its90 training parents. No new
fit, checkpoint selection, test-based hyperparameter selection or W&B run occurs.
Saved learning curves include the existing fixed15-source validation scores.

`disturbance.csv`: each column of the80×3×2 displacement basis has unit Euclidean
norm across all coordinates, fixes the central atom, and has zero summed
displacement over the other79 atoms. Epsilon is the norm of the entire combined
displacement, not a per-atom amplitude. Vector displacement RMS equals
epsilon/sqrt(79); Cartesian component RMS equals epsilon/sqrt(237).
Median/p95/max pool all90 parents,79 mobile atoms and two directions.
The2.8Å reference is the recorded encoder length parameter, not a newly measured
bond length. Original MACE labels are infinitesimal AD responses; these finite
epsilons are diagnostics, and are used by the alternative force-only engines.

`response-scale.csv`: `raw_branch_response_rms_per_A` pools90×8×prefix×2
derivatives of raw Fourier features; `raw_mean_response_rms_per_A` first averages
the8 shots, then pools90×prefix×2. Features are dimensionless; derivatives are
per Å of collective displacement. Standardized responses divide coordinatewise
by the original training feature standard deviations, without a mean subtraction.
The single response scale is their train mean-response RMS over all256×2
coordinates. Half-MSE fields describe a zero-output predictor's target energies;
actual reconstructed model losses/gradients are in `gradient-balance.csv`.

`linearized-feature-change.csv`: epsilon times raw/standardized mean-response
RMS predicts central-to-plus change to first order. The plus-to-minus RMS is
twice that. These are linear estimates, not newly simulated trajectories.
The first128 coordinates describe20fs; the second128 describe joint20+100fs,
not a100fs marginal. No displacement or crystal fraction is inferred from a
Fourier feature change.

`shot-noise.csv`: for standardized values or responses, mean_target_square is
mean(mean_shot(target)^2). estimated_noise_variance_of_mean is the mean of
sample variance(ddof1)/8. Their difference estimates squared underlying mean
signal; their ratio is noise_fraction_of_mean_square. This is a pooled Monte
Carlo decomposition, not a calibrated per-parent certainty or an uncertainty
interval. Values are train-feature centered; responses are not centered.

`training-audit.csv`: saved histories supply selected epoch, total optimizer
updates, max recorded preclip gradient norm, and the number above clip threshold10.
One epoch is one accumulated full90-parent optimizer update, microbatch8.
Subtract0.5*log(2*pi) from logged value NLL to recover the value half-MSE.
The chosen-epoch training losses were computed before that epoch's optimizer
step, whereas selection scores were computed after it. Do not label those losses
as exact re-evaluations of the saved checkpoint.

`gradient-balance.csv`: reconstruct each seed's exact initialization and read
the best/last response checkpoints and best values8 checkpoint. Compute full
train90-parent gradients in fixed microbatches, separately for value half-MSE
and response half-MSE, using the original scales and response weight1. No
optimizer step occurs. Norms are Euclidean parameter-gradient norms; the ratio
is response_norm/value_norm (denominator floor1e-30), cosine is their dot product
divided by the product of norms (same floor), and combined norm is norm(gv+gh).
Rows report all trainable parameters and the encoder/head subsets separately.
Negative cosine indicates competing instantaneous parameter gradients, not proof
that the objectives are fundamentally incompatible. Learning rates and Adam
preconditioning are not included in these gradient norms. Actual checkpoint
losses in this table are re-evaluated at the saved weights.

The saved32-shot validation labels remain fixed across epochs. These diagnostics
cannot establish what prematurely stopped seeds would do after further training.
The static learning-curve figure marks existing selectors and draws no invented
continuation. All producer, metric and input identities are preserved.
