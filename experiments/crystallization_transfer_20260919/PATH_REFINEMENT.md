# Targeted improvements to structural trajectory forecasts

The first ten-fit comparison is complete. Longer training was not consistently
helpful. The direct model selected epoch3 at both budgets; its selection Brier
worsened from0.1047 to0.1288 by epoch12, and from0.1053 to0.1435 by epoch24.
Deterministic AR selected epoch7/12 or13/24 and subsequently worsened too. These
curves motivate lower learning rate, regularization and checkpoint-based stopping,
not an unconditional increase in duration.

Selection-only replay uses eight fixed windows per selection source. On this
smaller population Gaussian AR's Brier is0.1463 with sampled feedback, versus0.1353
with mean feedback; a teacher-forced future-input oracle reaches0.0186. That oracle
is a diagnostic of dependence on unavailable future states, never a valid forecast.
The old training recipe mixes teacher forcing and mean rollouts, while inference
feeds noisy samples. Test exact conditional-likelihood training and correlated
innovations to reduce this mismatch. The new 128-window/source selection population
is larger; its raw scores must not be paired directly with these replay values.

The mixture has effective component count2.65, with gate means approximately
0.056/0.111/0.469/0.363. It has not collapsed to one component, so a forced-uniform
gate penalty is not justified. Between-component mean variance is0.0535, versus
within-component variance0.851: much of its diversity is independent Gaussian
noise. Test physically meaningful onset/survival components and stronger structural
anchoring instead of simply increasing the number of mixture modes.

The diffusion failure is a representation/sampling defect in this implementation,
not evidence that diffusion cannot work here. Its 269 noisy channels pass through
a rank-at-most128 projection while it must recover all269 noise channels. A
null-space input perturbation changed denoiser output by only4.53e-6. At the final
noise level alpha_bar=5.93e-7, epsilon-to-clean conversion amplifies errors by1299.
Terminal noise MSE0.549 becomes clean-state MSE925,923 in the replay. The generated
first-frame event probability averages0.239 despite no first-frame events in these
120 sampled selection windows. Fix the formulation before drawing conclusions
about diffusion's quality or allocating longer unchanged runs.

Use stable velocity/clean prediction with a full-dimensional noisy skip and a
zero-terminal-SNR schedule, plus global projection of event curves onto valid
absorbing trajectories. The general motivation for stable terminal-noise treatment
is discussed in [Lin et al.](https://arxiv.org/abs/2305.08891); its efficacy here must
be demonstrated. The distinction between autoregressive likelihood and scheduled
sampling is discussed by [Huszár](https://arxiv.org/abs/1511.05101). These papers do
not validate the present materials model or forecast task.

Thirty one-seed screens, each capped at12epochs, cover:

| Family | Questions |
|---|---|
| Direct | Lower LR; regularization; structural loss weight; current-state residual; H48; observed motion |
| Deterministic AR | Same controls; no teacher forcing; shorter teacher-forcing decay |
| Gaussian AR | Exact teacher-forced likelihood; diagonal/rank8/rank16 innovations; lower LR |
| Mixture | Original control; two/four event-stratified components; stronger structural objective |
| Diffusion | Stable v128/v384; direct clean prediction; event weighting; sampling steps; EMA |

All retain the same fixed 150-source split, original MACE encoder, 96ps physical
forecasts, dense0.75ps onset labels and calibration/test protocol. Only the two
explicit motion ablations receive current velocity summaries. Source timelines
and all physical targets already exist; no new simulations or labels are needed.

Promote one setting per family to a36-epoch cap after all screens finish. Choose
lowest selection Brier among candidates within10% of that family's best physical
prediction error. Early stopping uses validation only. This retains physical
information as an explicit constraint on event-focused selection. New timing,
misses, physical errors and calibration will determine whether these changes help.
The ongoing native fine-tuning/scratch attention study and its longer promotions
remain a separate scientific protocol.

[Diagnostic artifacts](../../output/crystallization_transfer/path-diagnosis-20260919/technical/diagnostics.json)
· [Metric definitions](../../docs/metrics/crystallization_paths_refinement.md)
· [Execution](../../docs/crystallization_paths_20260919.md)

## Native encoder study remains in progress

At preparation of this follow-up, 21 native attention screens were complete and four running. Best completed selection NLL was 0.912759 for fine-tuning with 48 ps history, 0.917953 for the frozen reference, and 0.924220 for the scratch reference. These are provisional, single-seed selection comparisons, not final test improvements. The native queue already tests encoder/head learning rates, spatial and temporal attention, depth/width, radii, histories, repeated-history controls and tensor context. Its validation-selected 12/24-epoch promotions remain queued. Completing those controls is more informative than duplicating them in the new frozen-encoder trajectory study.

## Completed outcome

All 35 fits completed. See the [consolidated report](../../output/crystallization_transfer/recent-report-20260919/README.md#6-completed-targeted-trajectory-refinements) and [all-fit appendix](../../output/crystallization_transfer/recent-report-20260919/REFINEMENT_FITS.md). Stable diffusion removes the numerical failure but still trails direct/deterministic AR on event NLL. Gaussian covariance changes and event-stratified mixtures do not establish consistent improvement.
