# Forecast follow-up diagnostics, version 1

Replay eight fixed windows per selection source, never calibration or test, using
the selected checkpoint of the lowest-selection-Brier budget for each method.
Event Brier is the source-weighted mean over 128 future 0.75ps times. Path block
MSE uses the original block-balanced standardized structural loss. Training curves
report the best selection epoch and the final selection score, not training error.

Mean feedback and stochastic feedback use the same Gaussian AR checkpoint.
Teacher-forced Brier is a explicitly labeled future-input oracle diagnostic: it
is never deployable, eligible for selection, or compared as a legitimate forecast.
Mixture entropy is exp(-sum p log p), averaged over windows; report mean gate
weights, variance between component means, and mean within-component variance.

Diffusion diagnostics use forward-noised real selection targets at the terminal
training level to separate denoising from sampling. Clean reconstruction MSE
uses the actual epsilon-to-clean formula. The amplification factor is
1/sqrt(alpha_bar). Report input and hidden dimensions and perturb inputs along
the last right-singular vector of the rank-deficient input projection; a small
output change confirms that the missing noisy direction is invisible. This is
an implementation diagnostic, not a learned generalization score.

No test scores choose follow-up settings. Earlier test results have already been
inspected in this project; future reuse of this same test cohort is exploratory,
not a new untouched confirmatory holdout. Values have no uncertainty intervals.


Table export: 2026-09-19T13:57:35.288399+00:00. The machine-readable values retain full precision; blank values mean undefined or unrecorded, never zero. Nested metric names preserve the producer's grouping. The implementation hashes are in `../technical/metric-contract.json`.
