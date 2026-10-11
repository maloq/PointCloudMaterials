# Can a compact stochastic state preserve Ni future distributions?

The first experiment separates three hypotheses: compression to16 versus64
coordinates; memory of1 versus4 observations; and future-value learning versus
future values plus simulator responses. Each of the eight arms has two seeds
and3200 updates. The spatial backbone is the same frozen unrelaxed Small eSEN.
This makes the new learned bottleneck and recurrence identifiable without
repeating broad spatial pretraining or coupling the test to quench variability.

The64 observations in each local sequence span-0.3 to+6ps at native0.1ps.
Training rolls forward without teacher forcing for up to1ps; evaluation extends
to6ps. Structural and motion distributions, calibration, persistence controls
and original32/64-atom birth probes measure different aspects of usefulness.
Code size is reported with total memory size, not only per-frame dimension.

A small empirical-decoder diagnostic lifts codes to actual complete training
microstates. Independent fixed-box physical continuations compare these states
with held-out originals and random same-protocol controls. It cannot establish
new atomic generation or correct nucleation rates; it tests whether latent
proximity implies dynamically interchangeable original microstates.

The stored NPT process and100fs fixed-box response teacher retain their different
physical protocols. The local predictor has no temperature/time/box covariates;
any failure of closure is part of the result. No crystal loss, broad spatial
replay, AP optimization, or new neighborhood filtering.

Reproduce with `python -m src.research.ni_latent_sim.workflow submit --config
configs/ni_latent_sim/compact_memory_response_20261010.json` on a two-GPU
allocation in pointnet-torch214. Do not duplicate a recorded launch.
[Execution and data](../../docs/ni_latent_sim.md),
[exact metrics](../../docs/metrics/ni_latent_sim.md).

## Completed October 10: useful codes, inadequate autonomous dynamics

All 16 fits completed 3,200 updates, including all birth readouts, free-rollout
scores, both predeclared physical lifting diagnostics, and comparison plots.
The second worker finished at 03:37:47 CEST, 35.14 minutes after launch. The
8.5-hour limit was a ceiling, not the measured runtime: frozen spatial features
and existing physical response teachers made these small-network fits fast.
Both workers exited; no continuation was scheduled.

Results live in
`output/ni_latent_sim/compact-memory-response-20261010-v1/analyses/`.
The comparison tables preserve individual seeds and ancestor-bootstrap intervals.

Nucleus-64 AP below is the mean of two seeds. These are held-out, event-matched
localization probes on the compact input codes, not nucleation scores from
generated trajectories. Each lead includes 49 events from 22 held-out ancestors,
980 rows, and 0.20 positive prevalence. Training/selection used likelihood,
not AP. Four observations span 0.3 ps of actual history.

| Code / observations | Value, 3 ps | Response, 3 ps | Value, 6 ps | Response, 6 ps |
| --- | ---: | ---: | ---: | ---: |
| 16 / 1 | 0.779 | 0.776 | 0.709 | 0.696 |
| 16 / 4 | 0.788 | 0.787 | 0.712 | 0.721 |
| 64 / 1 | 0.778 | 0.765 | 0.693 | 0.710 |
| 64 / 4 | 0.774 | 0.781 | 0.731 | 0.713 |

Derivative supervision has no consistent AP benefit; individual-source bootstrap
intervals are wide. These inputs also include velocity features and descriptors,
so comparison against an earlier spatial-only model is not a matched attribution
of improvement to the new dynamics objective.

At 1 ps, structural-feature MSE is approximately 0.932–0.935 across settings,
versus 1.000 for the constant training-mean prediction and 1.734 for persistence.
The training-mean control was calculated from the existing `data.pt`: for each
held-out ancestor, average squared standardized target coordinates, then average
ancestors equally. Standardization uses the stored training-only mean and scale.
Thus the gain over the training mean is about 6–7%, substantially smaller than
the apparent gain over persistence. Descriptor MSE is approximately 0.704–0.708,
versus 0.817 for its training-mean control.

The 6-ps free rollouts are not reliable. Saved per-observation scores show severe
seed dependence: for d16/history4 at seed 20261010, structural MSE exceeds 100
in 78.47% of value-only inputs and 85.03% of response inputs. Both corresponding
second-seed runs have median MSE near 1.05. In d64/history4, rare extreme errors
can dominate mean MSE even while mixture NLL looks reasonable: components that
remain near the truth can hide escaping particles in the likelihood. These
diagnostics use `technical/rollout-scores.pt` without rerunning any model.
Training covered 1 ps; the 6-ps evaluation extrapolates an unconstrained stochastic
recurrence. This is a plausible instability mechanism, not an isolated causal test.

Held-out short-physical-response error is also poor. Group-weighted values across
the 200 held-out teacher groups are 0.520–0.531 for value-only models and
0.528–0.544 for response-supervised models. Predicting zero sensitivity gives
exactly 0.500 under the recorded half-squared-error normalization (the denominator
floor is inactive here). The response objective has therefore not demonstrated
generalizing simulator sensitivity in this experiment.

Empirical lifting remains exploratory: on eight queries, at 0.5 ps nearest-code
retrieval improves structural energy distance by about 4.7–5.8% over random
same-temperature retrieval; descriptor and 0.1-ps results are mixed. Retrieved
cells also have more similar volumes, leaving a physical confound. This does not
validate a learned atomic decoder or generated nucleation rates.

The next useful experiment should address rollout stability, validate sensitivity
learning against its zero-response control, and compare against a matched static
compression control. More birth AP alone would not resolve the central dynamics
hypothesis. No additional training was launched during this results review.

The subsequently requested [derivative stability factorial](DERIVATIVE_STABILITY.md)
is complete. It shows modest1ps-to6ps extrapolation gains from Jacobian control,
little extra gain with6ps training, and no successful held-out physical-sensitivity
learning. Its fixed-start controls also avoid the original large divergences,
so those differences cannot be attributed solely to the derivative penalty.
