# Interim results, 2026-10-18

All 42 training runs completed: 30 predictors at 32,000 updates and 12 native
encoders at 48 epochs. All 30 convergence/conditioning evaluations, 33 native
representation evaluations and four shared control bundles completed. The first
full-forecast execution completed only two of 30 arms; the remaining 28 encountered
an audit implementation memory error. Full forecasts are being re-exported in
an independent execution revision with unchanged models, samples and definitions.
See [recovery details](../../docs/ni_latent_validation.md).

## Connected trajectories: useful improvement on the fixed diagnostic panel

The following values are equal-seed means of source-balanced scores on **134 test
windows from 62 ancestors**, with 32 neural samples per origin and the fixed
32-point flow sampler. They are not results on all 14,880 test windows. Lower
path-energy scores are better. The 3,200-update checkpoints are early snapshots
within the new 32,000-update schedule, not the old independently completed
3,200-update runs; the learning-rate schedules differ from those historical runs.

| Predictor | 3,200-update path energy | 32,000-update path energy | Final temporal variance / saved MD |
| --- | ---: | ---: | ---: |
| Direct measurements, block flow | 0.6601 | 0.6148 | 92.8% |
| Scalar-support measurements, block flow | 0.6508 | 0.6193 | 90.1% |
| Frozen VAE, block flow | 0.6380 | 0.6405 | 70.5% |
| Frozen VAE, autoregressive MDN | 0.6190 | 0.6160 | 84.9% |

The direct block model's temporal-variance ratio rose from 46.3% to 92.8%.
This is a variance ratio, not a standard-deviation ratio or a guarantee of correct
temporal correlations. Its increment-correlation energy improved from 0.2585 to
0.1754. Marginal 90% interval coverage rose from 63.4% to 81.8%, still below nominal.
The constrained block model reached 79.4% coverage. VAE block training did not
improve the path-energy score over its early checkpoint.

The direct block model scores **0.6148 with the actual history versus 0.6810
with another history from the same source/regime**, and 0.7070 for unconditional
resampling of saved training futures. Persistence scores 1.2016. Conditioning
therefore contributes useful information on this panel.

For the constrained block model, expanding from 16 to 28 training ancestors
improves path energy **0.6938 → 0.6345** (8.5%) and coverage **68.7% → 77.1%**.
Both arms use 80 original-regime windows per training ancestor and the same test
panel. Source expansion also broadens temperature coverage, so this is not a
pure sample-count effect at a fixed distribution.

Source-paired, fixed-three-seed bootstrap differences (1,000 whole-source draws):

- Direct block final minus early checkpoint: −0.04530, source CI [−0.05028, −0.04032].
- Constrained 28-source minus 16-source block: −0.05931, source CI [−0.06299, −0.05562].
- Direct block actual minus shuffled history: −0.06618, source CI [−0.09811, −0.03807].

These intervals condition on the three fitted seeds; they are not a confidence
statement over arbitrary retraining or an untouched future external cohort.
Direct MDN variants become unstable despite finite normalized samples: unconstrained
MDN distance/variance calculations overflow in all seeds, while the scalar-support
MDN has a mean path score of 36.6484 and vastly excessive fluctuations. Scalar
ranges alone do not stabilize autoregressive evolution.

## Native FMRC: conditional information, inconsistent generated measurements

FMRC's own forward conditional-flow head on the expanded 28-source fit, at 1 ps,
is evaluated on **536 test windows from 62 ancestors**. Three-seed means:

| Conditioning/reference | All-measurement energy |
| --- | ---: |
| Actual initial state | 0.6037 |
| Another state matched on current order | 0.6390 |
| Another state from the same source/regime | 0.6661 |
| Saved training-future resampling | 0.6907 |
| Persistence | 1.1829 |

Marginal 90% coverage is 89.7%. This is evidence of useful conditional future
information and reasonable pooled marginal spread. It does not establish exact
conditional-ensemble calibration or coherent multi-time paths: these are separate
1-ps future samples, not a connected 6-ps simulation.

**81.4% of generated states violate at least one of the legacy scalar measurement
checks**, versus zero for the saved-data references. The 16-source native fit has
84.7% violations and an energy score of 0.6053. The expanded FMRC sampler therefore
does not yet generate physically admissible structural measurements. Its legacy
audit is weaker than the full scalar-plus-joint audit used for trajectory forecasts.

## SelfTICA: no consistent benefit from the new expansion

For a train-fitted four-frame latent Gaussian probe predicting qbar6 at 6 ps in
present-liquid test windows, normalized MSE worsens from **0.1439 to 0.2984** when
the native SelfTICA fit grows from 16 to 28 training ancestors. Current-state latent
probe MSE similarly worsens from 0.2116 to 0.4711. The corresponding FMRC values
are 0.1352 to 0.1336 for history and 0.1377 to 0.1379 for current-state probes.
These are specific retained-information diagnostics, not nucleus-birth AP, and
the expansion changes source/environment coverage. A successful training objective
does not by itself establish a better structural or generative representation.

## Full-test validity warning, currently one constrained-flow seed

The original full test completed for `bounded_direct_block-s20261027` on all
14,880 test windows. It has no scalar-bound or nonfinite violations, but **94.18%**
of generated states violate density Gram positive-semidefiniteness and **100%**
violate the checked l=0 radial-product identity at the declared tolerance.
The scalar-support MDN seed that completed also fails the joint checks.

Thus individually plausible values do not imply a jointly realizable local
structure. We have improved statistical prediction, but have not established a
physically consistent latent simulator, connected nucleus generation, or improved
nucleus-birth prediction. No new birth-AP assay was part of this queue.

## Evidence and remaining work

- [Frozen diagnostic exports](../../output/ni_latent_flow/validation-20261018-v2/analyses/comparison/tables/diagnostics_individual.csv).
- [FMRC conditional samples](../../output/ni_latent_flow/validation-20261018-v2/analyses/comparison/tables/kinetics_samples_individual.csv).
- [Future-information probes](../../output/ni_latent_flow/validation-20261018-v2/analyses/comparison/tables/kinetics_probe_individual.csv).
- [Source-paired diagnostic receipt](../../output/ni_latent_flow/validation-20261018-v2/technical/interim-paired-diagnostics-20261018.json).
- [Completed constrained-flow validity](../../output/ni_latent_flow/validation-20261018-v2/analyses/bounded_direct_block-s20261027/tables/validity.csv).
- [Full-test recovery revision](../../output/ni_latent_flow/validation-20261018-v2-eval-r2/technical/evaluation-revision.json).

The revised forecast queue uses six GPUs, saved final checkpoints and the
original 23:35 Paris deadline. The memory fix evaluates the same independent
4x4 eigendecompositions in chunks of 1024; it does not alter samples, tolerances,
scores or training. The first failed execution and its results remain preserved.
Report the full-test forecast/ordering comparison after this revision finishes;
the completed small-panel findings above must not be relabeled as full-test results.
