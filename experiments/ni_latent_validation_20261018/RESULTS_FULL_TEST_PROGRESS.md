# Full-test progress, October 18, 21:18 Paris

Nineteen of 30 corrected full-forecast evaluations have completed. Every completed
evaluation uses all 14,880 test windows from 62 ancestors and 32 neural trajectories
per input. No new failure receipts were present at this check. All scientific
training and the earlier smaller diagnostic/native-method evaluations are complete.
The remaining full forecasts continue under the original deadline.

## Completed trajectory comparisons

These are source-balanced means across the first two completed seeds, except
VAE-MDN, where only seed 20261025 has completed. They are provisional comparisons,
not checkpoint or model selection. The three-seed small-panel findings remain
separately recorded in [the earlier report](RESULTS_INTERIM.md).

| State and predictor | Completed seeds | 6-ps path energy, lower better | Temporal variance / MD | Empirical interval coverage |
| --- | ---: | ---: | ---: | ---: |
| Direct244, joint block flow | 2 | 0.61994 | 90.3% | 81.2% |
| Scalar-support direct244, joint block flow | 2 | 0.62343 | 87.8% | 79.0% |
| Frozen VAE32, joint block flow | 2 | 0.64889 | 69.0% | 67.6% |
| Frozen VAE32, autoregressive MDN | 1 | 0.61900 | 85.5% | 79.4% |

Persistence scores 1.20394. Energy is a proper sample-based distance score on
normalized measured paths, not an accuracy percentage or a prediction of individual
atomic coordinates. Variance ratios refer to the common normalized measurement
geometry. Interval coverage uses the empirical 5th/95th percentiles of only 32
samples; it is not an exact calibration test of population 90% intervals. For
example, even iid Uniform draws with these interpolated empirical quantiles have
expected coverage 27.9/33, about 84.5%, rather than 90%. For general distributions
the interpolation effect depends on their quantile geometry.

The direct model keeps four observed measurement states and generates all 60
future states jointly. VAE variants first compress the same central measurements
to 32 latent components. No new MEAM integration or simulator derivative is used
by this specific queue. These forecasts follow native 0.1-ps measured MD histories.

## What fails in autoregressive rollout

On the completed three-seed diagnostic panel, bounded direct MDN has marginal
state energy 0.57560 when every next-step prediction receives the actual MD history,
but 26.04309 when it recursively receives its own predicted history. Its full-test
path scores are 21.52 and 61.21 in the first two seeds. Unbounded direct MDN has
overflowed distance/variance scores in both completed full evaluations. Finite
sample coordinates alone therefore do not imply useful or stable dynamics.

The VAE-MDN counterpart has diagnostic marginal scores 0.57663 with real histories
and 0.61023 autonomously. Compression appears to help rollout stability in this
comparison, but it also discards information: decoding the actual saved futures
through the frozen VAE already incurs path score 0.42848 and retains only 73.3%
of actual temporal variance in seed 20261025. This is an oracle reconstruction
diagnostic, not a forecast or an additive lower bound on predictive error.

## Pre-structured liquid and rare crystalline entry

For inputs initially classified as liquid, actual futures spend a source-balanced
20.90% of their time in the operational high-order-liquid bin; the direct joint
flow predicts 20.25%. The bin means high qbar6 and coherence according to frozen
training-liquid thresholds while remaining noncrystalline. It is not an established
phase or proof of a productive nucleation precursor.

Matching the mean occupancy does not identify the correct local histories. Direct
flow occupancy MSE is 0.04190, corresponding to about 20.5 percentage points RMS
per-origin error; longest-dwell MSE is 0.51052 ps², about 0.71 ps RMS. These are
source-weighted errors, not the difference between the two population means.

At 6 ps, the same present-liquid evaluation gives:

| Quantity | Direct flow | Reference |
| --- | ---: | ---: |
| Predicted local persistent-entry probability | 0.723% | Actual source-weighted event frequency 0.874% |
| Local-entry Brier score, lower better | 0.01124 | Always-no-entry/persistence 0.00874 |
| Local-entry Brier using actual future measurements | — | 0.00453 |

Thus the model's overall event frequency is reasonably close while its end-to-end
event prediction does not beat the simple negative baseline. The last row exposes
the imperfect fixed measurement-to-phase classifier even with actual futures;
its inferred event frequency is 0.471%, below the PTM-based truth. Model and
phase-readout errors both matter. Local entry means three consecutive crystalline
central-atom observations. It is not connected 32/64-atom nucleus birth, and this
queue supplies no new nucleus-birth AP result.

## Source expansion and native temporal encoders

The constrained original16/original28 block comparison now has full-test path
scores 0.70263 versus 0.63902 (9.1% improvement) and coverage 67.9% versus 76.7%,
averaged over two seeds. This corroborates the small-panel trend. Both have 80
training windows per ancestor; the added sources also change temperature coverage.

The completed three-seed native FMRC and SelfTICA diagnostics are unchanged from
the earlier report: FMRC's actual 1-ps head uses conditioning beyond a coarse
current-order match, but does not establish connected-path generation. Enlarged
SelfTICA training worsens the six-ps qbar6 history-probe MSE from 0.1439 to 0.2984;
each of the three seeds worsens. FMRC is nearly unchanged at 0.1352 versus 0.1336.

Increasing the joint-flow sampler grid from 32 to 128 points changes direct-flow
small-panel path energy only from 0.61479 to 0.61644. Merely using more numerical
sampling steps does not fix the remaining predictive weaknesses. These grid points
are algorithmic flow integration, not extra femtosecond MD observations.

## Interpretation

The strongest positive finding is useful conditional trajectory statistics without
the earlier severe variance collapse. Direct measured-state generation is a strong
baseline; the learned spatial compression has not shown a clear overall advantage.
Average ordered-liquid occupancy is recovered better than the identity and timing
of rare ordering events. Joint measurement validity remains a separate decoder
diagnostic; neither aggregate path quality nor its failure rate alone establishes
the usefulness of an encoder for nucleus prediction.

Sources: corrected per-arm `tables/trajectory.csv` under
[`validation-20261018-v2-eval-r2`](../../output/ni_latent_flow/validation-20261018-v2-eval-r2/analyses/),
and original [fixed diagnostics](../../output/ni_latent_flow/validation-20261018-v2/analyses/comparison/tables/diagnostics_individual.csv).
All figures above retain the frozen score definitions, train/test ancestry and
actual completed-seed counts.
