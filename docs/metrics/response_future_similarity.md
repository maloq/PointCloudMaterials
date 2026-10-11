# Similarity of future outcomes in frozen response representations

This diagnostic uses all twelve completed models from the float32 CUDA MEAM
0.1/0.5 ps velocity-response campaign, with its fixed Al64 release and original
30 test ancestors. Each ancestor has one unrelaxed 80-atom observation and four
sampled initial velocity fields. No model, normalization or retrieval parameter
is fitted using these held-out observations. No additional simulations or
scientific training runs are created. The current nonlinear prediction head is
an evaluation space, not a newly fitted readout.

Inputs and selected checkpoints are pinned to the campaign binding and hashes.
Extraction imports the frozen training implementation. Its predictions must
reproduce the recorded held-out predictions within 2e-4 absolute error. The
Q-only model is evaluated once per geometry and its embedding repeated across
the four velocity fields, avoiding numerical tie-breaking among identical inputs.
QV is evaluated on all 120 saved position/velocity inputs. Positions, velocities,
embedding and inference arithmetic are float32. Metric arithmetic is float64 on
CPU. No temperature, simulation age, explicit time covariates, or new relaxation
enters the encoder or a fitted readout.

## Observations, target distances and noise

The phase_state view has 120 queries and 116 candidates per query: all four
observations from the query's own source are excluded. The geometry_mixture
view averages embeddings and outcome-feature means over the four recorded
velocity fields; it has 30 queries and 29 candidates. This is the empirical
mixture of four fixed fields, not an exact Maxwell-velocity marginal. It does
not subtract between-velocity variation as sampling noise.

Each test phase state has B=16 independent thermostat shots. For each feature,
m_i=mean_b phi_ib and v_i=sample_variance_b(phi_ib)/B. The corrected distance is

    D_ij = mean_features((m_i-m_j)^2 - v_i - v_j).

For the four-field mixture, m is the mean of the four conditional means and
v is sum_field(v_field)/16, accounting only for the independent shooting noise.
All negative estimates are retained. Mean off-source distance must be positive;
otherwise the noise-floor failure is explicit.

Targets are the raw 128 fixed Fourier features at 100 fs, the 128 at 500 fs,
and their equal-width concatenation. Each Fourier block combines absolute
future structural observables and change from the initial structure. The
concatenation compares two marginal feature means, not a joint trajectory law.
These diagnostics describe a finite distribution representation, not all
aspects of the physical future. Initial velocities outside the local80 input
also vary across the teacher's full-cell velocity fields.

Native embedding distance is mean squared Euclidean distance in z128, without
whitening or test-fitted scale changes. Prediction-head distance reverses the
recorded train-only per-feature standardization before comparing the relevant
Fourier block; its common translation is immaterial. The descriptor control
uses the existing 24 radial/angular observables of the same local80 positions,
standardized by the earlier recorded training-only normalizer. It has no fit.

## Scores and independent-shot reference

spearman: compute Spearman rank correlation between representation distances
and corrected full-bank future distances over the eligible candidates for
each query, then average over queries. Ranks use average ties. A constant
ranking fails explicitly instead of silently substituting zero.

For k=1 and k=5, neighbor weights are one below the kth distance and fractional
at the boundary, so each row has weight k. Thus Q-only ties across a candidate
geometry's four velocities are evaluated in expectation over uniform tie
breaking, rather than choosing an arbitrary field. The same treatment applies
to target-distance ties.

future_distance{k}: mean corrected future distance to the k neighbors selected
using representation distance. random_distance{k}: mean over all eligible
query/candidate pairs. gain{k}=1-future_distance{k}/random_distance{k}.
Positive gain means closer future outcomes than uniform random retrieval;
negative gain is worse. This is a relative distance reduction, not a class
accuracy or a percentage of mutual information. The representation never uses
held-out future shots to select its neighbors.

recall{k}: expected overlap with the future neighbors estimated separately
from each eight-shot half, averaged across the halves and queries, divided by
k. Noise makes these empirical neighbor identities uncertain. Expected random
overlap is k/116 for phase states and k/29 for mixtures.

The split_shot_reference selects neighbors using even-indexed shots and scores
their corrected future distance on odd-indexed shots, then swaps the halves.
Its random distance averages the two half-bank distance matrices; its recall
is the overlap of their respective future neighbors. Its spearman is their
per-query distance-rank correlation. This reference exposes finite-shot
reliability and avoids selecting and scoring on the same target noise; it is
not a formal ceiling or a trained-model competitor.

negative_target_distance_fraction describes the full-bank eligible pairs.
queries, independent_sources and candidates_per_query document the population.
Models/control/reference and spaces are separate rows, never extra samples.

## Aggregation and uncertainty

Summary means average the three predetermined training seeds. seed_sd is their
sample SD (blank for a single control); it is not a confidence interval or
three new datasets. Paired contrasts subtract before from after within the
same seed, then average those differences. All reported score differences
use the original units (multiply gain/recall by 100 for percentage points).

For each source ancestor, remove its observations from BOTH queries and
candidate galleries and recompute all rankings and scores. Jackknife SE is
sqrt((n-1)/n * sum((theta_minus_i-mean(theta_minus))^2)), n=30.
The reported approximate interval is the full-data estimate +/-1.96*SE.
For a seed-averaged or paired contrast, first average the matched deletion
estimates across seeds, then apply this formula. These intervals describe
source sensitivity conditional on the fitted models; coverage is approximate
for nonsmooth nearest-neighbor scores. They are not clipped to feasible ranges.
Velocity fields and the many dependent source pairs never inflate n beyond 30.

CSV exports use this metric family with implementation hashes. Derived
representations, distance matrices, deletion estimates, selected checkpoint
hashes and the frozen campaign input contract remain with the analysis.
The scientific training runs receive final diagnostic summaries through their
existing W&B IDs; no evaluation run or new training history is created.
