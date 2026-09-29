# Information in crystal-free liquid geometry

This protocol compares distance prediction on the frozen Al64 source/ancestry
release. The sealed dense cache identity is recorded in the configuration and
checked at load. Metadata replay preserves original row IDs and all held-out rows.
Query atoms are outside established crystal and a crystal exists elsewhere.
The primary population excludes established crystal in every consumed patch.
The separate positive control requires crystal in at least one patch. Crystal
absence and interior cases are outside this conditional study, not infinite-distance
negative examples. No label metadata, temperature, simulation age or time is an input.

Population weights start from half fixed-at-risk and half uniform proposals, with
equal source mass within each half. Uniform denominators include visibility-rejected
candidates. Condition once on the study population and renormalize within each role.
Nested training-source subsets further condition these probabilities; held-out
roles and rows remain identical. Source IDs have unique melt ancestries, checked
before treating sources as independent bootstrap units. One seed is used.

**Targets and score.** Use saved nearest-established-crystal distance in Å, verify
equality to the inherited interface target on clear rows, and replay against the
original positions and past-confirmed PTM lineage. Distance is right censored at
64 Å. All predictors return the same 25-component zero-inflated lognormal mixture
and minimize distance negative log likelihood. No angular loss, AP objective,
reconstruction pretraining, or proximity term enters the selector. VCReg is retained
for joint encoder fits and excluded from selection. Proximity scores at 20/32/48 Å
are evaluation diagnostics. Select the smallest full-validation distance NLL from
epoch 1 onward; retain every nominal epoch checkpoint.

**Tables.** `scores.csv` reports weighted distance NLL, RMSE between expected
`min(D,64)` and `min(target,64)`, plus Brier scores, target prevalences and mean
probabilities for D≤20/32/48 Å. `reliability.csv` uses ten fixed probability bins
[0,.1), …, [.9,1] and reports weighted mass, predicted probability and empirical
frequency. Empty bins have undefined values. Results cover train, selection,
calibration and test, without fitting on calibration/test. Standard scores also
cover original fixed all64 rows and predeclared distance/clearance subgroups.

**Clearance.** On each current snapshot, reconstruct every cached patch center and
query the minimum periodic distance from its actually consumed radius-8-Å atoms
to any established crystal atom. Take the minimum over all 25 patches. This is
label-side analysis only. Replay must match saved crystal visibility (0.002 Å
tolerance accommodates reconstruction rounding). Also report query distance >32 Å,
outside the entire observation envelope. The nearest-80 candidate restriction and
holes between patches are retained; no claim is made that a full 32-Å ball is seen.

**Descriptors.** Each patch provides the existing 32 smooth radial/count/bond-power
features: 24 radial Gaussians, weighted counts at 5/8 Å, and squared contracted
spherical-harmonic fields l=2,4,6 at 5/8 Å. Identical producers process every patch.
The descriptor predictor receives means and standard deviations in three radial
patch-center regions [0,4], (4,14], (14,24] Å and norms of the 32 spatial descriptor
gradients, totaling 224 rotation invariants. Feature normalization is fitted only
on each arm's training population. No PTM labels enter descriptors.

`distance-profiles.csv` reports local-patch physical features standardized by
training means/standard deviations, by target-distance bin. Within-snapshot means
are removed using the declared weights for the analysis population. The centered
values never become predictor inputs. `within-snapshot-associations.csv` reports
correlation of centered descriptors and centered capped distance, plus 2000-draw
paired source-bootstrap intervals. Familywise bounds apply Bonferroni across the
32 descriptors within a role. Each bootstrap sums source cross-products and variances,
preserving within-source correlation. Profiles are observational associations,
not evidence that proximity caused liquid restructuring.

**Paired comparisons.** Match predictions by exact original row index. Positive
NLL gain means reference NLL minus model NLL. RMSE reduction is
`1 - sqrt(weighted_model_MSE / weighted_reference_MSE)`. Resample independent sources
with replacement 2000 times, carrying each source's score sums and conditional mass.
Report ordinary 95% intervals and Bonferroni intervals across all non-prior arms
within each population/subset/role. Subgroups remain exploratory. The predeclared
practical threshold is 2% relative RMSE reduction. A simultaneous one-sided upper
bound below .02 excludes that benefit for the tested fitted model; it does not
bound the information available to every possible predictor. A single seed does
not measure training-seed uncertainty. The existing benchmark has informed earlier
development, so independent confirmation is still desirable.

**Learning curves and controls.** Descriptor MLP and scratch MACE use nested
25/50/100% eligible training-source sets, identical seed and update budget. The
parent-initialized MACE and frozen-parent context readout form a separate full-data
pair; they are not used to claim source-naive small-data learning. Parent exposure
is the historical supervised snapshot fit, not this study's labels. The visible
MACE control and its own no-input reference use a different population and must
not be compared to clear-input models by raw scores. The no-input reference is a
trainable 25-component distribution, rather than a single lognormal.

All metric CSV exports freeze this document and implementation hashes using
`snapshot_metric_docs`. Predictions, checkpoints, exact subset IDs, training logs
and configuration identities remain available. Frozen probes, descriptor and
distribution controls stay local; scientific encoder fits remain online in W&B.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.
