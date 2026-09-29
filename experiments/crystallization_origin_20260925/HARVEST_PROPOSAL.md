# Proposed regional crystal-emergence dataset

Status: the [training-source analysis](../../docs/nucleus_harvest.md) is now
implemented; the final training/evaluation release remains a proposal. Start from the
[completed audit](RESULTS.md), without additional simulation for the first
release. Preserve Al64-v1 and its existing-arrival benchmark unchanged.

## Target and available events

Predict the first new isolated crystal establishment in an atom-centered region
within **3 ps**, with **6 ps** secondary. Separate it from arrival of a lineage
already established elsewhere. A PTM-positive central atom is allowed when its
surroundings contain only subthreshold precursors.

The fixed source roles contain 206/34/41/67 isolated establishment candidates
in train/selection/calibration/test. These are upper bounds before history,
follow-up and quality requirements. The original training grid contains zero
regional positives despite those 206 candidates.

Start with Al. Keep the million-atom Al trajectory, with 42 candidates, as a
separate size/context transfer assay, subject to provenance and pretraining-
exposure checks. One trajectory is not 42 independent simulation replicates.
Mg and Ta can follow as material-transfer tracks with branch ancestry preserved.
Ti needs label review first: 88.3% of its positive 3 ps onset windows were
unresolved in the current audit.

## Event catalogue and quality

Keep 64 atoms sustained for 1.5 ps as an operational reference, alongside the
32/128-atom and 3 ps persistence variants. Do not equate this with the physical
critical nucleus. Retain crossing time, confirmation, growth/remelting, merging,
periodic geometry and ancestry confidence. Future growth is an auxiliary
outcome, not an undocumented requirement for accepting an establishment.

Review a training-only sample of establishments, transient ordered clusters,
merges and interfaces. Compare PTM against bond-orientational coherence and
averaged local order. Lechner and Dellago provide a complementary structure
discriminator, not validation of our metallic-nucleus threshold:
[original paper](https://arxiv.org/abs/0806.3345).
Size alone need not describe nucleation: Liang et al. found size, crystallinity
and chemical order relevant in Ni3Al
([original study](https://arxiv.org/abs/2004.01473)). This motivates diagnostics,
not species inputs or quantitative transfer of alloy results to pure metals.

Keep ambiguous cases explicitly flagged, never as ordinary negatives. Freeze
any revised labeling rules using training sources before exporting the new
evaluation release. Report accepted and unresolved fractions by source/type.

## Causal risk population

Predefine every tracked atom at each allowed observation origin as a possible
center. The target region follows that atom, using the existing 8 Å Al radius.
Birth locality is measured at birth relative to that atom; the predictor sees
the region centered at the atom's actual position at the forecast origin.

An eligible region has sufficient history and contains no established crystal.
Use only current components and previously confirmed lineage information. For
an initial clean pre-crossing task, exclude a region already containing a
component of at least 64 atoms, even if persistence is not yet confirmed. Allow
smaller ordered clusters and a crystalline central atom. Record analogous risk
rules for the sensitivity criteria.

The current audit backfills ancestry after future confirmation. **Do not use
that backfilled state for origin-time eligibility.** Build a causal pass with
confirmation restricted to observations at or before the origin; retrospective
lineage information is for labels only.

Distinguish the first regional transition as isolated establishment, arrival
of an external established lineage, interface-associated/ambiguous transition,
or no transition within the horizon. Arrival is a competing outcome: never
remove a window because a front arrives in its future. Keep later events in
the catalogue separately from this first-transition target.

Use the first crossing in the sustained streak as birth time, with future
persistence for confirmation only. Origins at/after that crossing do not count
as forecasts of that birth. Require full 6 ps follow-up plus 1.5 ps confirmation
for the first matched 3/6 ps release. Other windows are censored, not negatives.

## Efficient harvesting

1. Reuse full-cell PTM chunks, cluster graphs and birth catalogues. Stream dense
   causal risk labels; avoid constructing all atom-neighborhood graphs eagerly.
2. Query atoms within the target radius of each birth. Their identities identify
   positive rows of the predefined population. Retrieve ordinary neighborhoods
   around those atoms at earlier origins. Never center model inputs on a future
   cluster centroid, crop to future nucleus members, or expose future membership.
3. Sample a bounded number of positive rows per event: initially up to four
   centers and eight pre-birth origins from 0.75 to 6 ps. Add earlier lead-time
   controls where available. Randomly select within declared strata, deduplicate
   overlapping event windows, and record exact inclusion probabilities.
4. Initially sample about ten negative/competing examples per retained positive.
   Combine uniform background sampling with transient ordered embryos,
   dissolving clusters and arriving fronts. Record each stratum's sampling rate.
   Matching on current size/order is a diagnostic subset, not a restriction
   that removes those informative features from the whole training population.
5. Store source/frame/atom references and labels first. Materialize geometry
   lazily with the chosen encoder's full spatial halo. Reuse surrounding-patch
   geometry for vector-message and harmonic-hierarchy predictors. Keep caches
   outside the repository.

The cap gives an upper budget of 206 × 4 × 8 = **6,592 six-ps positive windows**
before filtering. This is not a yield forecast. Multiple crops/lead times do
not multiply independent births; report events and source groups separately.

Archive up to 12 ps of observed history before each origin where available.
Preserve native saved cadence: 0.75 ps for the main cohort, and the separate
0.5 ps external track. Do not interpolate or silently discard early births
needing shorter history. Record eligibility by history length and use matched
rows for snapshot/history comparisons.

## Splits, sampling weights and evaluation

Keep the original 90/15/15/30 Al source roles. Every crop, lead, label variant
and shooting descendant inherits its source-root role. Version the new center
sampling, risk rules, cadence and labels as a separate population. Group
external branches by preparation ancestry and audit prior structural-training
exposure before calling a transfer assay unseen.

Event enrichment is a computational sampling design, not the target prevalence.
Retain inclusion probabilities for positives, negatives, hard-example strata
and their overlaps. Use inverse-probability weighted predictive likelihood for
the declared population. Biased case-control subsampling needs correction;
Fithian and Hastie give one scheme for logistic regression
([original paper](https://arxiv.org/abs/1306.3706)). Their analytic correction
does not directly justify arbitrary neural-classifier calibration; use our
explicit sampling probabilities.

Freeze an outcome-blind evaluation sample of centers/origins, or evaluate the
whole feasible population, identically for every model. Choose density using
training coverage diagnostics before viewing new held-out outcomes. An
additional event-enriched evaluation can measure event recall, but unweighted
AP/calibration on that sample are not population metrics.

Report predictive NLL/Brier score, calibration, AP3/AP6 as diagnostics,
event-deduplicated detection recall and lead time, and false alarms per declared
region-observation exposure. Use source-level uncertainty and a predeclared
assignment rule to merge neighboring alerts for event recall. Compare against
simple present cluster-size/order/context readouts to assess information beyond
obvious incipient ordering. No AP-based loss, selection or ensemble fitting.

Encoder inputs stay geometry-only under the fixed material normalization.
Predictor inputs are the declared patch embeddings and relative spatial geometry;
history is a separately recorded variant. Temperature, age, absolute time,
material identity and future lineage are not model covariates. Original observed
MD defines labels; relaxed-input variants cannot redefine the physical event.

## First delivery and subsequent expansion

First harvest training sources and produce a quality/coverage report: accepted
distinct births, unresolved/rejected cases, usable pre-birth leads, control
strata and storage estimate. Then freeze the protocol and export the remaining
original roles. Do not promise 206 clean positives before that pass.

If more events are needed, favor independent liquid preparations and spontaneous
trajectories for population evaluation. Conditional shooting around promising
and failing training precursors can investigate growth versus dissolution.
Those branches share a parent and do not provide an unbiased nucleation rate.
Restart from preserved full-precision integration states, not float16 analysis
coordinates. A committor study additionally needs explicit competing basins,
stopping rules and a defined velocity/noise ensemble; it is a separate protocol
from this finite-horizon dataset.
