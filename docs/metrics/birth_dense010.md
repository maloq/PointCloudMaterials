# Native dense Al appearance prediction

This separate exploratory assay tests whether six observations spaced by exactly
0.10 ps improve appearance prediction over a single frame. It searches new birth
labels in 24 completed Al dense descendants, sources 860 through 883, all at
520 K. It never copies events from their older 0.75 ps ancestors. The 228-event
relaxed comparison used only the older 0.75 ps trajectories; the shooting search
used 0.30 ps. None of those prediction events supplies a native 0.10 ps example.
No trajectories are interpolated and no new simulation is launched here.

## Native search and event eligibility

Read the verified float16 original-MD trajectories and actual integer timelines
using the existing trajectory producer. All 6001 observations per source are
spaced by 0.10 ps, from 0 to 600 ps. Compute full-cell PTM with RMSD cutoff 0.1;
FCC, HCP and BCC count as crystalline. Track components of at least four atoms
with 3.6 Angstrom connectivity, at least four shared atoms, overlap fraction
0.25 and at most one missing observation. That gap is 0.10 ps here; the old
protocols had different physical gap allowances and remain separate.

Reuse the established ancestry/fate producer. Establishment requires 64 atoms
for 1.5 ps, namely 16 consecutive observations. Parent-present embryos,
left-censored crystals, interfaces within 8 Angstrom, and ambiguous lineages
remain separate from new isolated establishments. The failed-episode pool
requires peak size at least eight, an isolated single-root strong lineage and
atom-verified disappearance. The stronger subset must sustain eight atoms for
at least 0.75 ps, rounded up to 0.8 ps on this grid. Dissolution confirmation
rounds 2.25 ps up to 2.3 ps; growth persistence is exactly 1.5 ps. Candidate
counts and eligibility counts are distinct. Multiple centers are not new nuclei.

For each candidate, sample at most four centers within 8 Angstrom of its
retrospective centroid, scanning at most 64 seeded choices. Search backwards
up to 24 ps for the first locally observed crystal and require that crystal to
overlap the candidate core. A crystal already present at the search boundary
is left censored. Require no crystalline atom within 8 Angstrom in any of the
six native input observations before appearance. Take four same-source,
same-time liquid controls per positive center. Controls remain crystal free
from the input start through event confirmation/disappearance and at least
6 ps after the input endpoint. Insufficient follow-up excludes that center.
The target is any eligible crystal appearance, not establishment or survival.

This retrospective case/control sampling can expose persistent structural site
differences. It is not a prospective onset alarm or a population nucleation
rate. More frequent observations can also expose short-lived PTM flicker that
was invisible on the old grid; neither grid proves a thermodynamic nucleus.

## Model inputs and controls

No encoder is trained. The predictor consumes the existing 442 descriptors:
99 geometry, 40 bond order, 45 CNA and 258 TDA. They are computed from the nearest
80 center-relative atoms with periodic minimum images; the descriptor producer
uses only atoms at radius below 8 Angstrom. Inputs are original MD, with no
relaxation, velocities, species, conditions, temperature, absolute time, age,
source identity, event label, fate, future geometry or teacher. Timestamps
organize input observations and labels only. Each fit saves its input contract.

All four treatments use exactly the same eligible rows and folds:

- Snapshot uses the observation at minus 0.1 ps.
- Early snapshot uses minus 0.6 ps.
- History uses minus 0.6, 0.5, 0.4, 0.3, 0.2 and 0.1 ps relative to appearance,
  in chronological order, spanning 0.5 ps.
- Repeated control copies the minus 0.1 ps descriptors six times before fitting.

History and repeated control concatenate chronological descriptors, their
mean, standard deviation and last-minus-first, giving 3978 input values. Each
snapshot has 442. Repeated controls match input layout, not effective degrees
of freedom; duplicate features can change logistic regularization. Both snapshot
and repeated controls are needed. No PTM labels enter feature construction.

## Fitting and uncertainty

One seed, five outer ancestry folds and five inner selection folds. Use the
existing greedy event/row balancing, without conditions, geometry or scores.
All descendants of a preparation ancestor stay together. Original source roles
remain metadata; this explicitly new exploratory CV does not change the frozen
Al64 benchmark or claim an untouched confirmation set. Require both classes
in every fitting/selection/outer partition and enough eligible ancestors;
otherwise stop with coverage counts rather than silently changing the protocol.

Logistic regression uses C in [0.001,0.01,0.1,1,10], with scaling fit only on
the current fitting subset. CatBoost uses depth 4, learning rate 0.04, L2 10
and up to 1000 trees. Select C or tree count by pooled inner weighted binary
NLL and refit the outer training partition. No outer data select thresholds,
features or hyperparameters. No AP selection, calibration fitting or ensemble.
Two models and four treatments yield 40 final outer fits; descriptor controls
stay local and do not create W&B runs.

Each source/event has total weight one across all its positive centers and
their four controls. Weighted positive prevalence is therefore 20%. Report
weighted held-out NLL, Brier, AP, AUROC, matched AUC and matched probability gap
using the retained producer. Clip probabilities to [1e-7,1-1e-7] for scores.
Matched AUC is the fraction of four controls scored below their paired positive,
with half credit for ties. Report mean outer training NLL/Brier, per-fold
resubstitution/inner-selection/outer errors and ten-bin reliability tables.
Per-fit role merged_test means the outer holdout in this protocol.

Require exactly one outer-held-out prediction per row per treatment. A prior
fit to each outer training fold supplies the baseline. Report combined,
established and failed-event strata, preserving all matched controls. Pair
history against snapshot, repeated and early snapshot, and each model against
its training prior. Improvements have negative NLL/Brier differences.

Resample whole ancestors 2000 times with identical draws for each paired
comparison. Report percentile 95% intervals; these condition on fitted outer
predictions and omit training-seed and refitting uncertainty. Save prediction
hashes, selected models, fold membership, source/exclusion counts and exact
producer hashes alongside metric tables. Historical exports are unchanged.
