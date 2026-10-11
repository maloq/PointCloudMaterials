# Al shooting appearance and nucleus fate prediction

This exploratory extension adds eligible Al shooting events to the earlier
appearance cohort. It preserves all historical results and fixed dataset roles
as metadata. Ta is excluded. No MD, relaxation or encoder fitting is performed.
Descriptor readouts remain local diagnostics and do not create W&B runs.

## Population and observations

The completed shooting search supplies 77 isolated established births and 30
isolated brief dissolved candidates across 92 branches. Among the 77 births,
31 have documented continued growth and 46 remain right censored. These are
screening candidates, not guaranteed predictive samples or independent events.
Branch siblings and all parent configurations from one preparation ancestor
stay together. Parent-present embryo growth, existing crystals, interface
events and repeated continuation segments are excluded from this cohort.

Select at most four centers per event, within 8 Angstrom of its retrospective
centroid. Scan at most 64 centers using the retained seed. Require the first
observed local crystalline atom to overlap the candidate core, then retain
only histories whose 8 Angstrom surroundings are crystal free through all
native observations before that appearance. Coordinates use nearest 80 atoms,
centered and minimum-image wrapped. The shared descriptor producer subsequently
consumes only atoms at radius less than 8 Angstrom. Retrospective site selection
is explicit: this is not a prospective population-wide nucleation alarm.

Four actual observation offsets are −6, −4.5, −3, −1.5 ps relative to first local
PTM crystal appearance. Use earlier frame slots [0,2,4,6] on the 0.75 ps grid
and shooting offsets [-20,-15,-10,-5] on the exact 0.30 ps grid. No interpolation.
Snapshot uses the last observation; history uses chronological values, mean,
standard deviation and last-minus-first. Both collections therefore have the
same physical input spacing, span and nominal appearance lead. Label detection
still differs at 0.75 versus 0.30 ps, so comparisons include label resolution
and campaign differences. No claim of identical outcome-observation protocols.

Each appearance center has four controls from the same source and times, with
no local PTM crystal through confirmation/disappearance and at least 6.75 ps
after the input endpoint. Shooting rounds this minimum upward to 6.9 ps on its
native grid. Source/event weight totals one across case and controls; the
case-control prevalence is 20%, not natural nucleation incidence. The earlier
pool includes both established and eligible failed events, rather than only
the historical established-birth treatment.

The 442 input descriptors contain 99 geometry, 40 bond-order, 45 CNA and 258 TDA
values. No velocity, species, material ID, temperature, absolute time, simulation
age, label, fate, source identity or future geometry enters the feature matrix.
The encoder is absent. Inputs are unrelaxed original MD; there is no teacher.
Native timestamps organize observations and labels only. Al length scale is one.

## Fate labels

Fate classification keeps only positive appearance examples, removing every
liquid control. Positive means verified dissolution, including grew-then-dissolved;
negative means the existing audit documents continued growth. Right censoring,
unresolved merging and unresolved disappearance are excluded from this binary
task, with their counts exported. Establishment alone never supplies a growth
label. Each retained nucleus event has total weight one regardless of its number
of centers. Fate labels are read from hashed audit artifacts, never recomputed
from a proxy feature or establishment status.

The frozen growth definition requires an exclusive lineage to reach at least
128 atoms and twice its confirmation size, with sustained growth for 1.5 ps.
The original audit uses three 0.75 ps observations; the shooting audit uses six
0.30 ps observations. Native disappearance verification and available follow-up
remain collection-specific. Continued growth is an observed outcome, not proof
of permanent survival. This resolved-case comparison is not an eventual-fate or
fixed-horizon survival model; incomplete follow-up can select which cases enter.
Snapshot and history end before any locally observed crystal for both outcomes.

## Fitting and held-out evaluation

Keep the earlier five outer ancestor assignments unchanged; assign eligible
new shooting ancestors to five balanced folds using event/row counts and one
seed. Shared parents must map to one ancestor. No features, conditions or scores
choose folds. Within each outer training partition and each task/training pool,
construct five inner ancestry folds and require both labels in every training
and selection subset. Every outer holdout must contain both labels globally.

Appearance compares training on earlier data, shooting data, and their union.
Each predicts the same outer-held-out rows from both collections. Fate fits the
combined resolved pool and reports earlier, shooting and combined evaluation.
Two readouts and two observation types give 60 appearance and 20 fate outer fits.

Reuse the declared logistic C grid [0.001,0.01,0.1,1,10] and CatBoost depth4,
learning rate0.04, L2=10, up to1000 trees. Select C/tree count by pooled inner
weighted binary NLL, then refit all outer training rows. Scaling is learned only
from each fitting subset. Outer rows never select models, thresholds or scaling.
No AP optimization, calibration fitting or model averaging. One retained seed.
The 95th-percentile inner-OOF negative threshold is a secondary training-only
diagnostic in per-fit exports. Per-fit role `merged_test` means outer holdout
in this new protocol; historical fixed-test predictions are not reused.

## Scores and uncertainty

Require exactly one outer-held-out prediction for every retained row in each
treatment. Report weighted binary NLL, Brier, AP, AUROC and prevalence. The shared
`fit.scores` clips probabilities to [1e-7,1−1e-7]. AP/AUROC are undefined when an
evaluation population has only one class. Report mean outer resubstitution
train NLL/Brier and per-fold train, inner-selection and outer-test scores.
Training means are arithmetic fold averages for the fitted treatment, not
evaluation-subpopulation-specific errors. Selected inner scores are optimistic.

An outer-training weighted class prior supplies a held-out reference per fold.
Export model-minus-prior, history-minus-snapshot and combined-minus-single-pool
paired differences; NLL/Brier improvements are negative differences. Ten fixed
probability bins report weighted predicted mean and observed positive fraction.
Appearance also reports matched AUC/probability gap within each 1:4 matched set.
Fate has no liquid controls and therefore no matched-set score.

Resample whole preparation ancestors 2000 times using identical draws for
paired comparisons. Percentile95% intervals are conditional on the fitted OOF
predictions, not training-seed/refitting uncertainty or independent CV repeats.
Fate bootstrap draws that lose a class are omitted and their valid count is
reported; a one-class subpopulation has no two-class bootstrap interval.
Historical sources have already been inspected; this is exploratory CV.

All metric tables carry this frozen definition and implementation hashes.
Coverage exports retain candidate rejection reasons and resolved/censored labels.
Per-fit bindings preserve input-array hashes, exact folds, original source roles,
training pool, selectors, models and predicted probabilities.
