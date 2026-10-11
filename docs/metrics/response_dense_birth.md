# Native 0.10 ps nucleus-birth transfer

This is a new derived event-centered track of fixed Al64 ancestors, not a
resplit or replacement of the all64 uniform/legacy16 population. Inputs come
from complete main dense production trajectories in the three pinned v3
lineage releases: the October 5 all-cadence rerun, October 5 additions and
October 6 additions. Use only each trajectory's native 0.10 ps PTM view and
`origin=original`; historical partial views, repeated cadence views, shooting,
million-atom/external, recovery, diagnostic and relaxed data are not pooled.
All selected input identities are unique. Exact integer-frame timelines,
original melt ancestry and train/selection/calibration/test roles are checked
against the fixed Al64 release and the response encoder's actual parent records.
New labels may arrive while source preparation runs; fitting waits for every
selected source and all three final catalogue completion receipts. Missing
labels are never converted to negatives. A failure is not an empty source.

## Outcome and matched rows

A case is a v3 root `emergence_from_liquid` whose own family later sustains at
least 64 connected PTM-crystalline atoms for 1.5 ps. It must have its own
`establishment_confirmation_ps`, not just a descendant/composite confirmation.
The reference time is the first eight-atom crossing, not the later 64-atom
crossing. This retains the v3 minor-absorption/fragmentation handling and
excludes inherited crystals, interfaces, unresolved origins and composite
births. These are operational persistent establishments, not proven critical
nuclei or natural nucleation hazards. Time until establishment remains metadata;
it is not fixed to the response simulator's 0.5 ps training horizon.

Deterministically sample at most four central atoms from the first eight-atom
crossing's actual member IDs. Each atom is tracked backward to exact snapshots
0.1, 0.5 and 1.0 ps before that crossing. At all three observations there must
be no PTM component of at least eight atoms within 8 A of the center. Smaller
precursors and isolated crystalline atoms are allowed. Do not call these inputs
fully crystal-free. The same eligible cases and controls are used at all leads.
There is no history input or interpolation; lead is not a model feature.

Each case has four distinct matched controls from the same trajectory and
frames. Draw 1024 candidate atoms without replacement; exclude atoms within
16 A of the event centroid at onset and require no eight-atom component within
8 A at EVERY native observation from 1 ps before onset through 6 ps afterward.
Require complete follow-up. A control may develop a small precursor or a much
later nucleus; its label is not "never crystallizes". Reject pairs with fewer
than four eligible controls and record the reason. Deduplicate center/onset
observations within a source. No threshold is relaxed to meet minimum counts.
Cases and controls are retrospectively selected, with artificial prevalence
1/5. A fitted probability is a case/control probability, not population risk.
Future core/centroid information selects measurement locations only; neither
future coordinates nor component membership enters encoder inputs.

Weight each row by the inverse number of rows belonging to its event. Thus
each event has equal total weight; multiple centers, controls, frames, seeds
and related trajectories do not become independent ancestors. Minimum eligible
event counts are 5/2/2/3 for train/selection/calibration/test. A shortfall fails
with coverage saved; it does not trigger a source resplit.

## Frozen inputs and readouts

All twelve selected native-V, 1000 fs BAOAB damping, float64-integration response
models are frozen. Q receives original centered nearest80 coordinates with its
8 A mask and 5 A graph edges. QV additionally receives actual simultaneous
velocities of exactly those atom identities, converted A/ps to A/fs, with the
model's existing center-relative normalization. No redraws or finite differences.
Network inference remains float32; archived MD Q/V quantization remains float16.
Record raw manifest/velocity metadata and hash extracted arrays. No temperature,
age, time, source, ancestry, event or material metadata enters model tensors.
The independent eight-input replay uses float32 model inputs, matching the
original response trainer, and must reproduce predictions within 2e-4.

Compare the 128D embedding and existing nonlinear 256D future-feature head.
For each lead and space fit CatBoost and logistic diagnostic readouts. All
12 models, 2 spaces, 3 leads and 2 readouts produce 144 fits. The nonlinear
readout is primary; linear is an information diagnostic, not a requirement
that nonlinear pretraining create a linearly sufficient representation.
Structural controls fit the same two readouts on the established 442 current
geometry/bond-order/CNA/TDA descriptors plus two current component statistics:
fraction of nearest80 atoms in a tracked PTM component (minimum size4), and
largest component size represented among those80 atoms, zero for liquid.
The latter accesses full-cell connectivity and is explicitly a stronger
context control, not an equal-support encoder comparison. Three lead-specific
constant training-prior controls bring the total to 153 fits. No supervised
encoder fine-tuning, crystal-specific pretraining loss or new MD is performed.

Readouts preserve the established recipe: CatBoost depth4, up to1000 iterations,
learning rate .04, L2=10, Logloss selection/early stopping with patience100;
logistic C [.001,.01,.1,1,10] selected by validation NLL. Standardization is
training-weighted, with SD floor1e-4. All training sample weights are normalized
to mean1. Readout seed20261006 is fixed, not selected. Separate calibration
sources fit the existing monotone affine logit calibration with regularization
1e-4, slope [.01,20], intercept [-20,20]. The alarm threshold is the weighted
95th percentile of calibration-negative probabilities; use strict greater-than.
AP never selects checkpoints, fits, hyperparameters or ensembles.

## Scores and uncertainty

Export raw and calibrated NLL, Brier, weighted AUROC and AP. NLL/Brier are the
primary proper scores. Calibration-locked recall and achieved FPR are reported
for calibrated predictions. Probability clipping is [1e-7,1-1e-7]. Within-pair
AUC compares the case with four controls, awarding half for ties. Within-pair
top1 awards the reciprocal number tied for maximum if the case is at maximum,
otherwise zero; chance0.2. Give each pair score to its five rows and apply event
weights. Same formulas as the earlier response-birth assay; no old predictions
are reused on the new population.

Use 2000 bootstrap draws of entire held-out melt ancestors, with the SAME draws
for all fitted models and contrasts. Related trajectories/events move together.
Report mean per-model metrics across the three pretraining seeds, seed SD and
95% percentile intervals conditional on the fitted readouts. These are not
prediction ensembles. Paired differences compare response versus value-only
and QV versus Q. Report actual rows/events/ancestors, not inflated seed counts.
Descriptor/prior controls retain one fixed readout seed. No assertion of
significance follows from point estimates alone. Existing test ancestors have
been inspected in previous studies, so this is exploratory transfer evidence.

Scientific source, config, labels, cohort arrays, selected checkpoints and
exports are checksum-bound. Feature caches use the shared six-entry leased
retention policy. Diagnostic fits remain local; final associated metrics update
the existing twelve W&B training runs, with no new runs. The detached evaluation
has a nine-hour bound including label wait, preparation and fitting; partial
outputs are resumable and never reported as a completed comparison.

## All 0.10 ps records with saved velocities, October 6 scope revision

At the user's request, `all010-birth-v2` expands the earlier83 main production
trajectories to every exact0.10ps PTM view in the three v3 releases with saved
velocities. Do not restrict by dataset prefix, historical-partial status or
production date. The user explicitly permits skipping million-atom data without
velocities. Other coordinate-only archives are also listed as `no_saved_velocities`
for this matched Q/QV comparison; do not inject synthetic/zero/estimated motion.
The inspected100 views include96 native0.10ps records and4 exact stride-ten
views of native0.01ps histories. Eligible92 records have velocities and known
fixed Al64 melt roles; the8 unavailable-V records comprise the million-atom
continuous run, six older million-atom EAM branches and the older100k run.
Counts include historical overlapping views, not92 independent trajectories.
No interpolation occurs and finer histories are never relabeled native0.10ps.
Other unlabelled collections remain outside these three immutable releases.

The outcome, three leads, controls, neural inputs and frozen readout protocol
above remain unchanged. For finer histories, map event first_frame into the
exact0.10ps view, shift in that grid, then gather the source's original frame
and actual velocities. Control screening also advances through that exact grid.
Persist actual native cadence and0.10ps label cadence on each row. An available
coordinate-only cohort is not a negative cohort. Missing velocities exclude a
whole source view before Q/QV comparison, never model-specific evaluation rows.

Reuse completed earlier source preparations only after checking the old plan,
source mapping, label identity, observation recipe, source receipt and every
extracted-file checksum. Link immutable coordinate/velocity/descriptor arrays;
write new row metadata and alias receipts. Historical preparation remains intact.

Canonical physical-event IDs bind melt ancestor, onset time, original first
member atom IDs, full periodic onset Q/V and box. Exact duplicate matched cases
with the same physical event, central atom and all three Q/V input patches are
included once; remove their entire repeated five-row matched set and publish
an alias table. Prefer complete native0.10ps sources, then finer-grid views,
then partial histories, with longer coverage first. Physically different paths
retain different event IDs. Multiple remaining views/centers of one event share
one inverse-event weight; whole-ancestor bootstrap still groups related paths.
Report retained physical events, event views, duplicate pairs and melt ancestors
separately. Near-equal but nonidentical archived trajectories are not claimed
to be the same path or additional independent melts.

This replacement reuses the original22:48 Paris deadline. It may stop only the
previous evaluation coordinator/processes owned by this chat. Upstream lineage
jobs and simulations are read-only dependencies and are not modified. The new
bundle and W&B evaluation key `all010-lineage-birth-v2` preserve the earlier
scope and results. All readouts remain local; only existing training summaries
receive associated final metrics under `all010_birth_v2`.
