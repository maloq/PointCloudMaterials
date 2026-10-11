# Continuous-motion nucleus-birth transfer at 1, 3 and 6 ps

Reuse all92 saved-velocity trajectory views in the pinned all010 v3 inventory,
including exact0.1ps views of native0.01ps and historical partial archives.
These are not92 independent ancestors. The continuous-motion encoders were
trained on88 verified-MEAM views; the four unknown-potential historical views
remain explicitly included here as transfer data. No velocity-free million-atom
inputs, relaxed observations, other cadences or interpolated states are added.

Reuse the established response_dense_birth source extractor and physical-event
alias rules with leads changed to exactly1,3,6ps. Positive centers belong to
first8-atom root emergence from liquid that later sustains64 atoms for1.5ps.
Choose up to four onset-member centers before the input neighborhood filter.
Reject any center with an8+ connected PTM component within8A at ANY of the three
input times. All three horizons and models retain identical case/control rows.
Sample four same-source/time control atoms per positive, outside16A of onset,
with no8+ component within8A throughout the interval from6ps before onset to6ps
after onset, every0.1ps. Insufficient history/followup is excluded, not negative.
Sub-eight precursors and isolated PTM crystal atoms are allowed. The descriptor
control adds present full-cell component summaries and has broader context.

Onset full-cell Q,V, box, time, ancestry and first members define physical event
aliases. Repeated positive center/three-time Q,V observations are deduplicated
as matched five-row sets. Distinct histories remain explicit. Rows receive
weight1/(number of rows for their physical event), giving equal total event
weight. Native cadences and original ancestor roles remain attached. Coverage
exports list every source, event, exclusion, duplicate and fitting partition.
This retrospective1:4 case/control assay has positive prevalence0.20; it is NOT
population nucleation risk or a general any-crystallization detector.

## Frozen representations and input audit

Evaluate all nine1600-update continuous-motion fits: discrete, continuous,
derivative, each at three seeds. Use each run's validation-NLL-selected frozen
checkpoint only after its completion receipt verifies its checksum. Never select
checkpoints using birth AP. Source roles used for encoder fitting/selection are
checked to be original Al64 training ancestors. Frozen canonical source is
imported in a separate inference process with dependency hashes verified.
Replay the first original held-out prediction block with its recorded eight
particles, batch and seed. Require maximum mean-prediction deviation<=2e-4
on the first64 observations, whose Philox subsequences are common across the
campaign GPUs. Keep the full256-observation stochastic difference separately:
CUDA normal generation depends on GPU launch geometry. The A100/RTX diagnostic
matched the first216 observations within6.8e-6; only the last40 differed, exactly
where PyTorch maps later normal components on A100. Birth inference continues
to use the same full observation cohort and common RTX streams; no metric rows,
checkpoint choices or readout definitions change. The failed full-block audit
and original frozen execution are preserved; resumed exports bind this revision.

All models see one unrelaxed nearest80 Q,V patch, smooth8A support,5A edges,
constant atom channel, no external halo, history, species, temperature, age or
absolute-time covariates. The reused birth extractor stores velocities in A/fs;
explicitly multiply by1000 for MotionModel's A/ps convention. Structural128 is
E(Q); structural_motion256 concatenates E(Q) and0.1ps*DE(Q)[V]. Future_law48 is
the24 predicted normalized-descriptor means plus24 log total variances after
integrating the frozen latent process for that lead, using eight particles.
Total variance is mean(exp(logvar)+mean^2)-mixture_mean^2, clamped at1e-8 before
log. Fixed random streams are shared across neural treatments. Duration controls
integration only; it is never a neural input. Future_law reads no future atoms.

## Readouts and scoring

Pool original selection/validation, calibration and test, as previously
requested. Retain original roles; never fit, select or calibrate using pooled
birth labels. Sort eligible original-training ancestors by the fixed digest of
seed, ancestor and inner-selection. Reserve the first four for NLL selection;
refit chosen diagnostic readouts on all original-training ancestors. Require
both classes in each partition, at least five training events and three pooled
test events; fail without relaxing labels if unavailable.

Reuse response_dense_birth.pooled.fit: logistic regression with training-only
standardization and C selected by NLL from[.001,.01,.1,1,10], and nonlinear
CatBoost depth4, learning rate.04, l2=10, at most1000 iterations, patience100,
selected by inner NLL and then refit for that iteration count. No probability
recalibration. Controls are rich444 descriptors and the training constant prior.
These are local frozen diagnostic fits, with no new W&B runs. There are171 fits:
9 models x3 spaces x3 leads x2 readouts, plus9 descriptor/prior controls.

Report event-weighted binary NLL, Brier, AUROC, AP and matched-set AUC/top1 using
the established birth_prediction and response_birth formulas. Ties receive
fractional credit for matched top1; constant-prior AP and matched top1 are0.20.
AP is evaluation only. Use2000 common bootstrap draws of whole melt ancestors
for95% percentile intervals and paired treatment differences. Report means of
per-seed scores and seed standard deviation, never fitted ensembles. Intervals
are conditional on fitted readouts. Earlier related evaluations were inspected,
so these are exploratory birth-label holdout results.

Scientific CSVs bind this document and exact implementation hashes. Checkpoints,
predictions and evaluation receipts are retained; generated embeddings use the
shared six-entry leased cache. Final metrics update existing W&B run summaries.
