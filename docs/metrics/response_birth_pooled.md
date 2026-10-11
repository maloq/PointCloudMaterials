# Pooled 0.10 ps birth holdout and neighborhood audit

The user explicitly requests combining the original validation/selection,
calibration and test sources into one test. The parent is the immutable
all010-birth-v2 cohort: 92 source views with saved velocities, unchanged
root-emergence labels, eligibility, exact 0.1/0.5/1 ps leads and 1:4 controls.
All encoder spaces use identical rows. Original roles remain in every artifact;
this derived pooling does not rewrite the fixed Al64 release.

There are 19 eligible original-training events from 18 ancestors, and 11 pooled
test events from 9 ancestors. Earlier scores and checkpoints remain historical.
The old readouts used validation for model selection and calibration for a
probability transform, so their pooled predictions are not the new test result.

## Readout refitting

Freeze the twelve original response encoders. Rank original training ancestors
by the fixed digest of seed, ancestor identity and `inner-selection`. Reserve
the first four only for internal readout selection, with all related events
and views kept together. Fit candidates on the remaining original training
ancestors. Select logistic C from the parent's fixed grid by weighted binary
NLL, or CatBoost iteration count by internal NLL with the parent's maximum
1000 iterations and patience 100. Do not inspect pooled test labels for any
fitting choice. Refit the chosen readout on ALL original training ancestors.
Refit logistic normalization on all training rows; internal candidate fits use
only inner-fit normalization. Hyperparameters otherwise match the parent.

No probability recalibration or alarm threshold uses the new pooled test.
Report raw NLL, Brier, weighted AUROC and AP, and matched-set AUC/top1 using the
parent formulas. Inverse-event row weights make each physical event equal.
Constant prior uses original training labels. Three-seed neural summaries are
means of separate-model scores, not probability ensembles. Descriptor controls
have one fixed seed and include full-cell component statistics, a stronger
context than neural patches. AP is diagnostic only; no AP-based choices.

Use 2000 common bootstrap draws of entire pooled melt ancestors; percentile
intervals and paired contrasts are conditional on the fitted readouts. Frozen
encoders retain their original label-free response checkpoint selection, and
earlier birth results were inspected. This is exploratory birth-label holdout
evidence, not a newly untouched prospective test. No encoder is retrained and
no new W&B run is created; final metrics update existing training-run summaries.
The existing predictor/encoder input contract is copied and the selector and
evaluation roles are updated explicitly. Time, temperature and identities are
never tensor inputs.

## Neighborhood audit

Reproduce the original per-source random stream exactly: for every candidate
event, permute onset members and retain up to four, then draw the original
1024 control-candidate atoms to advance the stream. The original veto rejects
a center if ANY atom of ANY connected PTM component with size at least 8 lies
within 8 Å at ANY of the three leads. All sampled centers failing means the
whole event is unavailable at every lead. The audit asserts that its replayed
eligible-center count equals the saved count for every event.

Use original component labels and source geometry with periodic distances;
do not rerun classification or dynamics. Per center, lead and blocking component,
record full component size, nearest component-atom distance, number of component
atoms inside the 8 Å sphere, whether the center belongs to the component, and
overlap with the nucleus's first eight-atom-crossing member IDs. A nonzero overlap
is labelled shared-onset; zero overlap is labelled disjoint. These are atom-identity
descriptions, not proofs of common lineage or causal external interference.
The full neighborhood includes component atoms outside nearest80 encoder support.

Link blocking components to families through the frozen frame/component trace.
Establishment at observation means the family already confirmed 64 atoms for
1.5 ps, or was recorded as an initial >=64-atom crystal or inherited an established
crystal. Instantaneous component size >=64 is separately reported. Unmatched
fragments have unknown establishment status, never assumed established or liquid.
Inspect all onset-member centers additionally to count cases lost merely through
the four-center subsampling. This audit does not change evaluation eligibility.

Audit all 210 candidate event views across the 92 sources. Physical identity is
the parent's onset Q/V, box, time, first-member and ancestor digest. Keep every
view to explain original counts; unique-event summaries prefer the parent's
complete/native ordering before partial aliases. Rejection flags for unique
events follow that primary view's sampled centers, not an unrecorded union.
Shared/disjoint categories can overlap within an event. Distance/size distributions
of contacts count center-component observations and are not independent-crystal
statistics. Event minimum distance/maximum size are separately exported.

The audit and readouts run locally in one frozen, owned, detached job with two
GPU lanes and a CPU audit. Retain the six-entry leased encoder cache. A two-hour
cap also respects the previous absolute deadline. All CSVs bind these definitions
and implementation hashes; historical definitions and source preparations remain
unchanged.
