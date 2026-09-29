# Encoder mechanism diagnostics, version 1

This study uses fixed Al64 all64 identities and complete rows. Sources retain
90/15/15/30 train/selection/calibration/test roles. Temperature, age, explicit
clock, species/material identity and scale channels are never model inputs.
Normalization is fixed preprocessing; encoder inputs are centered coordinates,
a constant atom channel and center indicator. AP is a diagnostic only.

The28 September recovery preserves these numerical definitions and completed
exports. Each unfinished checkpoint evaluation now runs in its own process;
adaptation modes also use separate processes. Diagnostic probes are local and
scientific training remains online. The recovery validates retained checkpoint
hashes and the actual0.75-ps source intervals before submitting. New exports
use a separate output revision and capture the recovery producer in this
contract; historical exports keep their original implementation hashes.
Further recovery may reuse verified completed assays from an explicitly named
previous recovery with the same scientific configuration and checkpoint hashes.
Missing rows alone are evaluated in the new revision; historical numerical
exports are never overwritten. The native plotter now extends the original
metric K-means fit instead of refitting it, as recorded in encoder_quality.

## Readout controls

The completed-study report (`encoder_mechanisms.results`) reads verified saved
metrics and predictions; it performs no model fit or checkpoint selection.
Its structural summaries average the three fixed Al frames within each seed,
then give each of the three fitted seeds equal weight. Predictive summaries
average the already source-weighted test scores across these seeds. Raw and
calibrated proper scores, the jointly fitted head and the identically refitted
frozen readout are reported separately. Adaptation retains validation-NLL
selection after epoch12 and the separate fixed epoch24 endpoint.

Reported R1−R0/R1−R2 endpoint contrasts and adapted/scratch−frozen contrasts
match exact sample IDs, sources, roles and event labels across models and seeds.
For each source, they average the three seed-specific loss differences, then
resample the same30 whole test sources1000 times (evaluation seed20260926).
The95% interval is conditional on those fitted encoders and heads; it does not
represent uncertainty over a population of training seeds or adjust for
multiple comparisons. These are averages of losses, not an ensemble of
predictions. Individual fitted-seed contrasts are retained. The calculation
uses the existing `encoder_quality.metrics.paired_scores` definitions for
3/6-ps log loss and Brier score. Trajectory plots show all three seed curves
and their means without using test scores to choose epochs or treatments.

Six observed exports: five existing scratch/VICReg/Epi pretrained or adapted
models and the exact Epi frozen initial reference (including pool normalization).
That reference is a matched Epi initialization; no unverified claim of a retained
VICReg initialization is made. Existing exact-budget linear and 128-unit MLP
fits are reused with prediction/sample/checkpoint receipts. New widths are
linear, one hidden layer 128 or 256 with SiLU. All new probes: AdamW 0.001,
weight decay 1e-4, 24 full epochs, batch256, clipping5, validation hazard-NLL
selection after epoch12. Calibration uses calibration sources only.

Compare z128, z128+d32, and z128+P32(z). P32 is a fixed orthonormal Gaussian
projection, seed evaluation_seed+73, applied to train-standardized z. Both
augmented inputs have160 channels and equal head parameter counts. No new
information enters the projection control. d32 uses the unchanged radial/count/
angular-power producer. Density26 uses its first26 radial/count columns.
SOAP50 is DScribe SOAP on exactly the same current 80 candidates within8 Å,
constant pseudo-species H, center included, nonperiodic, sigma0.3 Å, radial
basis size4, angular maximum4. It has no extra neighbor halo. The chemical
symbol is constant, not an observed material channel.

Forecast proper scores/calibration and paired whole-source bootstrap differences
retain encoder_quality definitions. Compare joint-minus-z and joint-minus-
redundant, not a pooled winner. Linear/MLP comparisons are finite accessibility
assays; failure does not prove information is absent from the representation.

Distance diagnostics use raw z128, train-standardized z128, and its weighted
train covariance PCA32 without whitening. Compare standardized d32 at the same
32 dimensions. k in16/64/256 is chosen by selection-source event NLL; fitting
neighbors never include other roles. Physical kNN readout fixes k64 independently
of outcomes; source-weighted averages predict32 train-standardized descriptors
on test sources. Its NMSE averages target dimensions and then whole-source
means. Descriptor-distance prediction is a geometric ceiling control.

## Matched training and adaptation

R0: Epi+variance, alignment0. R1: Epi+variance, alignment25/51. R2: VICReg,
alignment25/51, variance25/51, covariance1/51. Epi coefficient is0.1 divided by
its initial scale. Both views and each view's frozen reference participate in
R0 and R1. Native MACE128/export128, batch/microbatch256, 24 epochs, common
cosine/warmup family at1e-4. Shared per-seed initial state and normalization,
exact same paired rows/order, seeds20260926/27/28. Endpoints0/4/8/12/18/24;
fixed24 primary and12 secondary. No downstream checkpoint promotion.

R1 fixed24 supplies frozen and trainable supervised arms. Scratch loads the
same retained epoch0, including pool normalization. All three receive identical
fresh head initialization and epoch permutations within seed. Fit24 epochs;
select validation hazard NLL from epoch12 and report fixed24 separately. Fresh
frozen linear/MLP readouts accompany the joint head. Head-only training must
leave encoder tensors unchanged. Seed uncertainty and conditional source
bootstrap uncertainty must remain separate.

## Displacement-matched sensitivity

From each of30 held-out sources, draw2 frame origins uniformly without replacement
and16 fixed benchmark centers, with source-specific deterministic seeds. Track
the origin's nearest80 atom IDs to the next0.75 ps frame. Recenter each view,
use its actual box, and rebuild graph edges/radius masks. This is a tracked
origin-neighbor assay, not independently selected future neighbors.

For each pair, generate3 center-fixed independent Gaussian perturbations and
rescale each to exactly the real pair's RMS displacement over79 noncentral
atoms. Check realized RMS numerically. Compare real and mean synthetic squared
embedding changes for the same pairs. Normalize RMS by sqrt(2 trace), where
trace is mean squared centered embedding norm on these same origin observations.
Report per-source real/synthetic RMS ratio, plus Spearman associations of excess
squared response (real minus mean synthetic) with displacement, fraction of
center bonds crossing3.6 Å, and D2min. D2min here is mean squared residual of a
least-squares 3×3 affine map on the origin's12 nearest neighbors, with their atom
identities retained. Undefined correlations stay null. This synthetic control
is not a thermal ensemble and these associations are not causal classification.

## Birth coverage

Freeze the completed training audit before processing the existing held-out
roles. Original event/ancestry/confirmation/censoring rules and uniform controls
are unchanged; no future-centered candidate rows enter the support tables.
Current established-clear excludes confirmed lineage or any size-qualified
component within the radius. PTM-all-clear excludes any FCC/HCP/BCC atom.
Report8 Å local and32 Å conservative spherical context envelope separately.
The latter is not a declaration that a future predictor consumes a full sphere.

Count eligible uniform atom-origin rows by role, criterion, required history and
horizon, cause-specific labels, distinct (source,event) pairs and sources with
births. Inverse known uniform inclusion probability estimates eligible atom-
origin exposure. It is not independent sample size, kinetic rate or a committor.
Counts across overlapping horizons/criteria must not be summed. Ineligible or
censored rows never become negatives. Coverage inspection consumes these sources
for benchmark design. Prediction fits remain gated on a sealed natural-risk
release with enough independent held-out events; no unsupported positives are
invented and no simulations are launched by this workflow.


Tracking revision (2026-09-26): diagnostic frozen readouts and per-checkpoint
evaluations keep their logs and results locally. Associated final scores update
a recorded scientific training run through the API, without creating or
restarting runs. Scientific training remains online. This changes logging and
validates identity/hash before cached readout reuse; objectives, selectors,
metric calculations and historical exported definitions are unchanged.


## Execution refactor

The code-cleanup revision consolidates artifact export, preparation, checkpoint
and execution helpers. Scientific formulas, rows, weights, fitting populations
and selectors are unchanged. New table exports include a per-table hash and
definition binding. Historical exported definitions and frozen source snapshots
remain authoritative; changed implementation hashes require a new export revision.
