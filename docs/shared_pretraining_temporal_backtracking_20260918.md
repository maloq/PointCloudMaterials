# Temporal-only backtracking continuation

This continues the mixed-material snapshot GATr at update 250. Its original
run is checkpointed, with best physical/TDA selection score 0.19193527. The
model, auxiliary heads, target normalizers, AdamW moments, RNG and global cosine
schedule are preserved. The new run has a separate identity and W&B history
because its objective changes. Total exposure stays twelve epoch equivalents:
250 old updates plus 1,242 new updates, ending at update 1,492.

## Changed work and objective

Spatial updates encode only anchor and spatial partner. They neither load nor
encode past/next temporal frames and have exactly zero backtracking contribution.
Temporal updates encode anchor, next and previous snapshots, each independently.
Only anchor/partner states receive physical/TDA/VICReg supervision. The previous
frame remains unlabeled context for the second difference.

The timestamp-corrected squared Euclidean second difference is unchanged. Its
fixed coefficient is calibrated once at the parent checkpoint using three full
training-only temporal batches of 2,048 anchors. No selection/test labels enter
calibration, and calibration performs no optimizer steps.

For each batch, compute the encoder parameter gradients of the other combined
objectives and of raw backtracking separately, before clipping. Candidate loss
weight is `(0.02 / 0.98) * base_loss / raw_backtracking`, targeting a 2% scalar
contribution on a temporal update. The gradient cap candidate is
`0.10 * norm(base_encoder_gradient) / norm(raw_backtracking_encoder_gradient)`.
Choose the smaller of the median loss candidate and the minimum gradient
candidate across the three batches, then round down to two significant digits.
The chosen coefficient is **21** (previously 0.001). On the reference batches,
weighted backtracking is 1.03–1.08% of total loss and its encoder-gradient norm
is 9.17–9.89% of the other objectives' norm. The gradient criterion, rather than
the 2% scalar target, limits the coefficient. Full values are in the calibration artifact.

This limits the backtracking/base encoder-gradient norm ratio to at most 10%
on the calibration batches. It is not a dynamic cap or a guarantee on later
updates. The coefficient remains fixed, so no extra gradient passes are added
to training. Temporal updates occur with probability one half; a 2% temporal
loss contribution is roughly 1% across updates when their base losses are
comparable. There is no additional factor of two or hidden normalization.

Expected encoder views decrease from 3.5 to 2.5 per anchor: about 29% less
encoder work, with an estimated 1.4x throughput benefit. This estimate precedes
new-run timing and is not a measured speedup. Batch, precision, learning rates,
data split and selection equations remain unchanged. Encoder deployment still
uses one snapshot.

## Execution and evidence

Recipe: `configs/shared_pretraining/gatr_temporal_backtracking/`. Reuse the
existing `shared_pretraining.queue submit --plan .../campaign.json` command.
It freezes code and starts detached in existing H100 allocation 997799, without
requesting another GPU. The continuation checkpoint is an immutable copy in the
new run; its hash and transition receipt are recorded. Exact subsequent resumes
require this new identity. A new baseline evaluation initializes the new run's
best checkpoint under the unchanged physical/TDA selection score.

The [compact W&B layout](shared_pretraining_logging.md) applies to this run.
The original run's frozen logs/checkpoints remain intact.

- New fit: `output/shared_pretraining/gatr-temporal-backtracking-20260918/technical/`
- Queue: `output/shared_pretraining/gatr-temporal-backtracking-campaign-20260918/technical/`
- Calibration/checks: `output/shared_pretraining/gatr-temporal-backtracking-checks-20260918/technical/`

29 checks passed, including two-view spatial sampling without past/next loads,
zero spatial curvature, full/cached gradient agreement, and preservation of
weights, optimizer, target statistics, RNG and schedule across the transition.
The same held-out Al selection cohort and learning-health checks remain in use;
this does not add other-material validation or establish predictive improvement.

The first production spatial updates reported exactly zero curvature; temporal
updates reported 1.04–1.08% loss contribution. All were finite. Initial continuation
selection reproduced the parent score exactly. Early steady timing was 10.8 s
for a spatial update and 16.3 s for temporal updates, supporting approximately
1.4x throughput at an equal mixture; this is a small startup sample. See
[the verification record](../output/shared_pretraining/gatr-temporal-backtracking-checks-20260918/RESULTS.md).
