# Supervised predictive-information metrics

Branch `crystallization_supervised`; label = original-MD sustained local first
onset. Encoder/head fitting uses90 independent sources, checkpoint/promotion
selection15, probability/threshold calibration15, historical testing30. Each
source has equal total weight and its observations share that weight equally.
Labels use five first-event bins ending0.75,3,6,9,12 ps, or survival through12 ps.
All examples come from the producer's prospective natural-risk population.

New native spatial models default to128 channels and a128-dimensional export:
634,496 encoder parameters and651,653 including the single-view onset head.
Named capacity ablations retain explicit widths. This default changes model
capacity, not metric definitions or population weights. Each new run retains its
resolved width, parameter counts and spatial runtime source hashes.

Live protocol `supervised_onset_information_v4`: encoder and frozen readouts
train by first-event/survival NLL and select by minimum source-weighted selection
NLL. Exact ties retain the earlier checkpoint/declared arm order. AP never
controls training, selection, promotion, ensemble weights or table sorting.
`best.pt` is the NLL-selected state; old `best3`/`best6` artifacts belong to
historical AP protocols and are not relabeled or resumed as v4.

`test_AP3`, `test_AP6`, `test_AP12` use sklearn weighted average precision on raw
cumulative risk, not trapezoidal PR area. Descriptor models are supervised
predictors, not learned atom-level encoders. The optional uniform risk mixture
is a fixed 50/50 ensemble of NLL-selected members, not a single encoder.
Readouts use only the exported state; descriptor controls use only descriptors.
No temperature, simulation age or explicit time covariates enter any predictor.
The constant control has no inputs. Each export links its frozen input ledger.

Training and checkpoint selection calculate NLL directly from hazard logits.
Exported `selection_event_nll`/`test_event_nll` reconstruct six first-event/survival
probabilities by differencing [0, cumulative risks at five horizons, 1], select
the realized event category and average -log(probability) with source weights.
The export clips realized probabilities below 1e-12 to handle finite-precision
saturation. This numerical floor is not used in the logit-space training loss.
Horizon raw/calibrated log losses are binary negative log likelihoods from the
existing weighted_scores producer, whose probabilities are clipped to [1e-7,
1-1e-7]. Both appear in the comparison CSV; raw Brier remains in JSON.

Historical v2/v3 studies used AP3 selection, AP6 tie-breaking, AP6-selected
alternative checkpoints and AP-tuned readouts/mixtures. Their frozen metric
snapshots remain unchanged. Four active v3 capacity jobs were stopped when the
user withdrew AP optimization; partial checkpoints remain historical AP fits.

Historical v1 neural readouts included five temperature indicators
(400,450,500,510,520 K), plus age/600 ps and its square. Its geometry encoder
did not receive those columns, and its `conditions` control meant temperature
plus age. Existing metric snapshots remain unchanged with a dated erratum beside
the old tables. Do not reinterpret their AP values as v2 results.

AP intervals resample whole independent sources1000 times. Every sampled source
copy carries equal total weight. Draws with no positive event are excluded and
their count retained in JSON; intervals are conditional on event-containing
draws. They do not estimate training-seed uncertainty. This is one seed per fit.

A single positive-slope affine log-odds map, shared over all horizons, is fitted
on calibration sources using mean3/6 ps BCE plus1e-4 squared parameter penalty.
It preserves temporal cumulative ordering. AP and its intervals use raw scores
to avoid numerical saturation ties; Brier and alarm probabilities use calibrated
scores. `raw_brier` remains in JSON. Calibration performance is in-sample for
the calibrator, not an independent estimate. The alarm threshold admits at most
5% weighted calibration negatives, respecting whole ties. Test recall and actual
test FPR are reported separately; test FPR is not forced to5%.

Training NLL uses the discrete-time first-event/survival likelihood. Sampling is
an equal mixture of the natural source-weighted distribution p and its by6ps
positive subset. Multiply each sampled loss by p/q; evaluation never oversamples.
There is no differentiable AP/ranking loss or full-population encoder replay in
v4. Microbatched NLL gradients are tested against an unsplit backward pass.
Likelihood trains all five event bins; AP3/AP6 are reporting diagnostics.

`dataset_spectrum`, `fitting_spectrum` and `test_spectrum` are source-weighted,
centered covariance eigenspectra of exported128-dimensional states. d95 is the
number of eigenvectors explaining95% of variance. Entropy and participation
ranks are spectral dimensions, not nonlinear intrinsic manifold dimensions.

Noise diagnostics fix the center and perturb other atoms with independent normal
coordinates of sigma=f*d12/sqrt(3), where d12 is their clean twelve-nearest-neighbor
mean distance. Report realized3D displacement RMS in A and RMS relative to d12.
Embedding response is sqrt(E_source||z_noisy-z_clean||²/(2*trace Cov_train(z))).
Edges are rebuilt. Paired-view diagnostics perturb one supplied view at a time;
they do not measure the response of a subsequent relaxation process. Dataset
selection is predetermined four test rows per source, with one fixed random seed.

The held-out paired cache has12 ps anchor cadence and no0.75 ps pairs. The
requested0.75 ps movement/stability and movement rank are explicitly absent,
not zero and not replaced by12 ps. All artifacts retain the producer/source,
configuration, checkpoint and metric implementation identities. Prior exported
metric definitions remain unchanged.

Online W&B logs are a secondary view of these recorded calculations. Encoder
training curves show sampled optimizer updates, not epoch averages; there is no ranking objective in the current protocol. Fresh readouts use independent update axes.
Final metrics from selected checkpoints/readouts are run summaries, preserving
their selector, horizon and split. Local predictions and frozen metric exports
remain the reproducible evidence; tracking does not change any calculation.

The active spatial runtime uses native cuEquivariance ir_mul layout, fused indexed
convolution and full-graph dynamic compilation. All irreps remain in the atom
normalization before scalar pooling. Bounded cached index plans preserve example
order/multiplicity and contain no learned values. This changes execution and
initialization consumption, not the definitions of NLL, AP or exported spectra.
Runtime/source identities distinguish fresh runs from historical checkpoints.

Runtime update (2026-09-25): supervised graph banks pad the atom axis to
`80 * microbatch` with disconnected zero-attribute, zero-weight nodes. Real
edges, graph count, central atoms and `n_ref` are unchanged, including in short
final batches and perturbed-coordinate evaluation. This is an arithmetic runtime
change, not a new physical input or metric. The input ledger records capacity;
source identities require fresh outputs for the changed implementation.

The fixed-Al64 adapter preserves sealed source roles and immutable sample IDs,
using current physical-coordinate observed/relaxed patches directly. It fits
geometric normalization only on training sources and adds no descriptor or
condition input. Live validation adds source-weighted raw Brier and binary log
loss at 3/6 ps alongside AP and first-event NLL. W&B names these
`validation/brier_score_*` and `validation/binary_log_loss_*`; the checkpoint
selector remains exclusively `validation/event_nll`. Counts and release identity
are static summary fields. Historical summary backfills expose only previously
calculated results and do not manufacture missing training curves.

The separate encoder/context epoch campaign uses exact shuffled full passes,
including partial batches, and NLL selection from epoch 12 onward. See
[encoder/context metrics](encoder_context.md). Historical update-budget studies
retain their original replacement sampling and checkpoint eligibility.
