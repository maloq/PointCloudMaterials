# Equivariant context event prediction

Population: the existing currently at-risk Al onset cohort, matched whole-source
train/selection/calibration/test roles. Test sources are historical, previously
examined. There are five physical event intervals ending at 0.75, 3, 6, 9, 12 ps
and one survival category. Model inputs have no explicit time or temperature.

Each patch head defines hazards and converts them to a normalized six-category
event distribution using products of survival probabilities. A learned shared
gate mixes patch distributions. `event_nll.<role>` is the source-weighted mean
negative log probability of the observed category, in natural-log units. Every
source has total weight 1 / number_of_sources. Smaller is better. Training and
checkpoint selection use this score; AP is never an objective or selector.

`horizons.<h>.<role>.average_precision` uses raw cumulative event probabilities
through horizon h and sklearn's weighted average precision with equal-source
weights. 3 ps is the main AP diagnostic, 6 ps secondary, 12 ps contextual.
Prevalence is its constant-ranking baseline. Brier and binary log loss use the
same population/weights. `raw_brier` and `raw_log_loss` precede calibration.
Other probability scores use one shared increasing affine log-odds calibration,
fitted only on calibration sources with 3/6 ps binary likelihood. AP always uses
raw scores to avoid saturation-induced ties. Recall/false-positive rate use the
existing calibration-set threshold at the declared 5% false-positive target.

AP intervals use 1,000 whole-source bootstrap draws. Draws with no positive
examples are excluded and counted, not treated as zero AP. Paired comparison
intervals use identical source resampling for a candidate and its symmetric
control. `difference` is candidate minus control: negative favors the candidate
for NLL, positive favors it for AP3/AP6. Bootstrap uncertainty covers sources,
not the single training seed. `valid_draws` documents missing-positive draws.

Implementation: `equivariant_context/model.py` (hazards and mixture),
`equivariant_context/train.py` (NLL/export/paired differences), the existing
`supervised_onset/evaluate.py` (calibration, AP intervals), and
`local_predictability/metrics.py` (weighting, scores, threshold). Frozen hashes
are exported alongside every metric CSV. No 0.75 ps stability, noise robustness,
manifold dimension, or full-pipeline rotation metric is inferred from AP/NLL.

Runtime update (2026-09-25): each predictor loads only its consumed cached fields:
`z, actual` for symmetric invariant; additionally `f1` for vector messages;
`f1,f2` for tensor attention; `f1,f2,f4,f6` for harmonic hierarchy. All also use
the same fixed nominal stencil. Normalization still uses the same train-only,
source-weighted statistics. Checkpoints and completion receipts list the actual
input fields. Hierarchical linear sums mix spatial-scale weights before sender
contraction; the nonlinear update and event probabilities are unchanged. Native
patch export uses disconnected zero-weight atom padding. These changes affect
runtime/source hashes, not the definitions of the metrics above.

Batched feature extraction (2026-09-25) preserves the same raw field definitions,
center exclusion, taper radii, `n_ref`, irreducible-component layout and row order.
The typed MACE forward is compiled separately from scalar training exports;
l=4,6 reductions are vectorized. Ordered CPU prefetch and pinned transfers change
execution order only. Output parity uses 1e-5 tolerance. Trained-reference GPU exports exhibit
run-to-run reduction noise; verification records both individual differences
and four-run mean differences instead of treating the reference as bitwise stable.
Partial-source resume and ancestry/shard validation remain mandatory. Extraction
runtime timings are operational diagnostics, not predictive scientific metrics.

The fixed-Al64 repeat uses the sealed all64 row population (126,545 windows),
with the same source roles and label definitions. Exported sample IDs permit
matched legacy16 subset evaluation; full-population Al64 metrics must not be
presented as directly matched to Al16. The input release identity is recorded.
Streaming feature normalization now accumulates source-weighted training moments
in float64, retaining the same scalar centering and equivariant channel RMS
definitions. Held-out fields never contribute to normalization statistics.

Online curves use `validation/event_nll`, `validation/average_precision_3ps/6ps`,
`validation/brier_score_3ps/6ps`, and `validation/binary_log_loss_3ps/6ps`.
The validation role is the producer's `selection` population. These are raw,
uncalibrated scores; AP remains diagnostic only. Final scalar `test/*` summary
fields name raw versus calibrated Brier/log loss explicitly. Their values are
the existing exported scores, not a second calculation. Training NLL curves
use importance-weighted sampled updates rather than a full population pass.

The separate encoder/context epoch campaign uses exact shuffled full passes,
including partial batches, and NLL selection from epoch 12 onward. See
[encoder/context metrics](encoder_context.md). Historical update-budget studies
retain their original replacement sampling and checkpoint eligibility.
