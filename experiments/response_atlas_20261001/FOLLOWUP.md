# Stronger controls after the response-atlas pilot

The pilot's three values-only models selected epoch250, the optimization boundary.
Four atomistic shots were noisy for some small responses. Address these gaps
before a larger active atomistic comparison.

1. **Optimization, labels and cost:** values8, values32 and responses8, five paired
   initializations, shared 64-shot selection labels, 250/1000/2500-update checkpoints,
   and measured 15/45-second acquisition-plus-training checkpoints. Score exact
   feature and derivative error on 4096 fresh toy points. This distinguishes
   optimization and value-label noise from useful response supervision; toy CPU
   economics do not extrapolate to MACE.
2. **Response precision:** fresh 32-shot AD measurements on four previously examined
   Al256 development states, fixed 4/8/16/32-shot prefixes, and 16 fresh CRN pairs
   per direction. Retain 20/100 fs and include weak-response parents 8 and 12.

No new native encoder, AP loss, time or temperature predictor is introduced.
Atomistic work measures the oracle, not prediction performance. Five-arm active
learning and independent-liquid-parent generalization remain untested.

Recipe: `configs/simulation/response_atlas_followup_20261001.json`.
Reproduce: `python -m src.research.response_atlas.followup submit --config configs/simulation/response_atlas_followup_20261001.json`.
Results: `${storage:training_storage}/response_atlas/followup-20261001`.
Definitions: [metrics](../../docs/metrics/response_atlas_followup.md).

## First completed findings

All 15 toy fits completed. At 45 seconds total measured CPU acquisition/training,
values8/value32/responses8 mean future-feature MSE is .027705/.011129/.006711;
derivative MSE is .218733/.134789/.050591. Response training beats the 32-shot
value control on both metrics for each of the five paired initializations.
The 15-second checkpoints select the same models. This addresses the original
optimization-boundary concern but remains a toy, shared-data result. It does
not establish MACE cost efficiency, active-selection value, or physical transfer.
Atomistic shot-precision collection is still running.
