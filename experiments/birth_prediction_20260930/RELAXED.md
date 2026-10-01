# Does removing thermal motion change pre-appearance predictive information?

Compare every retained unrelaxed input with its full-cell, fixed-box inherent
configuration. Keep all cases, liquid controls, source roles, history endpoints,
event weights, sampled atom identities and readout folds fixed. No new MD or
source sampling is needed. The 1,030 unique observed cells provide all 11,793
patches in the existing 1,475 histories.

Use the generating Al MEAM potential, FIRE and fmax <= 0.01 eV/A. Select patch
membership once in original MD geometry, then carry those IDs through the
minimization. Local patch coordinates are extracted before global quantization.
Full-cell relaxation has broader computational context than the local exported
patch; this is recorded explicitly. Original MD outcomes and crystal-free input
eligibility remain unchanged even if the inherent configuration gains ordering.

Refit the same six linear/boosted readouts on rich descriptors, rich/TDA MACE,
and VICReg MACE for all thirteen temporal/site controls. Keep encoders frozen
and their original normalization. Select readouts by validation likelihood;
AP remains diagnostic. Reuse the original independent fixed test and the five
readout folds separately. Source-held-out errors and paired domain intervals
measure the change in accessible information; train errors expose overfitting.
This is exploratory reuse of inspected sources, not new confirmatory evidence.

The experiment tests how quenched geometry transfers through retained encoders;
it does not test training a new relaxed-domain encoder. Static quenches are
not future evolution, and future-selected sites still limit prospective claims.

Recipe: `configs/birth_prediction/relaxed_temporal_20261001.json`.
Reproduction: `python -m src.research.birth_prediction.relaxed submit --config configs/birth_prediction/relaxed_temporal_20261001.json`.
Definitions: [paired relaxed metrics](../../docs/metrics/birth_prediction_relaxed.md).
Execution/resume: [workflow](../../docs/birth_prediction.md#full-cell-relaxed-input-comparison).
