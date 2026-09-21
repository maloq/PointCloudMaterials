# More informative matched crystallization evaluation

Question: does relaxed-input encoder training improve held-out local onset
prediction beyond input relaxation alone, historical frozen states and geometry?

Use a fixed 12 ps observation cadence on selection/calibration/test sources, with
no transition oversampling. Retain the 15-origin training cohort and all historical
source roles. This gives 338 distinct local onsets in 11,256 test windows before
timeout exclusions, versus eight in the earlier two-origin diagnostic. Independent
uncertainty units remain the 30 simulation sources, not the 338 atom onsets.

Compare nine expanded encoders, three earlier pilot encoders, the original parent
on unrelaxed/relaxed inputs, earlier SIGReg/EpiJEPA/VICReg MACE and old VICReg
MACE/GATr, plus matched geometry and condition baselines: 46 linear/neural readouts.
Freeze checkpoint selection from development metrics. Use matched readout budgets
and calibration-only alarm thresholds. Report AP, AUROC, hazard NLL, calibration,
timing MAE together with misses and timing-within-3-ps recall. Add paired whole-source
uncertainty and explicit distinct-local-event/source coverage.

Recipe: `configs/analysis/relaxed_encoder_large_test.json`.
Metric definitions: [dense evaluation](../../docs/metrics/relaxed_encoder_large_test.md).
The initial 15-origin evaluation continues separately; neither result is combined
with the two-origin pilot as if it used the same test population.
