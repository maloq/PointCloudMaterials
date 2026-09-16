# Organized and smooth local-state motion

**Discarded approach (16 September 2026):** replacement-embedding training on
frozen encoder features is no longer pursued. Scientific results and exact recipes
are retained. Embedding forecasting and native encoder training remain active.
See [scope and historical reproduction](../../docs/discarded_frozen_encoder_maps.md).

**Completed:** all 44 fits and evaluation finished. No candidate passed the
validation information gate or the joint 0.10 jump requirement. See
[results and explained comparison](RESULTS.md). This is the frozen-feature
protocol; actual MACE training is a distinct subsequent experiment.

Question: do a shared local direction constraint and weak temporal bending
regularization improve the smoothness/information balance of the frozen MACE
structural state? This implements stage B of the
[research proposal](../mace_velocity_20260915/LITERATURE_REVIEW_SMOOTH_MANIFOLD.md).

Use original source-level splits and tracked centers, nine genuinely consecutive
observations per selected source, actual physical time intervals, the complete
message-passing halo and smooth inner pooling. All 1,114 binary and ten native
NPZ velocity records contribute; one conversion containing scattered pairs is
explicitly excluded. This is 40,464 observations. The snapshot state receives the
current coordinate-derived structural block; velocities remain recorded for the
planned observed-history comparison. No forecasting labels or process identity.

Compare 32/64-dimensional state maps, 4/8 shared local directions and two seeds.
Each setting includes physical supervision only, direct slowness, slowness plus
shared directions, slowness plus bending, and the combined model. Four reference
fits retain the full 256-dimensional structure block. The full recipe contains
44 fits with 600 epochs each. The direction predictor is fitted for every control
without letting that auxiliary fitting change the control embedding.

The information gate permits at most 10% increased error in each current physical
target family, both overall and within the low-order population, on both sequence
and original-pair cohorts. Aim for original-pair RMS near 0.10 at 0.75 ps and test
whether 4–8 local directions capture at least 90% of held-out displacement energy.
Keep these requirements separate; small steps alone do not establish a manifold.
Report additional physical probes, bending, direction changes, source intervals,
membership sensitivity and a native-float32 storage round-trip diagnostic.

The previously examined test sources remain development evidence. Sparse-window
sampling and physical readout quality cannot establish transition timing or
spatially meaningful clusters; those remain subsequent assessments. History
inputs and MACE fine-tuning are later, conditional stages, not part of this launch.

Recipe: [mace_local_motion.json](configs/mace_local_motion.json).
Smoke: [mace_local_motion_smoke.json](configs/mace_local_motion_smoke.json).
Definitions: [metric documentation](../../docs/metrics/mace_local_motion.md).
Historical execution record: [workflow](../../docs/mace_local_motion.md).

Historical reproduction using archived source: `python -m src.research.mace_local_state.run --config
configs/analysis/mace_local_motion.json --stage motion-all` in a matching allocated
runtime. Completed findings are recorded in [RESULTS.md](RESULTS.md); implementation
alone is not evidence of improvement.

Implementation validation completed: 20 focused tests passed; the 20-source, 720-observation smoke workflow completed all six model types. A replay audit matched 160 observations to original paired-cache labels and embeddings (maximum relative feature difference 4.90e-6). These short smoke fits are not scientific performance results. The full comparison and evaluation are complete; see RESULTS.md and the run output.
