# Recovering local structure in continuous MACE embeddings

Question: can a representation retain the smooth context pilot's boundary
continuity and relaxed topology while recovering instantaneous topology and local
bond-order changes?

The first experiment compares ridge and residual nonlinear readouts on eight
retained representations: original, smooth-inner, tracked-center and concatenated
inner/center, each before and after the previous pilot continuation. Frozen fusion
uses the same backbone; trained fusion combines two separately trained encoders.
This is a diagnostic of complementary information, not evidence that concatenation
alone solves the problem with a single trained backbone.

The second experiment uses a single shared backbone emitting both readouts in
one graph pass. It compares two 12-epoch continuations from the same original
forecast checkpoint. Both retain the original VICReg objective on the inner
block. `dual_ssl` trains physical readouts on detached encoder features;
`dual_physics` additionally updates the encoder from those readout losses.
Targets are instantaneous and relaxed TDA plus q4/q6, density and mean first-shell
distance. They describe unaugmented anchor geometry. Both variants use identical
warm-started heads, optimizer settings, training rows and augmentation draws.

Both choose their checkpoint using physical-head validation error, including
epoch zero, then evaluate newly fitted ridge and the trained heads. Selection
differs from the earlier pilot's validation-VICReg selection. No held-out test
trajectory or source enters training or model selection. The cohort remains the
previously examined exploratory 18/6/6 source split. We do not claim fresh-source
confirmation or downstream forecasting improvements from these reconstruction
experiments.

The primary outcomes are balanced TDA errors, first-shell observable increment
error reductions, and controlled membership-crossing curves for both embeddings
and decoded hot TDA. Temporal variation is also measured separately for each 256D
block and after equal training-variance normalization of the two blocks. Retaining
center information is allowed to increase physically meaningful temporal response;
the experiment must still distinguish that response from finite boundary jumps.

Recipe: [mace_context_recovery.json](../../configs/analysis/mace_context_recovery.json).
With conda `pointnet`:

```bash
python -m src.research.mace_context.run --config configs/analysis/mace_context_recovery.json --stage recovery-readouts --device cuda:1
python -m src.research.mace_context.run --config configs/analysis/mace_context_recovery.json --stage recovery-verify --device cuda:0
python -m src.research.mace_context.run --config configs/analysis/mace_context_recovery.json --stage recovery-train --variant dual_ssl --device cuda:0
python -m src.research.mace_context.run --config configs/analysis/mace_context_recovery.json --stage recovery-train --variant dual_physics --device cuda:1
python -m src.research.mace_context.run --config configs/analysis/mace_context_recovery.json --stage recovery-summarize --device cpu
```

Completed results and selected histories are published to the
[result report](../../output/mace_context_recovery/forecast-seed20260910-20260914/README.md).
[Metric definitions](../../docs/metrics/mace_context_recovery.md) specify all
normalizations, validation choices and source bootstrap calculations.
[Cached findings](CACHED_RESULTS.md) precede encoder optimization.
The [completed joint-training findings](RESULTS.md) report both finished runs,
their selected epochs, physical readouts and remaining stability/topology tradeoffs.
Detached execution and checkpoint details belong in
[the operational guide](../../docs/mace_context_recovery.md).
