# Static Al analysis with the selected v6 MACE encoder

Question: how does the newest Al-only MACE–VICReg state organize the same six
static Al configurations analyzed with the v6 GATr encoder?

Use the selected final update 1,465 of `mace-vicreg-al-stable-20260918`, with
checkpoint hash pinned in `configs/analysis/structural_mace_v6_static.json`.
The checkpoint-selection score is 0.24064353108406067 on Al-only training
target normalization. Use the raw 128-channel state, excluding the projector
and physical/topology heads.

Keep the six frames (166, 170, 174, 175, 177, 240 ps), 684,723-center interior
grid, seven spherical clusters and complete static analysis from the
[v6 GATr protocol](STATIC_AL.md). MACE uses the trained fixed Al scale,
full local support, directed 5-model-unit edges, cuEquivariance and native
protected-FP32/selective-BF16 execution, compiled with precision casts preserved.
Each local graph retains its center-dependent taper. Clustering is fitted
independently; cluster IDs need
not correspond between encoders. The raw PCA diagnostic uses float64 full SVD.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_mace_v6_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_mace_v6_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_mace_v6_al.yaml
```

Results: `output/structural_static/mace-vicreg-v6-step1465-al-20260918/`.
Verification and full analysis are in progress; findings will be recorded here
after completion. These six static files overlap training inputs, so results
describe the learned representation rather than held-out accuracy. The models
also differ in training seed and selected update; this is not an isolated
architecture-effect estimate.

See [execution and precision details](../../docs/structural_static_analysis.md)
and [shared structural pretraining](../../docs/research_glossary.md#shared-structural-pretraining)
for the meanings of exported state and separate prediction heads.
