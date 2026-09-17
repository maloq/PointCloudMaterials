# Static Al analysis of structural GATr–VICReg

The RTX PRO 6000 run completed 4,096 updates. Its selected encoder is update
3,072, with source-balanced selection score 0.4045023210346699. The export
recipe pins the source SHA-256 and verifies every tensor against `best.pt`.
The source checkpoint and training state are preserved.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_al.yaml
```

Run in conda `pointnet`. Export refuses to overwrite an existing export.
Verification and analysis use a GPU. The standard pipeline caches inference
for reruns and publishes its gallery, plots and metric tables at
`output/structural_static/gatr-vicreg-step3072-al-20260918/`.

The six Al inherent snapshots are 166, 170, 174, 175, 177 and 240 ps.
The recipe preserves `static.yaml` analysis settings: seven spherical clusters,
standardization, PCA retaining 99% variance up to 64 components, L2 normalization,
t-SNE/UMAP, connected-regime analysis, representative structures, PTM/CNA and
spatial figures. It reuses the established three-edge-layer context grid with
684,723 centers. This excludes incomplete neighborhoods at nonperiodic borders.
No box is inferred from coordinate extrema.

Inference uses the trained raw 128-channel center state. The nearest-160 samples
in the standard loader supply the sampling grid and normalized representative
displays; spatial maps retain physical center coordinates. Encoder
inference independently extracts every source atom within the trained physical
support (about 16.87 Angstrom for Al). Offsets undergo the same float64
subtraction, float32 storage and fixed scaling as training. The Al radius is
9.121389139452193 Angstrom, with reference radius 9.192189; weights taper from
15 to 17 model units. Species is the trained Al channel, time is zero, and the
trained log material scale is supplied. No scale is refitted on analysis data.
FP32 inference disables TF32. Projection and physical/TDA heads are excluded.

Verification checks exported tensors exactly, matches all seven native training
input tensors and the encoder output on a real prepared static example, and
checks batch/padding/order independence on the first analysis snapshot. Each
full analyzed frame repeats the batch-size check on six centers. Receipts are
`technical/static-verification.json` and `technical/structural-inference-protocol.json`;
`technical/structural-inference-status.json` reports extraction progress.

These static snapshots contributed to the broad training release. Results are
descriptive representation diagnostics, not held-out accuracy or independent
source evidence. Cluster IDs are learned groups, not assigned thermodynamic
phases. See the [structural-state glossary](research_glossary.md#shared-structural-pretraining)
and [training protocol](../experiments/structural_pretraining_20260917/README.md).
