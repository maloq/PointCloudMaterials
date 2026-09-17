# Selected GATr–VICReg state on the standard Al snapshots

Question: what local structural groups and spatial patterns are present in the
128-channel center state learned by the RTX PRO 6000 structural-pretraining run?

Use the best selected encoder at update 3,072, after the run completed 4,096
updates. Its selection score is 0.4045023210346699 under the existing fifteen-source
selection protocol. Selection is not a held-out test. The exact checkpoint hash
is pinned in `configs/analysis/structural_gatr_static.json`.

Analyze Al inherent configurations at 166, 170, 174, 175, 177 and 240 ps. Use
all 684,723 centers from the existing three-edge-layer context grid, with overlap
0.5. The additional edge exclusion is needed for complete trained support.
These snapshots overlap the broad training sources, so this is a descriptive
representation analysis. Snapshot times are from one source lineage and do not
form independent test replicates.

Extract raw encoder z128 using every atom inside the fixed trained 16.869063
Angstrom Al support. Match native species, center identity, fixed material scale,
15–17 model-unit taper and FP32 arithmetic. The projector and physical/TDA
decoders are not part of this representation. Representative displays retain the standard normalized local cropped clouds;
spatial maps retain the source center coordinates in Angstrom.

Keep the standard `configs/analysis/static.yaml` analysis settings: seven
spherical clusters after channel standardization, PCA retaining 99% variance
(up to 64 components) and row L2 normalization; t-SNE/UMAP, connected regimes,
representatives, PTM/CNA, proportions and spatial figures. Cluster numbers are
descriptive learned groups rather than assigned phases.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_al.yaml
```

Results: `output/structural_static/gatr-vicreg-step3072-al-20260918/`.
Native prepared-example inputs and output matched exactly. Changing batch size
and center order on six real Al examples changed raw states by at most
2.384185791015625e-7. The full analysis completed successfully in 606 seconds;
the maximum replay difference across all six frames was 3.5762786865234375e-7.
The first raw principal component explains 98.33% of variance on the standard
8,000-sample subset. C1 grows from 0.033% at 166 ps to 84.71% at 240 ps, with
face-centered-cubic selected representatives at 170–240 ps (hexagonal-close-packed
at 166 ps). Read the [completed results](../../output/structural_static/gatr-vicreg-step3072-al-20260918/RESULTS.md)
and [gallery](../../output/structural_static/gatr-vicreg-step3072-al-20260918/index.html). See the
[execution and input contract](../../docs/structural_static_analysis.md).
