# Static Al rerun with the selected v6 GATr encoder

Question: how does the newest Al-only GATr–VICReg state organize the same six
static Al configurations used in the earlier analysis?

The completed `gatr-vicreg-al-stable-20260918` run has 1,465 updates. Use its
selected update-1,216 encoder, with checkpoint hash pinned in
`configs/analysis/structural_gatr_v6_static.json`. Its score 0.22607703506946564
uses Al-only training target moments, so it is not numerically comparable with
the earlier broad-material checkpoint-selection score.

Keep the six frames (166, 170, 174, 175, 177, 240 ps), 684,723-center interior
grid, seven spherical clusters and complete static analysis from the
[previous protocol](../structural_pretraining_20260917/STATIC_AL.md). Input support
and fixed Al scale remain unchanged. The new architecture and its native
protected-geometry/selective-BF16 execution policy are part of this rerun;
raw encoder z128 is analyzed without projector or calibrated prediction heads.
Clustering is fitted independently, so identically numbered clusters in the two
analyses need not correspond.

```bash
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_v6_static.json --stage export
python -m src.analysis.structural_adapter --config configs/analysis/structural_gatr_v6_static.json --stage verify
python -m src.analysis.pipeline configs/analysis/static_structural_gatr_v6_al.yaml
```

Results: `output/structural_static/gatr-vicreg-v6-step1216-al-20260918/`.
Native static inputs/output matched exactly, and all 480 saved compiled
selection states agreed with eager inference to maximum absolute error
8.344650268554688e-7. Five targeted static-input tests passed.

The full analysis completed on RTX PRO 6000 in 755 seconds. All 684,723 center
coordinates and six source hashes match the earlier analysis. The seven-cluster
fit converged; cosine silhouette is 0.3840 on its 3,000-center diagnostic sample.
C7 (stored ID 6) rises from 0.153% at 166 ps to 85.461% at 240 ps. Its global
representative is classified FCC by both PTM and adaptive CNA; representative
labels do not establish a phase label for every member of a cluster.

Raw PCA uses seven components for 95% variance, with 33.973% in PC1 on the
fixed 8,000-center sample. The initial float32 covariance calculation suffered
cancellation, so raw PCA and latent summaries were regenerated in float64 from
the inference cache. The regression test reproduced the error before the fix
and passes after it. Every other metric group and 144 cached/spatial/cluster
artifacts stayed unchanged; original diagnostics and their definitions are
archived inside this run.

Open the [gallery](../../output/structural_static/gatr-vicreg-v6-step1216-al-20260918/index.html)
and [findings and verification](../../output/structural_static/gatr-vicreg-v6-step1216-al-20260918/RESULTS.md).
These six static files are training inputs. This is descriptive analysis, not
held-out generalization or a controlled single-factor comparison with the old
model. Numbered cluster IDs are specific to each independently fitted run.

See [execution and precision details](../../docs/structural_static_analysis.md)
and the [structural-state glossary](../../docs/research_glossary.md#shared-structural-pretraining).
