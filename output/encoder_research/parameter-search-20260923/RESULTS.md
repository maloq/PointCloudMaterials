# Encoder parameter search: evidence and decisions

**Training update:** [replicated VISReg gains and remaining limitations](INTERIM-20260923-2050.md),23September20:50CEST. The review below records the pre-training evidence cut.

The numerical snapshot queue remains active. This analysis freezes **29 of40** completed native checkpoints; all29 completed without a recorded failure at the evidence cut. Earlier35-pass GeoFrame and completed two-seed distance/future results are also considered. New fits are a development parameter search, not independent final validation.

## What changed our understanding

The historical **VISReg epoch159 raw encoder** has the best liquid-neighbor error among these29 checkpoints (1.290) and the highest nonbulk Al planar-fault AP (0.952). Its two-blob UMAP appearance is therefore insufficient evidence that useful liquid distinctions disappeared. The epoch34 VICReg and epoch159 VISReg images also differ in objective and duration. A matched-budget VISReg comparison is necessary.

The broad MACE export has the best mean liquid-order readout in this cut (R²0.425), while geometry MACE gives stronger local boundary sensitivity (AUROC~0.571 versus0.535 for VISReg159). These are different useful properties. The more physically resolved VISReg representation still adds no demonstrated calibrated crystallization benefit.

| Encoder | Liquid order R² ↑ | Liquid-neighbor error ↓ | Nonbulk fault AP ↑ | Nonbulk boundary AP ↑ | Spatial AUROC ↑ | Brier difference vs physics ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| geoframe-visreg-epoch159 | 0.410 | 1.290 | 0.952 | 0.706 | 0.535 | +0.00286 |
| mace-expanded-dual-20260919-last | 0.425 | 1.512 | 0.759 | 0.646 | 0.567 | +0.00440 |
| factorial23-B-relaxed | 0.399 | 1.481 | 0.809 | 0.670 | 0.571 | +0.00524 |
| relaxed-cold-control | 0.285 | 1.610 | 0.839 | 0.611 | 0.549 | +0.00025 |
| relaxed-cold-vic-temp001 | 0.286 | 1.604 | 0.832 | 0.603 | 0.551 | -0.00059 |
| relaxed-cold-sig-temp3 | 0.273 | 1.668 | 0.778 | 0.581 | 0.541 | -0.00073 |

These are mean Al static-frame scores plus the existing15-root dynamic development assay. Nonbulk AP excludes bulk crystalline contexts and differs from earlier full-cohort AP. Error units use fitting-only target scaling. Frame averages are not confidence intervals; Brier intervals and coverage remain in the linked tables.

![Observed structural and predictive tradeoffs](plots/evidence-tradeoffs.png)

## Regularization and prediction

Within the temperature-scoped relaxed SIGReg pair, raising weight1→3 increases rank1.62→2.18 while liquid-neighbor error worsens1.636→1.668 (~1.9%) and nonbulk fault AP falls0.799→0.778. This single-seed comparison argues against optimizing rank alone; it does not establish a universal best weight. Mild cold VICReg0.01 retains more order and faults than weight0.1, but provides no resolved Brier improvement.

Across the29 raw exports, liquid-order R² correlates with **higher/worse Brier difference** (Spearman+0.617). Liquid-neighbor error correlates with Brier difference at−0.823: the better structural neighbors are associated with worse calibrated risk scores in this mixed-family cut. These associations are confounded by family, training data, budgets and shared initialization; no p-values or causal claims are made. Within-family tables are retained. The practical conclusion is to measure prediction separately, not to reject structure preservation.

The completed two-seed distance/future factorial found only~0.27% lower withheld-neighbor error at distance weight0.1, below its1% threshold. Future-residual supervision did not replicate a future-order benefit and slightly harmed native radial reconstruction. The new MACE sweep increases learning rate and distance strength independently and leaves the unproductive future loss off.

## Metric audit

The old topology mean R² is dominated by rare columns with tiny evaluation variance (one frame contains per-column R² below−28000 even though many ordinary columns are positive). We keep those historical values. The added assay reports fit-standardized NMSE and raw-space liquid-neighbor error, with identical fitting-only column selection for every model. This is a new documented diagnostic, not a correction silently applied to old exports.

## New training queue

**28 fits /14 recipes /two seeds per recipe.** Twenty GeoFrame fits: eight VICReg combinations (MLP/identity × covariance1/5 × FactorVAE off/on) and two VISReg combinations (MLP/identity, historical lambda0.4 and FactorVAE0.1). Eight MACE fits: encoder LR1e-5/1e-4 × distance0/1.

GeoFrame runs35 complete passes, assessed initially and at12/24/35. Native MACE runs4096 sampled updates, assessed initially and at1024/2048/4096; its sampling protocol does not define literal shuffled epochs. Encoder and projector exports remain separate. Frozen code, matched initialization, batch/augmentation seeds, and the existing assay cohorts make within-family changes interpretable.

Promotion requires at least1% lower liquid-neighbor error in both seeds with no>2% worse order/topology NMSE, no>0.02 AP or spatial-AUROC loss, and all GeoFrame materials reported. Predictive promotion additionally requires improved Brier and no worse residual future error in both seeds, with selected-step0 probes excluded. These are practical exploratory thresholds; uncertainty and sparse event coverage remain visible. Pareto candidates retain the tradeoffs instead of receiving a single weighted score.

[Live training comparison](index.html) · [Scientific protocol](../../../experiments/encoder_parameters_20260923/README.md) · [Snapshot review CSV](tables/snapshot-review.csv) · [Exploratory associations](tables/exploratory-associations.csv) · [Definitions](tables/METRICS.md) · [Two-seed distance/future evidence](../../structural_state/future-metric-20260923/RESULTS.md)

No fresh Ta/Zr nucleation-fate evidence was added. Static potential provenance remains unknown; apparent ordered-liquid regions remain candidates, not validated nuclei.
