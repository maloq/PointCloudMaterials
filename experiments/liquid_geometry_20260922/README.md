# Does liquid embedding geometry explain crystallization prediction?

Recipe: [liquid_geometry_20260922.json](../../configs/analysis/liquid_geometry_20260922.json).
Motivation: the [plot and provenance audit](../../output/representation_audit/liquid-structure-20260922/RESULTS.md)
found that attractive liquid clusters, dimensionality, physical information and
forecast quality need not agree. The following protocol is fixed before this
study's new metric results. Existing checkpoint and forecast results informed
its hypotheses; this is an exploratory follow-up, not a blinded confirmatory test.

## Questions and experiments

| Question | Matched experiment | Interpretation |
| --- | --- | --- |
| Is useful information hidden by a poor distance metric? | Ten completed encoders × raw, standardized, shrinkage-whitened and physical-residual distance; same held-out observations and31 training neighbors | Better physical/topological/future retrieval after a distance change supports a geometry problem. It does not prove optimal information retention. |
| Does rank alone explain performance? | Within-source liquid rank versus exact-row matched linear/MLP onset forecasts; separate hot/cold domains; shuffled feature control | Rank without better physical neighbors or forecasts is insufficient evidence of useful structure. |
| Does improved geometry accompany the latest forecast improvement? | Observed vs relaxed MACE in the completed symmetric reuse study, paired by source and separately for direct/AR-MSE/mixture/diffusion heads | Source-level associations are descriptive. There are two encoders; four heads are not four new representations. |
| Does later training remove information from the actual encoder? | Genuine same-run early/late MACE and GATr; native producer/support; encoder128 and projector64 assessed separately | Falling physical/current-future probe skill together with worsening geometry would support information erosion. Projector-only deterioration suggests a different intervention. |

The primary population is current original-MD PTM Other, independently classified
from the embeddings, with q6<0.35 sensitivity queries. Source train/test roles are
fixed. Original-MD order targets prevent relaxed encoders receiving easier relaxed
test targets. Temperature-matched neighbors and current-order nuisance baselines
limit trivial temperature/crystal-axis explanations. No outcome labels select
rows or fit distance transformations. KNN onset probabilities do use training
outcomes, as expected for a supervised forecast evaluation.

The future ridge probe must improve on all current order8 descriptors plus
temperature/time. Sparse36ps same-atom drift is reported jointly with rank and
physical skill; a constant representation is not a successful smooth embedding.
The temporal audit uses true identities, not paths drawn through UMAP.

The historical early/late sample lacks PTM labels and uses development selection
sources, so it is explicitly an all-phase mechanistic diagnosis. It must not be
pooled with the primary liquid or latest forecasting cohorts. The original GATr
pair512→5860 is not a claim to reproduce the user's distinct step3072 static plot.

## Decision rules for the next training experiment

- If physical probes remain useful while whitening/physical distance improves
  retrieval, first test a learned metric or physical decoder alongside the frozen
  encoder. Increasing rank pressure alone is not justified.
- If later raw-encoder physical/future skills fall, test preservation of current
  geometry and conditional future structure during training, with the same
  source-held-out evaluation. Projector penalties alone may miss this failure.
- If geometry improves without forecast improvement, retain geometry for
  interpretability but select prediction objectives using held-out onset and
  future-structure scores. Current local structure may not contain sufficient
  history or spatial context for nucleation prediction.

Intervals use500 paired, temperature-stratified source bootstrap draws. They
condition on the existing related trained models and one seed, and do not prove
a universal rank/performance relation. Details and exact formulas are in the
[metric definitions](../../docs/metrics/liquid_geometry.md).

Results: [run directory](../../output/representation_audit/liquid-geometry-20260922/)
and its `RESULTS.md`, `plots/`, `tables/`. Reproduction and Slurm instructions are
in the [workflow](../../docs/liquid_geometry.md).

## Completed CPU findings

The [initial findings](../../output/representation_audit/liquid-geometry-20260922/FINDINGS.md)
record the completed ten-encoder and latest-context analyses. Within seven
relaxed-input encoders, physical-neighbor gain tracks12ps MLP AP (rho0.857),
whereas rank tracks it negatively (rho-0.786). Physical distance fitting can
improve frozen kNN predictions. In the separate latest contextual cohort,
relaxed MACE forecasts improve despite worse recovery of instantaneous hot
order, so geometry metrics are not universal forecast proxies. Source intervals,
domain confounding and related-fit limitations are reported in the findings.
The historical early/late job is now complete. The
[23September update](../../output/representation_audit/liquid-geometry-20260922/UPDATE-20260923.md)
finds exact export collapse in the old GATr final checkpoint, independently
reproduced on CPU and localized to its saturated final readout. MACE shows a
modest decline in current physical decoding and almost unchanged0.75ps future
decoding. These all-phase development endpoints do not establish a universal
liquid-specific erosion mechanism or diagnose the current normalized architecture.
