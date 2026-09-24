# Finding and using analysis and visualization

[Handbook](README.md) · [Searchable catalogue](../../output/encoder_research/catalogue/index.html)

Use the catalogue's **Reports, tables and galleries** view, filter by family and
search a run/checkpoint name. `gallery` finds existing HTML pages, `report` the
interpretation, and `csv` the numbers. The database contains original records
behind those tables; figures remain in their original run. The older
[general registry](../../output/registry/index.html) also covers simulations and
execution state, which this encoder-specific catalogue intentionally excludes.

## Existing analysis workflows

| Workflow | What it does | Entry point / retained reference |
| --- | --- | --- |
| Standard checkpoint pipeline | Native embeddings, clustering, PCA/UMAP/t-SNE, representatives, spatial views, temporal/MD and optional topology stages | `python -m src.analysis.pipeline CONFIG.yaml`; [template](../../configs/analysis/static.yaml), [pipeline](../../src/analysis/pipeline.py) |
| Structural MACE/GATr export | Reconstruct exact native support/precision; export and verify states before standard plots | [static adapter guide](../structural_static_analysis.md), `src.analysis.structural_adapter` |
| FactorVAE/GeoFrame archive | Original static analyses, representatives and projections; no re-inference required | [four restored galleries](../../output/factor_vae_archive/index.html) |
| MACE context/recovery | Smooth-boundary interventions, inner/center readouts, instantaneous/relaxed TDA and physical changes | [context records](../../experiments/mace_context_20260914/README.md), [recovery](../../experiments/mace_context_recovery_20260914/README.md) |
| Conditional information | Compare embeddings with radial/geometry controls; add-backs and matched physical/onset tests | [local622 results/gallery](../../experiments/gatr_conditional_information_20260918/LOCAL_LAST622.md), `configs/analysis/conditional_information_local_last.json` |
| Equivariant stream analysis | Directional turns, rotations, cage corrections and interactive3D | [GATr directional study](../../experiments/gatr_equivariant_20260918/README.md) |
| Trajectory stability | Common trajectory/lag comparisons with train-reference scales and source uncertainty | [protocol](../../experiments/trajectory_stability_20260918/README.md), `configs/analysis/trajectory_stability.json` |
| State and movement spectra | Whole-dataset, within-track and temporal-increment dimensions; exact-lag stability and per-track coverage | [guide](embedding_dynamics.md), `python -m src.research.trajectory_stability.audit --config configs/analysis/encoder_dynamics_20260924.json` |
| Input-noise response | Matched Gaussian perturbations, normalized response, numerical floor and ratio to natural 0.75 ps motion | [guide](input_noise.md), `python -m src.research.trajectory_stability.noise --config configs/analysis/encoder_noise_20260924.json` |
| Liquid geometry audit | Rank, neighborhoods, four distance interventions, conditional physical/future probes and matched onset associations | [execution guide](../liquid_geometry.md), `src.research.liquid_geometry.study` |
| BCR conditioning audit | Code swaps/constants, fresh decoders, initial/final and pooled/exported probes, relaxed transfer | [findings and command](../../experiments/bcr_followup_20260922/RESULTS.md) |
| Structural-state analysis | Actual training heads, fixed physical probes, raw-distance neighbors, frozen hazards and paired comparisons | [execution guide](../structural_state.md), [repaired review](../../output/structural_state/repaired-review-20260923/README.md), [factorial](../../output/structural_state/future-metric-20260923/RESULTS.md) |
| Frozen forecasting/context | Decoded physical errors, path distributions, risk/PR/reliability curves and timing | [forecast follow-up](../crystallization_followup.md), [structured-context figures](../../configs/analysis/structured_context_figures.json) |

These are different scientific protocols, not flags to combine indiscriminately.
Use a saved run configuration or copy an appropriate active recipe into a new
output directory. Generic `static.yaml` still names a historical example checkpoint;
it is a template, not a guaranteed ready-to-run latest-model configuration.
Some old analysis paths require their saved source archive. Check the linked
protocol before rerunning.

## What our visualizations mean

**UMAP/t-SNE/PCA.** Useful for finding dominant variation, outliers and model
failure. Cluster islands can reflect phase, density, source identity, temperature,
preprocessing or projection settings. Quantitative clustering scores are calculated
in the declared latent/preprocessed space, not inferred from island separation.
Compare identical sampled observations and preprocessing; show physical colors and
within-liquid views. Do not fit independent projections and treat axis locations
as aligned coordinates. Raw covariance diagnostics use float64 centering/SVD where
small variation sits on a large mean; a past float32 diagnostic produced spurious
variance structure.

**Spatial maps and representatives.** The standard pipeline maps cluster labels
and observables back to atomic neighborhoods, shows frame proportions and real
representative structures, and explores connected regimes. These can reveal
interfaces and smooth physical gradients that a projection hides. Cluster IDs are
arbitrary across models. Six-snapshot static analyses overlap some training data;
they are descriptive, not held-out structural accuracy. An interior regular grid
is not an exhaustive per-atom scan.

**Temporal plots.** Show tracked identities, natural-time embedding increments,
physical order and alarms together. Include observation/history length, lag and
boundary changes. A plot that re-encodes observed future snapshots is not an
autonomous forecast. Interpolated continuity probes are not simulated dynamics.
For forecasts, display persistence and uncertainty, and count missed events along
with detected-event timing.

**PR, reliability and paired effects.** PR curves/AP show ranking; reliability
plots and Brier/log loss assess probabilities. Include the natural source-weighted
prevalence, count support and empty-bin behavior. Paired source-difference plots
are more informative than unrelated bars with pooled error bars. Keep seed effects
visible: the latest factorial's constant second-seed MLP is more important than
small differences between objective variants.

**Neighbor and conditional-information plots.** Display the actual neighbor
structures and withheld physical discrepancies, with temperature/order-matched
controls. High rank or a visually broad liquid cloud is not enough. Associations
across related trained models are descriptive; four heads on one encoder are not
four independent encoder observations.

## Fast, reproducible analysis practice

1. Pin the checkpoint hash, export feature (`encoder`, `projector`, pooled or typed
   tensor stream), native graph support, scaler and producer revision.
2. Reuse the saved row manifest and cache **with its metadata**. Run native/batch
   verification when exporting a new architecture or precision path. Do not reuse
   a cache merely because shapes match.
3. Compute the inexpensive scalar/physical/onset tables before expensive manifold
   projections or rendering. Use existing arrays for a report-only refresh.
4. Save readable `plots/`, `tables/` and a short result report; put predictions,
   configs, hashes, logs and source receipts in `technical/`.
5. Preserve each export's metric contract. Register the result here, refresh the
   catalogue, and link the finding from its scientific protocol.

`runtime.profile: fast` in the standard template avoids selected expensive
rendering/projection checks; inspect its explicit settings rather than assuming
it is the same figure set. `figure_only` requires compatible retained inference
arrays and metadata. Some archived regenerable caches were intentionally pruned;
plots and metrics remain useful, but figure-only reruns can require full inference
first. See [output registry/cache rules](../output_registry.md).

The new catalogue itself is an offline HTML file: no service, account or external
JavaScript is needed. Opening local links works with the configured storage
mounts. For browser access through HTTP, serve the repository, not only the
catalogue directory, so relative documentation links remain reachable:

```bash
python -m http.server 8765 --bind 127.0.0.1
# /output/encoder_research/catalogue/index.html
```

## Broad matched snapshot screen, September23

The [40-checkpoint queue](../encoder_screen.md) reuses the 37 completed GeoFrame
reference evaluations and applies the same physical reference and conditioned
future readouts to native MACE, GATr, JEPA, relaxed and VISReg exports. Numerical
results and cached-embedding spatial/UMAP/context panels are separate stages.
The [result page](../../output/encoder_research/screen-20260923/index.html) shows
completed, pending and failed tasks; a submitted model is not a measured result.
Native observation budgets and training data differ, so a cross-family difference
does not isolate an architecture or loss effect.

The current screen's spatial panels have eightfold sampling within their original
slices, with independently assigned reference labels and the original KMeans
fit. Dense plot sampling is separate from the fixed numerical comparison cohort.
