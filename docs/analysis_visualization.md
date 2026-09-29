# Analysis visualization reference

Run `python -m src.analysis.pipeline configs/analysis/static.yaml --checkpoint CHECKPOINT
--output-dir output/static-al/review`. Templates live under `configs/analysis/`.
Configure inputs, clustering, visualization, inference cache and equivariance there.
Training-side analysis overrides the runtime checkpoint/output locations.

Checkpoint recipes declare `extends: static.yaml` and contain only their changes
to the common settings. Load them through
`src.analysis.config.load_checkpoint_analysis_config`; paths inside a recipe
are repository-relative, and `extends` is relative to its recipe. Temporal and
TMF templates retain their distinct protocols. Blender, HDBSCAN and SwAV remain
available through their existing configuration blocks.

Numerical analysis runs once per output directory. For a new calculation, choose
a new run or analysis revision. To refresh navigation and expose retained plots,
use [result publication](research_results_system.md). To restyle the saved native
MACE representatives, use the saved-representatives command below. The former
`figure_only` path mixed rendering with fresh diagnostics and has been removed.

New standard analyses publish a named bundle under `analyses/standard-v1/`,
with grouped `plots/`, numerical `tables/` and the native producer tree in `data/`.
The detailed filenames below refer to that native tree. Existing analyses keep
their original root or `technical/` paths; the [publication workflow](research_results_system.md)
can expose their full hierarchy without numerical recomputation.

The named bundle is the plot location: the pipeline does not also create empty
run-level `plots/` or `tables/` directories. Its gallery has an **Interactive views**
section; HTML entry pages also live alongside PNGs under the bundle's `plots/`.
These entry pages open the retained originals in `data/`, preserving relative assets.

Cluster proportions export PNG by default. Set
`real_md.time_series.paper_enabled: true` to additionally save and publish
`cluster_proportions_stacked_area_paper.svg`. This does not affect t-SNE export options.
When refreshing an existing gallery, `publish --record RUN/run.json
--include-paper-svg` exposes an already retained SVG; publication never generates one.

### Output files

The script writes the following into the output directory:

| File | Description |
|---|---|
| `analysis_metrics.json` | All numerical metrics |
| `latent_tsne_clusters.png` | t-SNE coloured by cluster labels |
| `latent_tsne_ground_truth.png` | t-SNE coloured by ground-truth phases (if available) |
| `latent_pca_analysis.png` | PCA projection and explained variance |
| `latent_pca_3d.png` | 3D PCA projection |
| `latent_statistics.png` | Comprehensive latent statistics |
| `equivariance.png` | Equivariant latent error distribution |
| `md_space_clusters.png` | 3D MD-space cluster scatter |
| `md_space_clusters.html` | Interactive 3D Plotly version |
| `cluster_figure_set_k<K>/` | Fixed-k figure set (see below) |
| `real_md_qualitative/` | Real-data qualitative analysis bundle: representatives, time series, spatial views, descriptors, transitions, report |

#### Cluster figure set (`cluster_figure_set_k<K>/`)

Every MD cluster view is produced as a standard matplotlib render. An optional
Blender Cycles raytraced render (`*_raytrace.png`) can also be enabled.

| File | Description |
|---|---|
| `01_md_clusters_all_k<K>.png` | MD space with all clusters (view 1) |
| `01_md_clusters_all_k<K>_view2.png` | Same, rotated 90 degrees |
| `01_*_raytrace.png` | Blender Cycles raytraced renders (when enabled) |
| `02_md_clusters_crystal_like_k<K>.png` | Clusters whose representative center is PTM FCC, HCP, or BCC |
| `02_*_raytrace.png` | Blender Cycles raytraced crystal-like renders (when enabled) |
| `03_cluster_count_icl_k<K>.png` | ICL curve vs number of clusters |
| `04_cluster_representatives_k<K>.png` / `.html` | One sparse, hull-free representative view; identical displayed atoms and edges in PNG and offline HTML |

### Crystal-like cluster views

The analysis detects crystal-like clusters independently for every snapshot.
It runs PTM on each cluster representative and renders only clusters whose
representative center is FCC, HCP, or BCC. The detected IDs are stored in the
figure-set metadata as `crystal_like_cluster_ids`; they are not configured
manually and may differ between snapshots. If a snapshot has no crystal-like
representative, no `02_md_clusters_crystal_like_*` image is written for it.

Raytrace options live under `figure_set.raytrace`. The standard preset uses
two views, 1200 px, 32 Cycles samples, and denoising. Blender is launched once
per snapshot; that process reuses the cluster scene for every all-cluster and
crystal-like view. Set `figure_set.raytrace.high_quality: true` to override the
render size and sample count with the 1600 px / 64-sample quality preset.

The raytraced renderer estimates physically consistent ball size from the full
labeled MD-space cloud, not the sampled render subset. The sphere-size flag acts
as a multiplicative scale on that estimate.

Raytraced outputs require a working Blender executable (`blender`) in PATH
or an explicit absolute path via `figure_set.raytrace.blender_executable`.

### Real MD qualitative workflow

For the real crystallization trajectory, edit the analysis config directly. A
minimal example looks like:

```yaml
checkpoint:
  path: output/2026-03-02/17-22-18/VICREG_FT_l512_N128_M80_RI_MAE_Invariant-epoch=11.ckpt
  output_dir: output/real-md/qualitative

inputs:
  data_config: configs/data/loaders/static_al_80.yaml
  real_data_files: [166ps.npy, 170ps.npy, 174ps.npy, 175ps.npy, 177ps.npy, 240ps.npy]

real_md:
  selected_k: 6
  cluster_groups:
    ordered: [0, 1]
    intermediate: [2, 3]
    liquid_like: [4, 5]
  spatial:
    zoom_specs:
      - name: nucleus
        frame: 240ps.npy
        cluster_ids: [0, 1]
        half_extent: [18.0, 18.0, 18.0]
```

The qualitative bundle is written to `real_md_qualitative/` inside the analysis
directory and includes:

- representative-neighbourhood galleries by cluster
- frame-wise cluster proportion tables and stacked plots
- filtered and zoomed spatial renders
- 2D latent projections coloured by cluster, frame, and optional physical scalar
- per-cluster descriptor summaries
- transition flow diagrams between consecutive frames
- `summary.json` and `README.md` for paper reuse

### Local-order figures

The [MACE local-order gallery](../output/structural_static/mace-epi-epoch12-al-20260926/analyses/order-v1/index.html)
contains three PNG figures and their interactive HTML counterparts:

1. **Cluster representatives:** the same original seven samples and 64 displayed
   atoms. Thin, visible cluster-colored connections are reduced to 67–81 per
   panel (previously 243–264). Keep all cutoff-valid edges within the focal shell;
   outside it show only mutual two-nearest connections within the cutoff. This
   display thinning does not alter any metric or impose connectivity.
2. **Detected local order:** full-source PTM matches highlighted on those same
   neighborhoods, with the same cluster colors. Black outlines and marker shapes
   identify FCC/HCP/BCC/ICO; unmatched atoms are faint. The 0.10 RMSD cutoff and
   enabled templates are explicit. The full source is analyzed before cropping,
   so missing neighbors at the edge of a drawn patch cannot create false disorder.
3. **Order across clusters and samples:** PTM composition for every one of the
   684,723 saved centers, composition by snapshot, and q4/q6 and neighbor q6
   alignment across 2,500 sampled neighborhoods. Up to 64 centers are sampled
   uniformly per cluster/frame with a fixed seed; tiny strata use every member.
   The resulting per-cluster sample sizes are 320–384. Unweighted sampled
   distributions describe the stratified sample, not the population mixture.

All figures retain the white background, seven-cluster palette and dark atom
outlines. The radial tint now uses distance from the actual focal atom with a
stronger pale-to-dark gradient. No shell hull is rendered. Each PNG and HTML pair
shares atoms and edges; the HTML pages include their plotting runtime.

```bash
# New scientific calculation over retained cluster assignments; refuses overwrite.
python -m src.analysis.cluster_order \
  --config configs/analysis/mace_epi_epoch12_order.json --stage compute

# Rendering/publication over that saved calculation; no numerical recomputation.
python -m src.analysis.cluster_order \
  --config configs/analysis/mace_epi_epoch12_order.json --stage render
```

The new `analyses/order-v1/` bundle preserves the original `standard-v1` evidence.
Detailed CSVs, source atom/sample identities, population arrays, RMSD-threshold
sensitivity and frozen [metric definitions](metrics/cluster_order.md) are included.
PTM Other is not a liquid-phase label; ICO records local fivefold order. Related
snapshots and overlapping neighborhoods do not support iid confidence intervals.

### Snapshot representatives

The standard figure-set producer now uses the same sparse geometry and radial
coloring as the local-order gallery. Each snapshot exports one
`04_cluster_representatives_k<K>.png` and its self-contained `.html` counterpart.
Both show the nearest 64 atoms (or the available configured support, if smaller),
with the actual cluster IDs and saved sample indices. PTM/CNA continue to use their
original configured support; display thinning does not change their diagnostics.
The drawing cutoff uses the focal CNA shell when available. With CNA disabled it
uses the explicit display rule `1.2 × median(first 12 neighbor distances)`.
Connections are geometric drawing aids, not inferred chemical bonds.

The former reciprocal-shell and connected-kNN renderers have been removed from
the standard producer. For already published native-MACE runs, regenerate only
the display using frozen representative identities and source coordinates:

```bash
python -m src.analysis.saved_representatives \
  --run output/structural_static/cd-mace128-epoch12-al-20260926
python -m src.analysis.saved_representatives \
  --run output/structural_static/mace-epi-epoch12-al-20260926
```

This validates recorded evidence and source hashes, reconstructs each saved focal
shell, and writes the new PNG/HTML pair without rerunning the encoder, clustering,
PTM or CNA. Source atom IDs, cutoffs, drawn edges, colors and implementation hashes
are recorded in `analyses/standard-v1/technical/representative-rendering-v2.json`.
The publication step removes superseded aliases from `plots/` and the gallery;
original legacy images remain under `data/` with their original hashes. Both
six-snapshot CD-MACE128 and self-supervised EPI galleries have been updated.

## Linked structure exploration

`analyses/exploration-v1/data/explorer.html` is a self-contained offline entry page
with five tabs: synchronized spatial slices, order versus PTM fit, a twelve-sample
atlas for each cluster, full-source radial profiles, and exact native-128D
embedding neighbors with coarse-order-matched controls. All spatial and scatter
points open the same neighborhood inspector. Use its PTM RMSD slider to examine
acceptance sensitivity. It preserves the white background and cluster palette,
strong radial atom tint, dark outlines and sparse connections, without hulls.
PNG atlas pages and a radial profile overview are grouped under bundle `plots/`.
SVG output is disabled unless the explorer config explicitly sets `svg: true`.

```bash
python -m src.analysis.cluster_explorer --config configs/analysis/cd_mace128_explorer.json --stage compute
python -m src.analysis.cluster_explorer --config configs/analysis/cd_mace128_explorer.json --stage render
```

Compute requires the published `standard-v1` and `order-v1` inputs. The same
workflow uses `configs/analysis/mace_epi_explorer.json` for the self-supervised run.
Spatial views use 4,000 uniformly sampled centers per snapshot, and expose that
sampling in the interface. Scatter/atlas/radial use retained frame×cluster strata;
retrieval searches every saved center of the query snapshot, excludes centers
within 20 Å, and reports the metric and candidate counts. See the frozen
[definitions](metrics/cluster_explorer.md). Static frames cannot establish future
crystallization or atom-tracked transitions.

The selected sparse representative figure in `order-v1` supersedes the combined
standard-gallery representatives. Each snapshot retains its own updated PNG/HTML pair. Their original files and hashes remain available as
historical evidence; redundant aliases no longer occupy the current gallery.
