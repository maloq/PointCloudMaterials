# Analysis visualization reference

Run `python -m src.analysis.pipeline configs/analysis/static.yaml --checkpoint CHECKPOINT
--output-dir output/static-al/review`. Templates live under `configs/analysis/`.
Configure inputs, clustering, visualization, inference cache and equivariance there.
Training-side analysis overrides the runtime checkpoint/output locations.

New human-facing files are in `plots/` and `tables/`; the detailed filenames below
refer to the producer tree under `technical/`. Existing analyses keep the old tree.
See [the current layout](research_layout.md).

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
| `04_cluster_representatives_k<K>*.png` | Representative variants with reciprocal-shell edges and aligned/PCA reference views |
| `04_cluster_representatives_k<K>*_raytrace/cluster_*.png` | Blender ball-and-stick representative renders (when enabled) |

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
