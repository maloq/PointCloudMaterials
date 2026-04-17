# Pytorch Implementation of PointNet

## Installation

### Create a new uv environment

```bash
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
```
### Install all other requirements

```bash
pip install -r requirements.txt
```

---

## Post-training analysis

`src/training_methods/contrastive_learning/predict_and_visualize.py` runs a
comprehensive post-training analysis on a trained checkpoint. It produces
latent-space visualisations, clustering diagnostics, MD-space cluster figures,
equivariance evaluation, and more.

### Configuration-driven usage

```bash
python src/training_methods/contrastive_learning/predict_and_visualize.py
```

All checkpoint-analysis settings now live in
`configs/analysis/checkpoint_analysis.yaml`. Edit that file to choose:

- `checkpoint.path`, `checkpoint.output_dir`, and `checkpoint.cuda_device`
- data overrides under `inputs`
- clustering/t-SNE/HDBSCAN under `clustering`, `md`, and `tsne`
- fixed-k figure rendering under `figure_set`
- real-MD qualitative outputs under `real_md`
- inference cache and equivariance settings under `cache` and `equivariance`

`predict_and_visualize.py` no longer accepts CLI flags. If arguments are passed,
it raises an error and points back to the analysis config.

Training-side auto-analysis uses the same config file and only overrides the
runtime checkpoint/output paths after training finishes.

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
| `01_md_clusters_all_k<K>_view3.png` | Same, rotated 180 degrees |
| `01_md_clusters_all_k<K>_view4.png` | Same, rotated 270 degrees |
| `01_*_raytrace.png` | Blender Cycles raytraced renders (when enabled) |
| `02_md_clusters_set_<IDS>_k<K>.png` | Selected cluster subset (if `figure_set.visible_cluster_sets` is set) |
| `02_*_raytrace.png` | Blender Cycles raytraced subset renders (when enabled) |
| `03_cluster_count_icl_k<K>.png` | ICL curve vs number of clusters |
| `04_cluster_representatives_k<K>*.png` | Representative variants with reciprocal-shell edges and aligned/PCA reference views |
| `04_cluster_representatives_k<K>*_raytrace/cluster_*.png` | Blender ball-and-stick representative renders (when enabled) |

### Cluster subset views

To render only selected clusters, set `figure_set.visible_cluster_sets` in
`configs/analysis/checkpoint_analysis.yaml`, for example:

```yaml
figure_set:
  visible_cluster_sets:
    - [0, 1, 2]
    - [3, 4, 5]
```

Raytrace options live under `figure_set.raytrace`.

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
  output_dir: outputs/real_md_qualitative_example

inputs:
  data_config: configs/data/data_ae_Al_80.yaml
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
