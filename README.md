# Pytorch Implementation of PointNet

## Installation

### Create a new uv environment

```bash
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
```

### Install pytorch3d

```bash
conda install -c fvcore -c conda-forge fvcore -y
pip install iopath black usort flake8 flake8-bugbear flake8-comprehensions scikit-image matplotlib imageio plotly opencv-python
conda install pytorch3d -c pytorch3d-nightly -y
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

### Basic usage

```bash
python src/training_methods/contrastive_learning/predict_and_visualize.py \
    path/to/checkpoint.ckpt
```

Results are written to `<checkpoint_dir>/analysis/` by default.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `checkpoint_path` | *(required)* | Path to a trained `.ckpt` file |
| `--output_dir DIR` | `<ckpt_dir>/analysis` | Directory for analysis outputs |
| `--cuda_device N` | `0` | CUDA device index |
| `--max_batches_latent N` | all | Limit batches used for latent collection |
| `--max_samples_visualization N` | config value | Cap samples for t-SNE |
| `--data_file FILE` | config value | Override input data files (repeat for multiple) |
| `--visible_cluster_sets IDS [...]` | none | Cluster ID subsets for separate views (comma-separated per set) |
| `--pretty_render_resolution N` | `2200` | Image width/height in pixels for sphere renders |
| `--pretty_render_sphere_radius N` | `7` | Sphere radius in pixels for sphere renders |

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
| `clustering_analysis.png` | Clustering quality metrics |
| `equivariance.png` | Equivariant latent error distribution |
| `md_space_clusters.png` | 3D MD-space cluster scatter |
| `md_space_clusters.html` | Interactive 3D Plotly version |
| `cluster_figure_set_k<K>/` | Fixed-k figure set (see below) |
| `cluster_profiles_by_k/` | Per-cluster structure & property profiles |

#### Cluster figure set (`cluster_figure_set_k<K>/`)

Every MD cluster view is produced in two versions: a fast matplotlib scatter
and a Blinn-Phong sphere render (`*_pretty.png`) comparable to OVITO.

| File | Description |
|---|---|
| `01_md_clusters_all_k<K>.png` | MD space with all clusters (view 1) |
| `01_md_clusters_all_k<K>_view2.png` | Same, rotated 90 degrees |
| `01_md_clusters_all_k<K>_view3.png` | Same, rotated 180 degrees |
| `01_md_clusters_all_k<K>_view4.png` | Same, rotated 270 degrees |
| `01_*_pretty.png` | Sphere renders of the above (always generated) |
| `02_md_clusters_set_<IDS>_k<K>.png` | Selected cluster subset (if `--visible_cluster_sets` given) |
| `02_*_pretty.png` | Sphere render of each subset view |
| `03_cluster_count_icl_k<K>.png` | ICL curve vs number of clusters |
| `04_cluster_representatives_k<K>.png` | Nearest-centroid representative per cluster |

### Cluster subset views (`--visible_cluster_sets`)

To produce separate figures showing only specific clusters, pass one or more
comma-separated sets of cluster IDs:

```bash
python src/training_methods/contrastive_learning/predict_and_visualize.py \
    path/to/checkpoint.ckpt \
    --visible_cluster_sets '0,1,2' '3,4,5'
```

This generates `02_md_clusters_set_0-1-2_k6.png` (and `*_pretty.png`) etc.
The same sets can be specified in the Hydra config:

```yaml
analysis_cluster_figure_visible_sets: ["0,1,2", "3,4,5"]
```

Tune the render quality with:

```bash
--pretty_render_resolution 3000   # larger image (default 2200)
--pretty_render_sphere_radius 10  # bigger spheres (default 7)
```

### Hydra configuration

Most analysis behaviour is controlled by Hydra config keys stored alongside the
checkpoint. Key settings include:

| Config key | Default | Description |
|---|---|---|
| `analysis_cluster_k_values` | `[3,4,5,6]` | K values to evaluate |
| `analysis_cluster_method` | `auto` | Clustering algorithm |
| `analysis_cluster_figure_set_enabled` | `true` | Generate the fixed-k figure set |
| `analysis_cluster_figure_set_k` | `6` | K used for the figure set |
| `analysis_cluster_figure_visible_sets` | `[]` | Cluster ID subsets for separate views (list of comma-separated strings) |
| `analysis_tsne_max_samples` | `8000` | t-SNE sample cap |
| `analysis_hdbscan_enabled` | `false` | Run HDBSCAN in addition to KMeans |
| `analysis_cluster_profile_enabled` | `true` | Generate per-cluster structure profiles |
| `analysis_md_use_all_points` | `true` | Use full dataset for MD plots |