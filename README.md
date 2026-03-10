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
| `--data_config FILE` | none | Load a plain data YAML override such as `configs/data/data_ae_Al_80.yaml` |
| `--analysis_config_override FILE` | none | Merge one or more YAML overrides on top of the checkpoint config |
| `--visible_cluster_sets IDS [...]` | none | Cluster ID subsets for separate views (comma-separated per set) |
| `--raytrace_render_enabled` | `false` | Also generate Blender Cycles `*_raytrace.png` renders |

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
| `02_md_clusters_set_<IDS>_k<K>.png` | Selected cluster subset (if `--visible_cluster_sets` given) |
| `02_*_raytrace.png` | Blender Cycles raytraced subset renders (when enabled) |
| `03_cluster_count_icl_k<K>.png` | ICL curve vs number of clusters |
| `04_cluster_representatives_k<K>*.png` | Representative variants with improved reciprocal-shell edges, camera-aligned/PCA reference views, and cleaner degree-capped edges |
| `04_cluster_representatives_k<K>*_raytrace/cluster_*.png` | Blender ball-and-stick representative renders (when enabled) |

### Cluster subset views (`--visible_cluster_sets`)

To produce separate figures showing only specific clusters, pass one or more
comma-separated sets of cluster IDs:

```bash
python src/training_methods/contrastive_learning/predict_and_visualize.py \
    path/to/checkpoint.ckpt \
    --visible_cluster_sets '0,1,2' '3,4,5'
```

This generates `02_md_clusters_set_0-1-2_k6.png` etc.
The same sets can be specified in the Hydra config:

```yaml
analysis_cluster_figure_visible_sets: ["0,1,2", "3,4,5"]
```

Raytrace settings are configured from Hydra, for example:

```yaml
analysis_cluster_figure_raytrace_enabled: true
analysis_cluster_figure_raytrace_blender_executable: blender
analysis_cluster_figure_raytrace_resolution: 1600
analysis_cluster_figure_raytrace_samples: 64
```

The raytraced renderer estimates physically consistent ball size from the full
labeled MD-space cloud, not the sampled render subset. The sphere-size flag acts
as a multiplicative scale on that estimate.

Raytraced outputs require a working Blender executable (`blender`) in PATH
or an explicit absolute path via `--raytrace_blender_executable`.

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
| `analysis_cluster_figure_raytrace_enabled` | `false` | Enable Blender Cycles raytrace renders |
| `analysis_cluster_figure_raytrace_blender_executable` | `blender` | Blender executable path/name |
| `analysis_cluster_figure_raytrace_resolution` | `1600` | Raytrace image width/height |
| `analysis_cluster_figure_raytrace_max_points` | `null` | Deprecated; raytrace always uses all points |
| `analysis_cluster_figure_raytrace_samples` | `64` | Cycles sample count |
| `analysis_cluster_figure_raytrace_projection` | `perspective` | Raytrace camera projection |
| `analysis_cluster_figure_raytrace_fov_deg` | `34.0` | Raytrace perspective FOV |
| `analysis_cluster_figure_raytrace_camera_distance_factor` | `2.8` | Raytrace camera distance factor |
| `analysis_cluster_figure_raytrace_sphere_radius_fraction` | `0.0105` | Size scale for auto-estimated raytrace ball radius (`0.0105` = 1.0x) |
| `analysis_cluster_figure_raytrace_timeout_sec` | `1200` | Timeout per Blender render |
| `analysis_tsne_max_samples` | `8000` | t-SNE sample cap |
| `analysis_hdbscan_enabled` | `false` | Run HDBSCAN in addition to KMeans |
| `analysis_cluster_profile_enabled` | `true` | Generate per-cluster structure profiles |
| `analysis_md_use_all_points` | `true` | Use full dataset for MD plots |
| `data.analysis_data_files` | `null` | Canonical analysis-frame list for real datasets (`null` = use `data.data_files`) |
| `analysis_real_md_enabled` | `true` | Run the real-MD qualitative pipeline on real datasets |
| `analysis_real_md_k` | `6` | Which `k` to use for qualitative analysis and static real-MD plots |
| `analysis_real_md_cluster_groups` | `{}` | Named cluster subsets for filtered spatial views |
| `analysis_real_md_spatial_zoom_specs` | `[]` | Manual zoom boxes for full/filtered spatial views |
| `analysis_real_md_projection_method` | `umap` | 2D latent projection method (`umap`, `tsne`, `pca`) |
| `analysis_real_md_descriptor_enabled` | `true` | Compute post-hoc physical descriptor summaries |
| `analysis_real_md_descriptors` | see config | Optional Steinhardt/CNA/SOAP descriptor blocks |
| `analysis_real_md_transition_enabled` | `true` | Compute frame-to-frame cluster transition summaries |

### Real MD qualitative workflow

For the real crystallization trajectory, a practical entry point is:

```bash
python src/training_methods/contrastive_learning/predict_and_visualize.py \
    output/2026-03-02/17-22-18/VICREG_FT_l512_N128_M80_RI_MAE_Invariant-epoch=11.ckpt \
    --output_dir outputs/real_md_qualitative_example \
    --data_config configs/data/data_ae_Al_80.yaml
```

To add paper-specific cluster groups / zoom windows without editing the
checkpoint config, create a small override YAML and pass it with
`--analysis_config_override`. Example:

```yaml
analysis_real_md_cluster_groups:
  ordered: [0, 1]
  intermediate: [2, 3]
  liquid_like: [4, 5]

analysis_real_md_spatial_zoom_specs:
  - name: nucleus
    frame: 240ps.npy
    cluster_ids: [0, 1]
    half_extent: [18.0, 18.0, 18.0]
```

The qualitative bundle is written to `real_md_qualitative/` inside the analysis
directory and includes:

- representative-neighbourhood galleries by cluster
- frame-wise cluster proportion tables and stacked plots
- full-box, filtered, and zoomed spatial renders
- 2D latent projections coloured by cluster, frame, and optional physical scalar
- per-cluster descriptor summaries
- transition heatmaps between consecutive frames
- `summary.json` and `README.md` for paper reuse

Use `data.analysis_data_files` as the single canonical way to choose which real
MD frames are analysed.
