# SPD Prediction and Visualization Pipeline

This module provides tools for analyzing trained SPD (Shape-Pose Disentanglement) models through latent space clustering and visualization using **synthetic atomistic data**.

## Key Features

- **Automatic Data Loading**: Loads synthetic training data directly from model config
- **Ground Truth Integration**: Uses actual atomic positions and metadata from training data
- **Latent Extraction**: Extracts invariant latent representations with spatial coordinates
- **Caching**: Automatic caching of predictions to avoid recomputation
- **Clustering Analysis**:
  - KMeans clustering (with user-specified number of clusters)
  - HDBSCAN clustering (automatic cluster detection)
- **Comprehensive Visualization**: 5-panel comparison showing:
  1. **Ground truth phases** (from synthetic metadata)
  2. **Ground truth grains** (from synthetic metadata)
  3. **Ground truth grain boundaries** (from synthetic metadata)
  4. **KMeans cluster predictions** (placed at sample centers)
  5. **HDBSCAN cluster predictions** (placed at sample centers)
- **Reference Structures Analysis**: Visualize reference structures with:
  - Original structure samples (generated from structure definitions)
  - Reconstructions from the SPD model
  - Latent space visualization (PCA projection)

All visualizations use the same visual style as `src/data_utils/synthetic/visualization.py`.

## Files

- `predict_and_visualize.py`: Main pipeline implementation
- `example_predict_viz.py`: Example usage script
- `README_PREDICTION_VIZ.md`: This documentation

## Requirements

The pipeline requires:
- A trained SPD model checkpoint
- The model must have been trained on `SyntheticPointCloudDataset`
- The model's config must contain one of:
  - `data.env_dirs` (list of environment directories)
  - `data.data_path` (single environment directory)
  - `data.synthetic.data_dir` (single environment directory)

## Usage

### Command Line Interface

```bash
python src/training_methods/spd/predict_and_visualize.py \
    --checkpoint <path_to_checkpoint.ckpt> \
    --n-phases 3 \
    --output-dir output/spd_analysis \
    --cache-dir predictions_cache
```

### Arguments

- `--checkpoint`: Path to trained SPD model checkpoint (required)
- `--n-phases`: Number of phases for KMeans clustering (default: 3)
- `--box-size`: Size of simulation box for visualization (default: auto-detected from metadata)
- `--output-dir`: Directory to save outputs (default: output/spd_predictions)
- `--cache-dir`: Directory for caching predictions (default: predictions_cache)
- `--force-recompute`: Force recomputation even if cache exists
- `--cuda-device`: CUDA device ID (default: 0)
- `--max-samples`: Maximum number of samples to process (default: None = all)
- `--reference-structures`: Path to reference_structures.npy for structure analysis (optional)

### Python API

```python
from src.training_methods.spd.predict_and_visualize import (
    predict_and_cache,
    perform_clustering,
    create_visualization,
    visualize_reference_structures,
)
from pathlib import Path

# Step 1: Extract latents from synthetic data
# (Automatically loads synthetic data from model's config)
(latents, sample_coords, phase_labels, grain_labels,
 model, cfg, atoms, metadata) = predict_and_cache(
    checkpoint_path="path/to/checkpoint.ckpt",
    cuda_device=0,
)

# Step 2: Perform clustering
kmeans_labels, hdbscan_labels = perform_clustering(latents, n_phases=3)

# Step 3: Create visualization
create_visualization(
    atoms=atoms,
    metadata=metadata,
    sample_coords=sample_coords,
    phase_labels_sample=phase_labels,
    grain_labels_sample=grain_labels,
    kmeans_labels=kmeans_labels,
    hdbscan_labels=hdbscan_labels,
    box_size=100.0,
    output_path="output/visualization.png",
)

# Step 4: (Optional) Visualize reference structures
visualize_reference_structures(
    checkpoint_path="path/to/checkpoint.ckpt",
    reference_structures_path="path/to/reference_structures.npy",
    output_path=Path("output/reference_structures_analysis.png"),
    cuda_device=0,
    target_atoms=64,
    box_size=10.0,
)
```

## Output Files

### Cached Predictions

Predictions are cached in `<cache_dir>/<checkpoint_name>_synth_predictions.npz`:
- `latents`: (N, latent_dim) array of latent codes
- `sample_coords`: (N, 3) array of sample center coordinates
- `phase_labels`: (N,) array of ground truth phase labels
- `grain_labels`: (N,) array of ground truth grain labels

### Cluster Assignments

Cluster assignments are saved in `<output_dir>/cluster_assignments.npz`:
- `kmeans`: (N,) array of KMeans cluster labels
- `hdbscan`: (N,) array of HDBSCAN cluster labels

### Visualization

A comprehensive 5-panel visualization is saved as `<output_dir>/clustering_visualization.png`.

### Reference Structures Analysis (Optional)

If `--reference-structures` is provided, an additional visualization is saved as `<output_dir>/reference_structures_analysis.png` with:
- Row 1: Original reference structure samples (generated from structure definitions)
- Row 2: Reconstructions from the SPD model
- Row 3: Latent space visualization (PCA projection, colored by phase)

## Requirements

- torch
- numpy
- matplotlib
- scikit-learn
- hdbscan
- tqdm

## Example

See `example_predict_viz.py` for a complete working example.

```bash
python src/training_methods/spd/example_predict_viz.py
```

## Notes

- Ground truth visualization uses actual atomic positions from the synthetic environment
- Prediction visualization uses sample center coordinates (where point clouds were sampled)
- The visualization samples up to 6000 points per panel for performance
- HDBSCAN noise points are shown in light gray
- Colors are automatically assigned to phases/clusters
- The visualization style matches `src/data_utils/synthetic/visualization.py`

## Troubleshooting

### "Config does not contain synthetic data path"

This error means the model was not trained on synthetic data, or the config format is not recognized. This pipeline **requires** a model trained on `SyntheticPointCloudDataset`. The checkpoint config must contain one of: `data.env_dirs`, `data.data_path`, or `data.synthetic.data_dir`.

### "atoms.npy not found"

Make sure the synthetic environment directories specified in the model's config still exist and contain both `atoms.npy` and `metadata.json`.

### CUDA out of memory

Reduce the `--max-samples` parameter to process fewer samples, or reduce the batch size in the model config.

### Different number of clusters than expected

- For KMeans: Adjust the `--n-phases` parameter
- For HDBSCAN: Tune `min_cluster_size` and `min_samples` in `perform_clustering()` function (edit the source code)

### Dataset size doesn't match coords warning

This warning occurs if the sampling process generates a different number of samples than expected. This is usually harmless - the code automatically truncates to match sizes.
