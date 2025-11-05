"""
Example usage of the SPD prediction and visualization pipeline.

This script demonstrates how to:
1. Load a trained SPD model from checkpoint
2. Extract latent representations from synthetic training data
3. Perform clustering analysis (KMeans and HDBSCAN)
4. Create visualizations comparing ground truth and predictions

The script automatically loads synthetic data from the model's config.
"""

import os
import sys

sys.path.append(os.getcwd())

from src.training_methods.spd.predict_and_visualize import (
    predict_and_cache,
    perform_clustering,
    create_visualization,
    visualize_reference_structures,
)
from pathlib import Path


def main():
    # Configuration
    checkpoint_path = "output/2025-11-05/15-21-37/synth_SPD_FoldingSphereAttnRes_l36_P80_Sinkhorn_256-epoch=04.ckpt"
    n_phases = 3  # Expected number of phases for KMeans
    output_dir = "output/spd_analysis"
    cache_dir = "output/spd_analysis/predictions_cache"
    max_samples = None  # Set to limit number of samples (e.g., 1000), or None for all

    # Path to reference structures (set to None to skip)
    reference_structures_path = "output/synthetic_data/baseline_box_no_perturb/reference_structures.npy"

    # Step 1: Extract latents (or load from cache)
    print("\n=== Step 1: Extracting latents from synthetic data ===")
    (latents, sample_coords, phase_labels, grain_labels,
     model, cfg, atoms, metadata) = predict_and_cache(
        checkpoint_path=checkpoint_path,
        cuda_device=0,
        cache_dir=cache_dir,
        force_recompute=False,  # Set to True to recompute
        max_samples=max_samples,
    )

    print(f"Latents shape: {latents.shape}")
    print(f"Sample coords shape: {sample_coords.shape}")
    print(f"Phase labels shape: {phase_labels.shape}")
    print(f"Atoms shape: {atoms.shape}")

    # Step 2: Perform clustering
    print("\n=== Step 2: Performing clustering ===")
    kmeans_labels, hdbscan_labels = perform_clustering(latents, n_phases)

    # Step 3: Create visualization
    print("\n=== Step 3: Creating visualization ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Infer box size from metadata
    box_size = metadata.get("box_size", 160.0)

    create_visualization(
        atoms=atoms,
        metadata=metadata,
        sample_coords=sample_coords,
        phase_labels_sample=phase_labels,
        grain_labels_sample=grain_labels,
        kmeans_labels=kmeans_labels,
        hdbscan_labels=hdbscan_labels,
        box_size=box_size,
        output_path=output_path / "clustering_visualization.png",
    )

    print(f"\nClustering visualization saved to {output_path / 'clustering_visualization.png'}")

    # Step 4: Create reference structures visualization (optional)
    if reference_structures_path:
        print("\n=== Step 4: Creating reference structures visualization ===")
        visualize_reference_structures(
            checkpoint_path=checkpoint_path,
            reference_structures_path=reference_structures_path,
            output_path=output_path / "reference_structures_analysis.png",
            cuda_device=0,
            target_atoms=80,
            box_size=10.0,
        )
        print(f"Reference structures visualization saved to {output_path / 'reference_structures_analysis.png'}")

    print(f"\n=== Done! ===")
    print(f"All outputs saved to {output_path}")


if __name__ == "__main__":
    main()
