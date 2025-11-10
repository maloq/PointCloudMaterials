"""
Example usage of the SPD prediction and visualization pipeline.

This script demonstrates how to:
1. Load a trained SPD model from checkpoint
2. Extract latent representations from synthetic training data
3. Perform clustering analysis (KMeans and HDBSCAN)
4. Create visualizations comparing ground truth and predictions
5. Compare reference structures with actual dataset samples (NEW!)

The script automatically loads synthetic data from the model's config.

NEW FEATURES:
- Compare reference structures (from reference_point_clouds.npy) with actual dataset samples
- Visualize preprocessing at each step with diagnostic output
- 4-row visualization: Originals, Canonicals, Reconstructions, Latent space
- Color-coded: Blue=Reference, Green=Dataset, Purple=Canonical, Orange=Reconstruction

Use this to investigate preprocessing inconsistencies or reference structure issues.
"""

import os
import sys

sys.path.append(os.getcwd())

from src.training_methods.spd.predict_and_visualize import (
    predict_and_cache,
    perform_clustering,
    create_visualization,
    visualize_reference_structures,
    create_synthetic_dataset_with_coords,
)
from pathlib import Path


def main():
    # ========================================================================
    # CONFIGURATION - Edit these paths to match your setup
    # ========================================================================

    # Path to your trained model checkpoint
    checkpoint_path = "output/2025-11-09/21-21-24/synth_SPD_FoldingSphereAttnRes_l48_P80_sinkhorn+chamfer_1500-epoch=74.ckpt"

    # Expected number of phases for KMeans clustering
    n_phases = 3

    # Output directory for all visualizations
    output_dir = "output/spd_analysis"
    cache_dir = "output/spd_analysis/predictions_cache"

    # Number of samples to process (None = all samples)
    max_samples = None  # Set to e.g., 1000 for faster testing

    # Path to reference point clouds (set to None to skip reference analysis)
    reference_structures_path = "output/synthetic_data/baseline_box_no_perturb/reference_point_clouds.npy"

    # ========================================================================
    # COMPARISON SETTINGS - For investigating preprocessing issues
    # ========================================================================

    # Compare reference structures with actual dataset samples
    # This helps identify preprocessing inconsistencies
    compare_with_dataset = True  # Set to False for faster execution

    # Limit dataset samples for comparison (to avoid memory issues)
    max_samples_for_comparison = 5000

    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================

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

        # Create dataset for comparison if requested
        dataset_for_viz = None
        if compare_with_dataset:
            print(f"Creating dataset for comparison (max {max_samples_for_comparison} samples)...")
            # Get environment directories from the loaded config
            env_dirs = None
            if hasattr(cfg.data, 'env_dirs'):
                env_dirs = cfg.data.env_dirs
            elif hasattr(cfg.data, 'data_path'):
                env_dirs = [cfg.data.data_path] if isinstance(cfg.data.data_path, str) else cfg.data.data_path

            if env_dirs:
                dataset_for_viz, _ = create_synthetic_dataset_with_coords(
                    env_dirs, cfg, max_samples=max_samples_for_comparison
                )
                print(f"Dataset created with {len(dataset_for_viz)} samples")
            else:
                print("Warning: Could not determine environment directories for dataset comparison")

        visualize_reference_structures(
            checkpoint_path=checkpoint_path,
            reference_structures_path=reference_structures_path,
            output_path=output_path / "reference_structures_analysis.png",
            cuda_device=0,
            dataset=dataset_for_viz,
            compare_with_dataset=compare_with_dataset,
        )

        comparison_note = " (with dataset comparison)" if compare_with_dataset else ""
        print(f"Reference structures visualization{comparison_note} saved to {output_path / 'reference_structures_analysis.png'}")

    print(f"\n=== Done! ===")
    print(f"All outputs saved to {output_path}")
    print("\nVisualization includes:")
    print("  - Clustering visualization: Ground truth vs KMeans vs HDBSCAN")
    if reference_structures_path:
        if compare_with_dataset:
            print("  - Reference structures analysis: Reference (blue) vs Dataset samples (green)")
            print("    Shows: Originals, Canonicals, Reconstructions, and Latent space")
        else:
            print("  - Reference structures analysis: Reference structures only")


if __name__ == "__main__":
    main()
