from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.training_methods.spd.eval_spd import load_spd_model
from src.data_utils.data_load import pc_normalize, SyntheticPointCloudDataset


def extract_actual_dataset_samples(
    dataset: SyntheticPointCloudDataset,
    num_samples_per_phase: int = 3,
) -> Dict[str, List[np.ndarray]]:
    """Extract actual samples from the synthetic dataset, organized by phase."""
    phase_samples: Dict[str, List[np.ndarray]] = {}
    idx_to_phase = {idx: name for name, idx in dataset._phase_to_idx.items()}
    phase_counts: Dict[str, int] = {name: 0 for name in idx_to_phase.values()}

    for i in range(len(dataset)):
        if all(count >= num_samples_per_phase for count in phase_counts.values()):
            break

        phase_idx = dataset._phase_labels[i]
        phase_name = idx_to_phase.get(phase_idx, f"unknown_{phase_idx}")

        if phase_counts[phase_name] < num_samples_per_phase:
            pc_tensor = dataset.samples[i]
            phase_samples.setdefault(phase_name, []).append(pc_tensor.numpy())
            phase_counts[phase_name] += 1

    return phase_samples


def visualize_reference_structures(
    checkpoint_path: str,
    reference_structures_path: str,
    output_path: Path,
    cuda_device: int = 0,
    dataset: SyntheticPointCloudDataset | None = None,
    compare_with_dataset: bool = True,
) -> None:
    """
    Create visualization of reference structures, their reconstructions, and latent space.
    Optionally compare with actual dataset samples.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    model, cfg, device = load_spd_model(checkpoint_path, cuda_device=cuda_device)
    model.eval()

    radius = float(cfg.data.radius) if hasattr(cfg.data, "radius") else None
    print(f"Model training radius: {radius}")
    if radius is None:
        print("WARNING: Could not determine radius from config. Using max_norm normalization.")

    print(f"Loading reference point clouds from {reference_structures_path}")
    ref_point_clouds = np.load(reference_structures_path, allow_pickle=True).item()

    if not isinstance(ref_point_clouds, dict) or len(ref_point_clouds) == 0:
        print("No reference point clouds found in the provided file")
        return

    structure_names: List[str] = []
    point_clouds: List[np.ndarray] = []
    source_types: List[str] = []

    print("\n=== Processing Reference Structures ===")
    print("APPLYING FIX: Normalizing by radius (not max_norm) to match dataset preprocessing")
    for name in sorted(ref_point_clouds.keys()):
        if name.startswith("intermediate_"):
            continue

        pc = np.asarray(ref_point_clouds[name], dtype=float)
        if pc.ndim != 2 or pc.shape[1] != 3:
            print(f"    Warning: Invalid point cloud shape for {name}, skipping")
            continue
        if len(pc) == 0:
            print(f"    Warning: Empty point cloud for {name}, skipping")
            continue

        mean_before = pc.mean(axis=0)
        max_norm_before = np.max(np.linalg.norm(pc, axis=1))
        print(f"  {name} [BEFORE]: mean={mean_before}, max_norm={max_norm_before:.4f}")

        pc_centered = pc - pc.mean(axis=0, keepdims=True)

        if radius is not None:
            pc_normalized = pc_normalize(pc_centered, radius)
        else:
            pc_normalized = pc_normalize(pc_centered, None)

        mean_after = pc_normalized.mean(axis=0)
        max_norm_after = np.max(np.linalg.norm(pc_normalized, axis=1))
        scale_factor = max_norm_before / max_norm_after if max_norm_after > 0 else 1.0
        print(f"  {name} [AFTER]:  mean={mean_after}, max_norm={max_norm_after:.4f}, scale_factor={scale_factor:.4f}")

        point_clouds.append(pc_normalized)
        structure_names.append(f"ref_{name}")
        source_types.append("reference")

    if len(point_clouds) == 0:
        print("No valid structures to visualize")
        return

    print(f"\nLoaded {len(structure_names)} reference structures")

    if compare_with_dataset and dataset is not None:
        print("\n=== Extracting Actual Dataset Samples ===")
        dataset_samples = extract_actual_dataset_samples(dataset, num_samples_per_phase=2)

        for phase_name, samples_list in sorted(dataset_samples.items()):
            for idx, sample_pc in enumerate(samples_list):
                mean_ds = sample_pc.mean(axis=0)
                max_norm_ds = np.max(np.linalg.norm(sample_pc, axis=1))
                print(f"  {phase_name}_sample{idx}: mean={mean_ds}, max_norm={max_norm_ds:.4f}")

                point_clouds.append(sample_pc)
                structure_names.append(f"ds_{phase_name}_{idx}")
                source_types.append("dataset")

        print(f"Added {sum(len(v) for v in dataset_samples.values())} dataset samples")

    print(f"\nTotal structures to process: {len(structure_names)}")

    latents_list: List[np.ndarray] = []
    reconstructions_list: List[np.ndarray] = []
    canonicals_list: List[np.ndarray] = []
    originals_list: List[np.ndarray] = []
    recon_names: List[str] = []
    recon_source_types: List[str] = []

    print("\n=== Running Model Inference ===")
    with torch.no_grad():
        for name, pc, source_type in zip(structure_names, point_clouds, source_types):
            print(f"  Processing {name} ({source_type})...")

            mean_pre = pc.mean(axis=0)
            max_norm_pre = np.max(np.linalg.norm(pc, axis=1))
            print(f"    Before model: mean={mean_pre}, max_norm={max_norm_pre:.4f}")

            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

            try:
                inv_z, recon, cano, rot = model(pc_tensor)

                recon_np = recon.cpu().numpy()[0]
                cano_np = cano.cpu().numpy()[0]

                mean_recon = recon_np.mean(axis=0)
                max_norm_recon = np.max(np.linalg.norm(recon_np, axis=1))
                print(f"    Reconstruction: mean={mean_recon}, max_norm={max_norm_recon:.4f}")

                latents_list.append(inv_z.cpu().numpy()[0])
                reconstructions_list.append(recon_np)
                canonicals_list.append(cano_np)
                originals_list.append(pc)
                recon_names.append(name)
                recon_source_types.append(source_type)
            except Exception as e:
                print(f"    Warning: Model inference failed for {name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"\nSuccessfully processed {len(recon_names)} structures")

    n_structures = len(recon_names)
    if n_structures == 0:
        print("No structures to visualize")
        return

    fig = plt.figure(figsize=(4 * n_structures, 16))

    border_width = 72.0 / fig.dpi
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(border_width)

    all_points = originals_list + reconstructions_list + canonicals_list
    if all_points:
        max_extent = max(np.max(np.abs(pc)) for pc in all_points)
        viz_limit = max(0.6, float(max_extent) * 1.1)
    else:
        viz_limit = 0.6

    color_map = {"reference": "blue", "dataset": "green"}

    for col, (name, pc, source_type) in enumerate(zip(recon_names, originals_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, col + 1, projection="3d")

        ax.scatter(
            pc[:, 0], pc[:, 1], pc[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c=color_map.get(source_type, "gray"),
        )
        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Original)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    for col, (name, cano, source_type) in enumerate(zip(recon_names, canonicals_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, n_structures + col + 1, projection="3d")

        ax.scatter(
            cano[:, 0], cano[:, 1], cano[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="purple",
        )

        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Canonical)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    for col, (name, recon, source_type) in enumerate(zip(recon_names, reconstructions_list, recon_source_types)):
        ax = fig.add_subplot(4, n_structures, 2 * n_structures + col + 1, projection="3d")

        ax.scatter(
            recon[:, 0], recon[:, 1], recon[:, 2],
            s=25, alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
            c="orange",
        )

        display_name = name.replace("ref_", "").replace("ds_", "")
        source_label = "REF" if source_type == "reference" else "DS"
        ax.set_title(f"[{source_label}] {display_name}\n(Reconstruction)", fontsize=8)
        ax.set_xlim(-viz_limit, viz_limit)
        ax.set_ylim(-viz_limit, viz_limit)
        ax.set_zlim(-viz_limit, viz_limit)

    if len(latents_list) > 1:
        latents_array = np.array(latents_list)

        pca = PCA(n_components=min(3, latents_array.shape[1]))
        latents_3d = pca.fit_transform(latents_array)

        if latents_3d.shape[1] < 3:
            latents_3d = np.pad(
                latents_3d,
                ((0, 0), (0, 3 - latents_3d.shape[1])),
                mode="constant",
            )

        ax = fig.add_subplot(4, n_structures, 3 * n_structures + n_structures // 2 + 1, projection="3d")

        for name, latent_pt, source_type in zip(recon_names[:len(latents_3d)], latents_3d, recon_source_types):
            color = color_map.get(source_type, "gray")
            marker = "o" if source_type == "reference" else "^"
            display_name = name.replace("ref_", "").replace("ds_", "")

            ax.scatter(
                latent_pt[0], latent_pt[1], latent_pt[2],
                s=200, alpha=0.9,
                edgecolors="black",
                linewidths=2,
                c=color,
                marker=marker,
                label=display_name,
            )

        ax.set_title("Latent Space (PCA)\nBlue=Reference, Green=Dataset", fontsize=10)
        ax.legend(loc="upper left", fontsize=6, ncol=2)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})" if len(pca.explained_variance_ratio_) > 1 else "PC2")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})" if len(pca.explained_variance_ratio_) > 2 else "PC3")

    title = "Reference Structure Analysis"
    if compare_with_dataset and dataset is not None:
        title += " (with Dataset Samples)"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nVisualization saved to {output_path}")
    print("Legend: REF=Reference structure, DS=Dataset sample")
    print("Colors: Blue=Reference, Green=Dataset, Purple=Canonical, Orange=Reconstruction")


__all__ = ["visualize_reference_structures"]
