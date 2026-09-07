"""Consolidate the five completed full-Al analyses and verify matched coverage."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    full = root / "full_static_Al"
    temporal = json.loads((root / "comparison/metrics.json").read_text())
    metadata = json.loads((REPOSITORY / "datasets/cache/data_ae_Al_80_analysis_coords/metadata.json").read_text())
    expected_count = metadata["total_samples"]
    names = list(temporal["checkpoints"])
    results, assignments = {}, {}
    reference_coords = {}
    for name in names:
        metrics = json.loads((full / name / "analysis_metrics.json").read_text())
        info = metrics["clustering"]["cluster_fit_info_by_k"]["7"]
        assert info["fit_sample_count"] == expected_count, (name, info["fit_sample_count"])
        frames = metrics["real_md_qualitative"]["frames"]
        assert {f["source_name"]: f["num_samples"] for f in frames} == metadata["source_counts"], name
        labels = []
        for frame in frames:
            path = full / name / "snapshots" / frame["output_name"] / "md_space/local_structure_coords_clusters.npz"
            with np.load(path) as data:
                if name == "baseline":
                    reference_coords[frame["source_name"]] = data["coords"]
                else:
                    assert np.array_equal(data["coords"], reference_coords[frame["source_name"]]), (name, frame["source_name"])
                labels.append(data["clusters"])
        assignments[name] = np.concatenate(labels)
        assert len(assignments[name]) == expected_count, name
        with np.load(full / name / "analysis_inference_cache.npz") as data:
            latents = data["inv_latents"]
            assert len(latents) == expected_count, name
            covariance = np.cov(latents, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
            probabilities = eigenvalues / eigenvalues.sum()
            positive = probabilities[probabilities > 0]
        results[name] = {key: info[key] for key in (
            "fit_sample_count", "silhouette_cosine", "silhouette_euclidean", "davies_bouldin", "calinski_harabasz",
            "cluster_validation_sample_size", "cluster_counts", "pca_components")}
        results[name].update(projector_effective_rank=float(np.exp(-(positive * np.log(positive)).sum())),
                             projector_largest_pc_fraction=float(probabilities[-1]), frames=frames)
    ari = {a: {b: float(adjusted_rand_score(assignments[a], assignments[b])) for b in names} for a in names}
    report = dict(total_neighborhoods=expected_count, matched_coordinates_verified=True,
                  checkpoints=temporal["checkpoints"], metrics=results, cluster_agreement_ari=ari)
    (full / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Complete static-Al comparison", "",
             f"Inference and k=7 spherical-k-means fitting use all {expected_count:,} neighborhoods in the six-frame regular-grid dataset. Coordinate identity across all five model outputs was verified. Standard boundary exclusions apply; these are not one-per-atom samples. Silhouette and related validation scores use the pipeline's fixed 3,000-row diagnostic subset, while labels cover the full dataset. Cluster IDs are independent between models; adjusted Rand index is invariant to their numbering.", "",
             "| Checkpoint | Cosine silhouette ↑ | Euclidean silhouette ↑ | Davies–Bouldin ↓ | Projector effective rank |",
             "|---|---:|---:|---:|---:|"]
    for name in names:
        m = results[name]
        lines.append(f"| [{name}]({name}/real_md/README.md) | {m['silhouette_cosine']:.4f} | {m['silhouette_euclidean']:.4f} | {m['davies_bouldin']:.4f} | {m['projector_effective_rank']:.2f} |")
    lines.extend(["", "## Cluster agreement (adjusted Rand index)", "", "| Model | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)])
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{ari[name][other]:.4f}" for other in names) + " |")
    (full / "COMPARISON.md").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(3)
    for i, name in enumerate(names):
        for j, representation in enumerate(("encoder", "projector")):
            values = [temporal["results"][name]["temporal"][m][representation]["all"]["temporal_relative_mse"] for m in ("Al", "Mg", "Ta")]
            axes[j].bar(x + (i - 2) * .16, values, width=.16, label=name)
    for ax, representation in zip(axes[:2], ("Encoder", "Projector")):
        ax.set(xticks=x, xticklabels=["Al", "Mg", "Ta"], ylabel="Temporal MSE / latent variance", title=f"{representation}: temporal drift ↓")
    axes[0].legend(fontsize=8)
    axes[2].bar(np.arange(len(names)), [results[name]["silhouette_cosine"] for name in names], color=[f"C{i}" for i in range(len(names))])
    axes[2].set(xticks=np.arange(len(names)), xticklabels=names, ylabel="Cosine silhouette", title=f"Static Al: {expected_count:,} neighborhoods\n3,000-row silhouette diagnostic ↑")
    axes[2].tick_params(axis="x", labelrotation=40)
    figure.tight_layout()
    figure.savefig(root / "overview.png", dpi=180)
    plt.close(figure)
    print(full / "COMPARISON.md")


if __name__ == "__main__":
    main()
