import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.manifold import TSNE


def compute_tsne(
    latents: np.ndarray,
    *,
    random_state: int = 42,
    perplexity: int | None = None,
) -> np.ndarray:
    """Project high-dimensional latents to 2D using t-SNE.

    Picks a safe perplexity based on the number of samples when not provided.
    """
    num_samples = int(latents.shape[0])
    if num_samples < 2:
        raise ValueError("Cannot compute t-SNE with fewer than 2 samples.")

    if perplexity is None:
        # Ensure 2 <= perplexity < num_samples
        auto = max(2, min(30, (num_samples - 1) // 3))
        perplexity = min(auto, num_samples - 1)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        perplexity=perplexity,
    )
    return tsne.fit_transform(latents)


def save_tsne_plot(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    *,
    out_file: str,
    title: str,
) -> None:
    """Save a t-SNE scatter plot, colored by cluster labels, to *out_file*.

    Cluster label ``-1`` (e.g., HDBSCAN noise) is colored light gray.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Build a robust color mapping
    unique_labels = np.unique(labels)
    non_noise = [int(l) for l in unique_labels if int(l) != -1]
    colormap = plt.cm.get_cmap("tab20", max(1, len(non_noise)))
    label_to_color: Dict[int, Any] = {}
    color_index = 0
    for lbl in sorted(unique_labels, key=lambda x: (x == -1, x)):
        lbl_int = int(lbl)
        if lbl_int == -1:
            label_to_color[lbl_int] = "lightgray"
        else:
            label_to_color[lbl_int] = colormap(color_index)
            color_index += 1

    plt.figure(figsize=(7, 6), dpi=150)
    for lbl in unique_labels:
        lbl_int = int(lbl)
        mask = labels == lbl
        plt.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            s=6,
            alpha=0.8,
            c=[label_to_color[lbl_int]],
            label=str(lbl_int),
            linewidths=0,
        )

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    # Only show legend if number of clusters is manageable
    if len(unique_labels) <= 20:
        plt.legend(title="cluster", markerscale=2, fontsize=8, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()