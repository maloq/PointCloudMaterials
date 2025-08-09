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
    n_iter: int | None = None,
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

    # Build kwargs to avoid passing None for sklearn parameters
    tsne_kwargs: Dict[str, Any] = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
        "perplexity": perplexity,
    }
    if n_iter is not None:
        tsne_kwargs["n_iter"] = int(n_iter)

    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(latents)


def save_tsne_plot(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    *,
    out_file: str,
    title: str,
    show: bool = False,
) -> None:
    """Save a t-SNE scatter plot, colored by cluster labels, to *out_file*.

    Cluster label ``-1`` (e.g., HDBSCAN noise) is colored light gray.
    """
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Build a robust color mapping for arbitrary (possibly non-integer) labels
    unique_labels = np.unique(labels)
    # Normalize numpy scalar types to Python scalars for consistent comparisons
    def to_py(v: Any) -> Any:
        return v.item() if isinstance(v, np.generic) else v

    unique_labels = [to_py(v) for v in unique_labels]

    def is_noise_label(v: Any) -> bool:
        return v in {-1, -1.0, "-1"}

    non_noise_labels = [v for v in unique_labels if not is_noise_label(v)]
    colormap = plt.cm.get_cmap("tab20", max(1, len(non_noise_labels)))
    label_to_color: Dict[Any, Any] = {}
    color_index = 0
    # Sort with noise last for visibility
    for lbl in sorted(unique_labels, key=lambda x: (is_noise_label(x), str(x))):
        if is_noise_label(lbl):
            label_to_color[lbl] = "lightgray"
        else:
            label_to_color[lbl] = colormap(color_index)
            color_index += 1

    plt.figure(figsize=(7, 6), dpi=150)
    for lbl in unique_labels:
        mask = labels == lbl
        plt.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            s=6,
            alpha=0.8,
            c=[label_to_color[lbl]],
            label=str(lbl),
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
    if show:
        plt.show()
    else:
        plt.close()