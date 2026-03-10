import os
import inspect
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.manifold import TSNE


def _log_saved_figure(path: Path | str) -> None:
    print(f"[analysis][savefig] {Path(path).resolve()}")


def compute_tsne(
    latents: np.ndarray,
    *,
    random_state: int = 42,
    perplexity: int | None = None,
    n_iter: int = 1000,
) -> np.ndarray:
    """Project high-dimensional latents to 2D using t-SNE.

    Picks a safe perplexity based on the number of samples when not provided.
    """
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"latents must be a 2D array of shape (N, D), got {arr.shape}."
        )
    num_samples = int(arr.shape[0])
    if num_samples < 2:
        raise ValueError("Cannot compute t-SNE with fewer than 2 samples.")

    if perplexity is None:
        # Ensure 2 <= perplexity < num_samples
        auto = max(2, min(30, (num_samples - 1) // 3))
        perplexity = min(auto, num_samples - 1)
    else:
        perplexity = int(perplexity)
    if perplexity < 1 or perplexity >= num_samples:
        raise ValueError(
            "Invalid t-SNE perplexity. Expected 1 <= perplexity < number of samples, "
            f"got perplexity={perplexity}, num_samples={num_samples}."
        )

    # Build kwargs to avoid passing None for sklearn parameters
    tsne_kwargs: Dict[str, Any] = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
        "perplexity": perplexity,
    }
    if n_iter is not None:
        n_iter_int = int(n_iter)
        if n_iter_int <= 0:
            raise ValueError(f"n_iter must be > 0, got {n_iter_int}.")
        tsne_params = inspect.signature(TSNE.__init__).parameters
        if "max_iter" in tsne_params:
            tsne_kwargs["max_iter"] = n_iter_int
        elif "n_iter" in tsne_params:
            tsne_kwargs["n_iter"] = n_iter_int
        else:
            raise RuntimeError(
                "Unsupported sklearn TSNE signature: expected parameter "
                "'max_iter' or 'n_iter', but found neither. "
                f"Available parameters: {sorted(tsne_params.keys())}."
            )

    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(arr)


def save_tsne_plot(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    *,
    out_file: str,
    title: str | None,
    show: bool = False,
    legend_title: str = "cluster",
    class_names: Dict[Any, str] | None = None,
    cluster_color_map: Dict[int, str] | None = None,
    paper_style: bool = False,
    label_prefix: str | None = None,
) -> None:
    """Save a t-SNE scatter plot, colored by labels, to *out_file*.

    Label ``-1`` (e.g., HDBSCAN noise) is colored light gray.
    If *cluster_color_map* is provided, use those colors instead of tab10.
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
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10)).astype(np.float32)
    label_to_color: Dict[Any, Any] = {}
    color_index = 0
    # Sort with noise last for visibility
    for lbl in sorted(unique_labels, key=lambda x: (is_noise_label(x), str(x))):
        if is_noise_label(lbl):
            label_to_color[lbl] = "lightgray"
        elif cluster_color_map is not None and int(lbl) in cluster_color_map:
            label_to_color[lbl] = cluster_color_map[int(lbl)]
        else:
            label_to_color[lbl] = tab10_colors[color_index % len(tab10_colors)]
            color_index += 1

    fig_size = (6.8, 5.8) if paper_style else (8, 7)
    marker_size = 14 if paper_style else 18
    marker_alpha = 0.88 if paper_style else 0.85
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)
    for lbl in unique_labels:
        mask = labels == lbl

        # Determine label text
        if class_names is not None and lbl in class_names:
             label_text = class_names[lbl]
        elif class_names is not None and int(lbl) in class_names: # Handle potential type mismatch (int/float/np.int)
             label_text = class_names[int(lbl)]
        else:
             label_text = str(lbl)
        if label_prefix is not None and class_names is None and not is_noise_label(lbl):
            label_text = f"{label_prefix}{label_text}"

        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            s=marker_size,
            alpha=marker_alpha,
            c=[label_to_color[lbl]],
            label=label_text,
            linewidths=0,
            rasterized=not paper_style,
        )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Only show legend if number of labels is manageable
    if len(unique_labels) <= 20:
        if paper_style:
            ax.legend(
                title=legend_title,
                markerscale=2.2,
                fontsize=9,
                title_fontsize=10,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=False,
                ncol=1,
            )
        else:
            ax.legend(
                title=legend_title,
                markerscale=3,
                fontsize=11,
                title_fontsize=12,
                loc="best",
                frameon=True,
                fancybox=True,
                framealpha=0.85,
                edgecolor="0.8",
            )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    _log_saved_figure(out_file)
    if show:
        plt.show()
    else:
        plt.close(fig)
