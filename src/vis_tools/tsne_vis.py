import os
import inspect
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
from sklearn.manifold import TSNE


def _log_saved_figure(path: Path | str) -> None:
    print(f"[analysis][savefig] {Path(path).resolve()}")


def _normalize_label_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _is_noise_label(value: Any) -> bool:
    return value in {-1, -1.0, "-1"}


def _resolve_label_text(
    label: Any,
    *,
    class_names: Dict[Any, str] | None,
    label_prefix: str | None,
) -> str:
    if class_names is not None and label in class_names:
        return str(class_names[label])
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        label_int = None
    if class_names is not None and label_int is not None and label_int in class_names:
        return str(class_names[label_int])

    label_text = str(label)
    if label_prefix is not None and class_names is None and not _is_noise_label(label):
        return f"{label_prefix}{label_text}"
    return label_text


def _build_label_color_map(
    labels: np.ndarray,
    *,
    cluster_color_map: Dict[int, str] | None,
) -> tuple[list[Any], Dict[Any, Any]]:
    unique_labels = [_normalize_label_scalar(v) for v in np.unique(labels)]
    non_noise_labels = [v for v in unique_labels if not _is_noise_label(v)]
    tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10)).astype(np.float32)
    label_to_color: Dict[Any, Any] = {}
    color_index = 0
    for label in sorted(unique_labels, key=lambda value: (_is_noise_label(value), str(value))):
        if _is_noise_label(label):
            label_to_color[label] = "lightgray"
        elif cluster_color_map is not None and int(label) in cluster_color_map:
            label_to_color[label] = cluster_color_map[int(label)]
        else:
            label_to_color[label] = tab10_colors[color_index % len(tab10_colors)]
            color_index += 1
    if len(label_to_color) != len(unique_labels):
        raise RuntimeError(
            "Internal error while building label color map: "
            f"len(unique_labels)={len(unique_labels)}, len(label_to_color)={len(label_to_color)}, "
            f"non_noise_labels={non_noise_labels}."
        )
    return unique_labels, label_to_color


def _scatter_labeled_embedding(
    ax: Any,
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str | None,
    legend_title: str,
    class_names: Dict[Any, str] | None,
    cluster_color_map: Dict[int, str] | None,
    paper_style: bool,
    label_prefix: str | None,
    show_legend: bool,
) -> None:
    coords_arr = np.asarray(coords, dtype=np.float32)
    labels_arr = np.asarray(labels)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise ValueError(
            "Expected 2D embedding coordinates with shape (N, 2), "
            f"got shape={tuple(coords_arr.shape)}."
        )
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError(
            "Embedding coordinates and labels must have the same number of rows, "
            f"got coords.shape[0]={coords_arr.shape[0]}, labels.shape[0]={labels_arr.shape[0]}."
        )

    unique_labels, label_to_color = _build_label_color_map(
        labels_arr,
        cluster_color_map=cluster_color_map,
    )
    marker_size = 14 if paper_style else 18
    marker_alpha = 0.88 if paper_style else 0.85

    for label in unique_labels:
        mask = labels_arr == label
        ax.scatter(
            coords_arr[mask, 0],
            coords_arr[mask, 1],
            s=marker_size,
            alpha=marker_alpha,
            c=[label_to_color[label]],
            label=_resolve_label_text(
                label,
                class_names=class_names,
                label_prefix=label_prefix,
            ),
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

    if show_legend and len(unique_labels) <= 20:
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


def _resolve_umap_backend(requested_backend: str) -> dict[str, Any]:
    backend_norm = str(requested_backend).strip().lower() or "auto"
    if backend_norm not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "UMAP backend must be one of ['auto', 'cpu', 'gpu'], "
            f"got {requested_backend!r}."
        )

    gpu_available = False
    gpu_probe_error: str | None = None
    try:
        import torch
    except ImportError as exc:
        gpu_probe_error = f"PyTorch import failed while probing CUDA availability: {exc}"
    else:
        gpu_available = bool(torch.cuda.is_available())

    if backend_norm == "cpu":
        return {
            "backend": "umap-learn",
            "device": "cpu",
            "requested_backend": backend_norm,
            "gpu_available": bool(gpu_available),
            "reason": "CPU backend forced by configuration.",
        }

    if gpu_available:
        try:
            from cuml.manifold import UMAP as CuMLUMAP
        except ImportError as exc:
            if backend_norm == "gpu":
                raise ImportError(
                    "UMAP backend 'gpu' requires RAPIDS cuML "
                    "(cuml.manifold.UMAP), but it is not installed."
                ) from exc
            return {
                "backend": "umap-learn",
                "device": "cpu",
                "requested_backend": backend_norm,
                "gpu_available": True,
                "reason": (
                    "CUDA is available, but RAPIDS cuML is not installed. "
                    "Falling back to CPU umap-learn."
                ),
            }
        return {
            "backend": "cuml",
            "device": "gpu",
            "requested_backend": backend_norm,
            "gpu_available": True,
            "reason": None,
            "umap_class": CuMLUMAP,
        }

    if backend_norm == "gpu":
        raise RuntimeError(
            "UMAP backend 'gpu' requires a CUDA-capable runtime, "
            f"but torch.cuda.is_available() returned False. "
            f"Probe details: {gpu_probe_error or 'ok'}."
        )

    return {
        "backend": "umap-learn",
        "device": "cpu",
        "requested_backend": backend_norm,
        "gpu_available": False,
        "reason": (
            "CUDA is not available for UMAP acceleration."
            if gpu_probe_error is None
            else gpu_probe_error
        ),
    }


def _build_umap_reducer(
    backend_info: dict[str, Any],
    *,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> Any:
    reducer_kwargs = {
        "n_components": 2,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": str(metric),
    }
    if str(backend_info["backend"]) == "cuml":
        reducer_cls = backend_info.get("umap_class")
        if reducer_cls is None:
            raise RuntimeError("Missing cuML UMAP class in backend_info.")
        try:
            reducer_signature = inspect.signature(reducer_cls.__init__)
        except (TypeError, ValueError):
            reducer_signature = None
        if reducer_signature is not None and "output_type" in reducer_signature.parameters:
            reducer_kwargs["output_type"] = "numpy"
        if reducer_signature is not None and "random_state" in reducer_signature.parameters:
            reducer_kwargs["random_state"] = int(random_state)
        return reducer_cls(**reducer_kwargs)

    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP projection requested but umap-learn is not installed.") from exc

    try:
        reducer_signature = inspect.signature(umap.UMAP.__init__)
    except (TypeError, ValueError):
        reducer_signature = None
    if reducer_signature is not None:
        if "random_state" in reducer_signature.parameters:
            reducer_kwargs["random_state"] = int(random_state)
        if "transform_seed" in reducer_signature.parameters:
            reducer_kwargs["transform_seed"] = int(random_state)
    return umap.UMAP(**reducer_kwargs)


def compute_umap(
    latents: np.ndarray,
    *,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.15,
    metric: str = "euclidean",
    backend: str = "auto",
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"latents must be a 2D array of shape (N, D), got {arr.shape}."
        )
    num_samples = int(arr.shape[0])
    if num_samples < 2:
        raise ValueError("Cannot compute UMAP with fewer than 2 samples.")

    n_neighbors_int = int(n_neighbors)
    if n_neighbors_int < 2:
        raise ValueError(f"UMAP n_neighbors must be >= 2, got {n_neighbors_int}.")
    if not np.isfinite(float(min_dist)) or float(min_dist) < 0.0:
        raise ValueError(f"UMAP min_dist must be finite and >= 0, got {min_dist}.")
    n_neighbors_used = min(n_neighbors_int, num_samples - 1)
    if n_neighbors_used < 2:
        raise ValueError(
            "UMAP requires at least 2 neighbors after sample-count adjustment, "
            f"got n_neighbors_used={n_neighbors_used}, num_samples={num_samples}."
        )

    backend_info = _resolve_umap_backend(backend)
    reducer = _build_umap_reducer(
        backend_info,
        random_state=int(random_state),
        n_neighbors=int(n_neighbors_used),
        min_dist=float(min_dist),
        metric=str(metric),
    )
    embedding = np.asarray(reducer.fit_transform(arr), dtype=np.float32)
    if embedding.shape != (num_samples, 2):
        raise RuntimeError(
            "UMAP produced an unexpected embedding shape. "
            f"Expected {(num_samples, 2)}, got {tuple(embedding.shape)}."
        )
    info = {
        "method": "umap",
        "backend": str(backend_info["backend"]),
        "device": str(backend_info["device"]),
        "requested_backend": str(backend_info["requested_backend"]),
        "backend_reason": backend_info.get("reason"),
        "n_neighbors_requested": int(n_neighbors_int),
        "n_neighbors_used": int(n_neighbors_used),
        "min_dist": float(min_dist),
        "metric": str(metric),
        "random_state": int(random_state),
    }
    if return_info:
        return embedding, info
    return embedding


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

    fig_size = (6.8, 5.8) if paper_style else (8, 7)
    fig, ax = plt.subplots(figsize=fig_size, dpi=200)
    _scatter_labeled_embedding(
        ax,
        tsne_coords,
        labels,
        title=title,
        legend_title=legend_title,
        class_names=class_names,
        cluster_color_map=cluster_color_map,
        paper_style=paper_style,
        label_prefix=label_prefix,
        show_legend=True,
    )
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    _log_saved_figure(out_file)
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_embedding_comparison_plot(
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    *,
    out_file: str,
    left_title: str | None,
    right_title: str | None,
    overall_title: str | None = None,
    legend_title: str = "cluster",
    class_names: Dict[Any, str] | None = None,
    cluster_color_map: Dict[int, str] | None = None,
    label_prefix: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    left_arr = np.asarray(left_coords, dtype=np.float32)
    right_arr = np.asarray(right_coords, dtype=np.float32)
    left_labels_arr = np.asarray(left_labels)
    right_labels_arr = np.asarray(right_labels)
    if left_arr.shape != right_arr.shape:
        raise ValueError(
            "Left/right embedding coordinates must have the same shape, "
            f"got left={tuple(left_arr.shape)}, right={tuple(right_arr.shape)}."
        )
    if left_arr.ndim != 2 or left_arr.shape[1] != 2:
        raise ValueError(
            "Embedding comparison expects 2D coordinates with shape (N, 2), "
            f"got {tuple(left_arr.shape)}."
        )
    if left_labels_arr.shape[0] != left_arr.shape[0]:
        raise ValueError(
            "Left embedding comparison labels must match coordinate rows, "
            f"got left_labels.shape[0]={left_labels_arr.shape[0]}, coords.shape[0]={left_arr.shape[0]}."
        )
    if right_labels_arr.shape[0] != right_arr.shape[0]:
        raise ValueError(
            "Right embedding comparison labels must match coordinate rows, "
            f"got right_labels.shape[0]={right_labels_arr.shape[0]}, coords.shape[0]={right_arr.shape[0]}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), dpi=200)
    _scatter_labeled_embedding(
        axes[0],
        left_arr,
        left_labels_arr,
        title=left_title,
        legend_title=legend_title,
        class_names=class_names,
        cluster_color_map=cluster_color_map,
        paper_style=False,
        label_prefix=label_prefix,
        show_legend=False,
    )
    _scatter_labeled_embedding(
        axes[1],
        right_arr,
        right_labels_arr,
        title=right_title,
        legend_title=legend_title,
        class_names=class_names,
        cluster_color_map=cluster_color_map,
        paper_style=False,
        label_prefix=label_prefix,
        show_legend=True,
    )
    if overall_title:
        fig.suptitle(overall_title, fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    _log_saved_figure(out_file)
    plt.close(fig)
