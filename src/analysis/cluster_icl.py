"""ICL curve computation and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .cluster_colors import _l2_normalize_rows


def _prepare_icl_features(
    latents: np.ndarray,
    *,
    l2_normalize: bool,
    standardize: bool,
    pca_variance: float | None,
    pca_max_components: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(latents, dtype=np.float32)
    if x.ndim != 2:
        x = np.reshape(x, (x.shape[0], -1))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 samples to compute ICL curve, got {x.shape[0]}."
        )

    info: dict[str, Any] = {
        "input_dim": int(x.shape[1]),
        "l2_normalize": bool(l2_normalize),
        "standardize": bool(standardize),
    }

    if l2_normalize:
        x = _l2_normalize_rows(x)
    if standardize and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    if (
        pca_variance is not None
        and float(pca_variance) > 0.0
        and x.shape[1] > 2
        and x.shape[0] > 3
    ):
        n_max = min(int(pca_max_components), x.shape[1], x.shape[0] - 1)
        if n_max >= 2:
            pca = PCA(n_components=n_max, random_state=random_state)
            proj = pca.fit_transform(x)
            if float(pca_variance) >= 1.0:
                keep = n_max
            else:
                csum = np.cumsum(pca.explained_variance_ratio_)
                keep = int(np.searchsorted(csum, float(pca_variance)) + 1)
                keep = max(2, min(keep, n_max))
            x = proj[:, :keep]
            info["pca_components"] = int(keep)
            info["pca_explained_variance"] = float(
                np.sum(pca.explained_variance_ratio_[:keep])
            )
        else:
            info["pca_components"] = int(x.shape[1])
            info["pca_explained_variance"] = 1.0
    else:
        info["pca_components"] = int(x.shape[1])
        info["pca_explained_variance"] = 1.0

    info["output_dim"] = int(x.shape[1])
    return x.astype(np.float32, copy=False), info


def _compute_icl_curve(
    features: np.ndarray,
    k_values: list[int],
    *,
    covariance_type: str = "diag",
    random_state: int = 42,
) -> dict[int, dict[str, float]]:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {x.shape}.")
    if x.shape[0] < 3:
        raise ValueError(f"Need at least 3 samples for ICL curve, got {x.shape[0]}.")

    # Keep covariance_type for backward-compatible function signature.
    _ = covariance_type

    curve: dict[int, dict[str, float]] = {}
    for k in k_values:
        k_eff = int(k)
        if k_eff < 2:
            raise ValueError(f"Invalid k value {k_eff}; expected >= 2.")
        if k_eff >= x.shape[0]:
            raise ValueError(
                f"Invalid k value {k_eff}: must be < number of samples ({x.shape[0]})."
            )
        model = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
        try:
            labels = model.fit_predict(x)
        except Exception as exc:
            raise RuntimeError(
                "Failed to fit KMeans for ICL curve "
                f"at k={k_eff}."
            ) from exc

        # KMeans surrogate for model-selection curve:
        # lower inertia is better; entropy term penalizes highly fragmented assignments.
        inertia = float(model.inertia_)
        counts = np.bincount(labels, minlength=k_eff).astype(np.float64)
        probs = counts / np.clip(counts.sum(), 1.0, None)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        icl = float(inertia + entropy)
        curve[k_eff] = {
            "bic": inertia,
            "entropy": entropy,
            "icl": icl,
        }
    return curve


def _save_icl_curve_figure(
    icl_curve: dict[int, dict[str, float]],
    *,
    selected_k: int,
    out_file: Path,
    y_label: str = "ICL",
) -> dict[str, Any]:
    if not icl_curve:
        raise ValueError("ICL curve is empty; nothing to plot.")
    k_values = sorted(int(k) for k in icl_curve.keys())
    scores = [float(icl_curve[k]["icl"]) for k in k_values]
    if selected_k not in icl_curve:
        raise ValueError(
            f"selected_k={selected_k} is missing from ICL curve keys {k_values}."
        )

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=220)
    ax.plot(
        k_values,
        scores,
        color="black",
        linestyle="--",
        linewidth=1.6,
        marker="o",
        markersize=6.5,
        markerfacecolor="black",
        markeredgewidth=0.0,
    )
    ax.axvline(
        x=float(selected_k),
        color="black",
        linewidth=1.2,
        linestyle=(0, (5, 5)),
    )
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(y_label)
    ax.set_xticks(k_values)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    argmin_idx = int(np.argmin(np.asarray(scores, dtype=np.float64)))
    return {
        "out_file": str(out_file),
        "k_values": [int(v) for v in k_values],
        "icl_values": [float(v) for v in scores],
        "icl_best_k": int(k_values[argmin_idx]),
        "icl_value_selected_k": float(icl_curve[int(selected_k)]["icl"]),
    }
