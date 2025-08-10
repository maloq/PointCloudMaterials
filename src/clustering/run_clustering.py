from __future__ import annotations
import os
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
from typing import Literal, Optional


sys.path.append(os.getcwd())


from src.eval_pipeline.predict_functions import _get_latents_from_dataloader  
warnings.filterwarnings("ignore")


def _extract_latents(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Return only the latent codes for *dataloader* using *model*."""
    latents, *_ = _get_latents_from_dataloader(model, dataloader, device=device)
    return latents



def _extract_coords(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    """Gather (x, y, z) coordinates when the dataloader provides them."""
    coords: list[np.ndarray] = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            _, c = batch
            c = np.asarray(c)
            if c.ndim == 1:
                c = c[None, :]
        else:
            # Fallback: zeros when no coords are supplied
            bsz = batch[0].shape[0] if isinstance(batch, (tuple, list)) else len(batch)
            c = np.zeros((bsz, 3), dtype=np.float32)
        coords.append(c)
    coords = [coord.squeeze() for coord in coords]
    coords_concat = np.concatenate(coords, axis=0) if coords else np.empty((0, 3))
    return coords_concat


def find_optimal_clusters(
    data: np.ndarray,
    *,
    range_n_clusters: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> Tuple[int, List[float]]:
    """Select *k* via silhouette score."""
    scores: list[float] = []
    for k in range_n_clusters:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(data)
        score = silhouette_score(data, labels) if k > 1 else 0.0
        print(f"k = {k:>2}: silhouette = {score:.4f}")
        scores.append(score)
    best_k = max(range_n_clusters, key=lambda idx: scores[idx - range_n_clusters.start])
    print(f"✓ Chosen k = {best_k}\n")
    return best_k, scores



def _best_binary_assignment(cluster_labels: np.ndarray, true_labels: np.ndarray) -> Tuple[float, np.ndarray]:
    """Hungarian‑based best assignment for binary clustering."""
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)
    n_clusters = len(np.unique(cluster_labels))

    best_acc = 0.0
    best_lab = None

    for mapping in product([0, 1], repeat=n_clusters):
        if mapping.count(1) in (0, n_clusters):
            continue  # trivial
        mapped = np.array([mapping[c] for c in cluster_labels])
        cm = confusion_matrix(y_true, mapped)
        row, col = linear_sum_assignment(-cm)
        acc = cm[row, col].sum() / cm.sum()
        if acc > best_acc:
            best_acc, best_lab = acc, mapped

    # Flip so that label 0 ≈ class 0
    cm_final = confusion_matrix(y_true, best_lab)
    if cm_final[0, 1] + cm_final[1, 0] > cm_final[0, 0] + cm_final[1, 1]:
        best_lab = 1 - best_lab

    return float(best_acc), best_lab



def evaluate_clustering(cluster_labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """Return Accuracy / ARI / NMI together with mapped labels.

    **Note**: Any *noise* points marked as -1 (e.g. by HDBSCAN) are ignored
    when computing the metrics.
    """
    keep = cluster_labels != -1
    if keep.sum() < len(cluster_labels):  # some noise present
        cluster_labels = cluster_labels[keep]
        true_labels = true_labels[keep]

    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    acc, mapped = _best_binary_assignment(cluster_labels, true_labels)
    return {"accuracy": acc, "ari": ari, "nmi": nmi, "mapped_labels": mapped}



try:
    import umap  # noqa: F401; for optional UMAP support
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

ClusteringAlgo      = Literal["kmeans", "hdbscan"]
DimReductionAlgoOpt = Optional[Literal["pca", "umap", None]]


def predict_clusters(
    model: torch.nn.Module,
    train_latents: np.ndarray,
    eval_latents: np.ndarray,
    eval_coords: np.ndarray,
    # ---------------- Clustering -------------------------------------------
    algorithm: ClusteringAlgo = "kmeans",
    # --- K-Means options ---------------------------------------------------
    n_clusters: int | None = None,
    kmeans_random_state: int = 42,
    # --- HDBSCAN options ---------------------------------------------------
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int | None = None,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_metric: str = "euclidean",
    # ---------------- Dimensionality reduction ----------------------------
    dim_reduction: DimReductionAlgoOpt = None,       # "pca", "umap", or None
    n_components: int = 32,                          # output dimension
    pca_whiten: bool = False,                        # PCA option
    umap_n_neighbors: int = 15,                      # UMAP option
    umap_min_dist: float = 0.1,                      # UMAP option
    umap_random_state: int = 42,                     # UMAP option
    # ---------------- Subsampling -----------------------------------------
    subsample_size: int | None = None,
    subsample_random_state: int = 42,
    # ----------------------------------------------------------------------
    device: str = "cpu",
) -> np.ndarray:
    """
    Cluster *all* (train + eval) latents with an optional dimensionality-
    reduction step, then return labels only for the evaluation points.

    Returns
    -------
    np.ndarray, shape (N_eval, 4)
        Columns 0-2: coordinates (zeros if not provided)
        Column 3  : cluster label (-1 = noise for HDBSCAN)
    """
    model.eval().to(device)

    # ---------------------------------------------------------------------- #
    # 0)  Concatenate latents                                                #
    # ---------------------------------------------------------------------- #
    all_latents = np.concatenate([train_latents, eval_latents], axis=0)

    # ---------------------------------------------------------------------- #
    # 1)  Optional subsampling                                               #
    # ---------------------------------------------------------------------- #
    if subsample_size is not None:
        if subsample_size > len(all_latents):
            raise ValueError(
                f"Subsample size {subsample_size} exceeds total samples {len(all_latents)}."
            )
        rng          = np.random.default_rng(subsample_random_state)
        fit_idx      = rng.choice(len(all_latents), size=subsample_size, replace=False)
        fit_latents  = all_latents[fit_idx]
    else:
        fit_idx      = None  # sentinel
        fit_latents  = all_latents

    # ---------------------------------------------------------------------- #
    # 2)  Dimensionality reduction                                           #
    # ---------------------------------------------------------------------- #
    reducer = None
    if dim_reduction is not None:
        if dim_reduction == "pca":
            reducer = PCA(
                n_components=n_components,
                whiten=pca_whiten,
                random_state=kmeans_random_state,  # keep seeds consistent
            )
        elif dim_reduction == "umap":
            if not _HAS_UMAP:
                raise ImportError(
                    "UMAP is not installed. Run `pip install umap-learn` or "
                    "set dim_reduction=None / 'pca'."
                )
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=umap_random_state,
            )
        else:
            raise ValueError(
                f"dim_reduction must be None, 'pca', or 'umap' (got {dim_reduction})"
            )

        # Fit reducer on the *same* data we’ll fit the clusterer to
        reducer.fit(fit_latents)

        # Transform *all* data to keep shapes aligned
        all_latents_reduced  = reducer.transform(all_latents)
        eval_latents_reduced = all_latents_reduced[-len(eval_latents) :]
        fit_latents_reduced  = (
            all_latents_reduced[fit_idx] if fit_idx is not None else all_latents_reduced
        )
    else:
        # No reduction
        fit_latents_reduced  = fit_latents
        eval_latents_reduced = eval_latents

    # ---------------------------------------------------------------------- #
    # 3)  Fit clustering                                                     #
    # ---------------------------------------------------------------------- #
    if algorithm == "kmeans":
        if n_clusters is None:
            n_clusters, _ = find_optimal_clusters(
                fit_latents_reduced, random_state=kmeans_random_state
            )

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=kmeans_random_state,
            n_init=10,
        ).fit(fit_latents_reduced)

        labels_eval = clusterer.predict(eval_latents_reduced)

    elif algorithm == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            prediction_data=True,
            metric=hdbscan_metric,
            core_dist_n_jobs = -1
        ).fit(fit_latents_reduced)

        labels_eval, _ = hdbscan.approximate_predict(clusterer, eval_latents_reduced)

    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    # ---------------------------------------------------------------------- #
    # 4)  Assemble output                                                    #
    # ---------------------------------------------------------------------- #
    return np.column_stack([eval_coords, labels_eval])



def cluster_and_evaluate(
    latents: np.ndarray,
    labels: np.ndarray,
    *,
    algorithm: ClusteringAlgo = "kmeans",
    random_state: int = 42,
    subsample_size: int | None = None,
    subsample_random_state: int = 42,
    **algo_kwargs,
) -> dict:
    """Simple 50/50 split evaluation (train / val / full metrics)."""
    split = len(latents) // 2
    train_lat, val_lat = latents[:split], latents[split:]
    train_lab, val_lab = labels[:split], labels[split:]

    if subsample_size is not None:
        if subsample_size > len(train_lat):
            raise ValueError(
                f"Subsample size {subsample_size} is larger than the number of "
                f"training samples {len(train_lat)}."
            )
        print(f"Subsampling train latents to {subsample_size} …")
        rng = np.random.default_rng(subsample_random_state)
        indices = rng.choice(len(train_lat), size=subsample_size, replace=False)
        train_lat = train_lat[indices]
        train_lab = train_lab[indices]


    if algorithm == "kmeans":
        print("→ Selecting k on train split")
        k, _ = find_optimal_clusters(train_lat, random_state=random_state)
        clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(train_lat)

        train_pred = clusterer.labels_
        val_pred = clusterer.predict(val_lat)
        full_pred = clusterer.predict(latents)

    elif algorithm == "hdbscan":
        if hdbscan is None:
            raise ImportError(
                "HDBSCAN requested but the 'hdbscan' package is not installed. "
                "Install with `pip install hdbscan`."
            )

        clusterer = hdbscan.HDBSCAN(prediction_data=True, **algo_kwargs).fit(train_lat)
        train_pred = clusterer.labels_
        val_pred, _ = hdbscan.approximate_predict(clusterer, val_lat)
        full_pred, _ = hdbscan.approximate_predict(clusterer, latents)

    else:  # pragma: no cover
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print("\n→ Evaluating")
    train_res = evaluate_clustering(train_pred, train_lab)
    val_res = evaluate_clustering(val_pred, val_lab)
    full_res = evaluate_clustering(full_pred, labels)

    for name, res in [("train", train_res), ("val", val_res), ("full", full_res)]:
        print(f"{name:>5}  acc={res['accuracy']:.3f}  ari={res['ari']:.3f}  nmi={res['nmi']:.3f}")

    return {
        "clusterer": clusterer,
        "algorithm": algorithm,
        "train": train_res,
        "val": val_res,
        "full": full_res,
    }



if __name__ == "__main__":

    data = np.load("output/latent_data_spd.npz")
    latents, labels = data["latents"], data["labels"]

    cluster_and_evaluate(
        latents,
        labels,
        algorithm="kmeans",
        n_clusters=4,
    )
