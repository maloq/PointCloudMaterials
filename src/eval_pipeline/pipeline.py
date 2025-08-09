from __future__ import annotations

"""High level evaluation pipeline."""

from typing import Any, Dict, List
import os
import sys,os
sys.path.append(os.getcwd())
from datetime import datetime
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.eval_pipeline.predictors import PREDICTOR_REGISTRY, Predictor
from src.eval_pipeline.metrics.silhouette import SilhouetteMetric
from src.eval_pipeline.metrics.ari import ARIMetric
from src.eval_pipeline.metrics.rotational_consistency import RotationalConsistencyMetric
from src.eval_pipeline.metrics.base import SingleRunMetric, MultiRunMetric
from src.eval_pipeline.config_merger import load_training_config, merge_and_save
from src.eval_pipeline.metrics.chamfer_distance import ReconstructionChamferMetric
from src.eval_pipeline.metrics.earth_mover_distance import ReconstructionEMDMetric
from src.vis_tools.tsne_vis import compute_tsne, save_tsne_plot

SINGLE_METRICS = {
    "silhouette": SilhouetteMetric,
    "ari": ARIMetric,
    "reconstruction_chamfer": ReconstructionChamferMetric,
    "reconstruction_emd": ReconstructionEMDMetric,
}

MULTI_METRICS = {
    "rotational_consistency": RotationalConsistencyMetric,
}


def _build_dataloader(
    cfg: DictConfig,
    model_type: str,
    batch_size: int | None = None,
    max_samples: int | None = None,
):
    file_paths = cfg.data.data_files
    if model_type == "autoencoder":
        from src.training_methods.autoencoder.eval_autoencoder import (
            create_autoencoder_dataloader,
        )

        return create_autoencoder_dataloader(
            cfg, file_paths, shuffle=False, max_samples=max_samples, batch_size=batch_size
        )
    if model_type == "spd":
        from src.training_methods.spd.eval_spd import create_spd_dataloader

        return create_spd_dataloader(
            cfg, file_paths, shuffle=False, max_samples=max_samples, batch_size=batch_size
        )
    if model_type == "soap":
        from src.training_methods.autoencoder.eval_autoencoder import (
            create_autoencoder_dataloader,
        )

        return create_autoencoder_dataloader(
            cfg, file_paths, shuffle=False, max_samples=max_samples, batch_size=batch_size
        )
    raise ValueError(f"Unknown model_type {model_type!r}")


def _cluster_latents(latents: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    """Cluster *latents* using *method* with given *params*.

    Supports ``kmeans``, ``gmm`` and ``hdbscan``.
    """

    method = method.lower()
    if method == "kmeans":
        from sklearn.cluster import KMeans

        # Cap n_clusters to the number of samples when necessary
        params = dict(params)  # shallow copy to avoid mutating caller
        n_clusters = params.get("n_clusters")
        if n_clusters is not None:
            n_samples = latents.shape[0]
            if n_samples < 1:
                raise ValueError("Cannot cluster: no samples provided in latents.")
            params["n_clusters"] = max(1, min(n_clusters, n_samples))

        model = KMeans(n_init="auto", **params)
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(**params)
    elif method == "hdbscan":
        from hdbscan import HDBSCAN

        model = HDBSCAN(**params)
    else:
        raise ValueError(f"Unknown clustering method {method!r}")
    return model.fit_predict(latents)


def run(eval_config_path: str) -> Dict[str, Any]:
    """Execute the evaluation pipeline described in *eval_config_path*."""

    eval_cfg = OmegaConf.load(eval_config_path)
    checkpoint = eval_cfg.prediction.checkpoint
    train_cfg = load_training_config(checkpoint)
    checkpoint_stem = os.path.splitext(os.path.basename(checkpoint))[0]
    now_str = datetime.now().strftime("%m-%d_%H-%M")
    out_dir = os.path.join("output/eval_results", f"{train_cfg.model_type}_{now_str}_{checkpoint_stem}")

    # ------------------------------------------------------------------
    # Compose dataset config (training data cfg + overrides from eval cfg)
    # ------------------------------------------------------------------
    dataset_cfg = OmegaConf.load(eval_cfg.dataset.config)
    if eval_cfg.dataset.get("override"):
        dataset_cfg = OmegaConf.merge(dataset_cfg, eval_cfg.dataset.override)
    train_cfg.data = dataset_cfg

    device = (
        f"cuda:{eval_cfg.prediction.cuda_device}"
        if (eval_cfg.prediction.cuda_device is not None and torch.cuda.is_available())
        else "cpu"
    )

    PredictorCls = PREDICTOR_REGISTRY[eval_cfg.prediction.model_type]
    predictor: Predictor = PredictorCls.from_checkpoint(train_cfg, checkpoint, device)

    batch_size_override = eval_cfg.prediction.get("batch_size")
    max_samples_override = eval_cfg.prediction.get("max_samples")
    dataloader = _build_dataloader(
        train_cfg,
        eval_cfg.prediction.model_type,
        batch_size=batch_size_override,
        max_samples=max_samples_override,
    )
    
    print("Predicting...")
    preds = predictor.predict(dataloader)
    results: Dict[str, Any] = {
        "latents": preds.latents,
        "reconstructions": preds.reconstructions,
        "metrics": {},
    }
    print("Predictions done")
    # Pre-compute a single t-SNE embedding for all clustering visualizations
    tsne_coords: np.ndarray | None = None
    try:
        tsne_coords = compute_tsne(preds.latents, perplexity=20, n_iter=1000)
        print("t-SNE embedding computed")
    except Exception as e:
        print(f"Warning: failed to compute t-SNE embedding: {e}")
    metrics_cfg = eval_cfg.get("metrics", {})
    single_run_cfg = metrics_cfg.get("single_run", [])

    # Support multiple clustering methods, each with its own parameters.
    # Fallback to legacy single method/params if provided.
    clusterings_cfg = metrics_cfg.get("clusterings")
    if not clusterings_cfg:
        clustering_method = metrics_cfg.get("clustering_method", "kmeans")
        clustering_params = metrics_cfg.get("clustering_params", {})
        clusterings_cfg = [{"method": clustering_method, "params": clustering_params}]

    # Compute clustering-independent single-run metrics once
    for m in single_run_cfg:
        cls = SINGLE_METRICS[m.name]
        if not getattr(cls, "requires_clustering", True):
            print("Running single run metric (no clustering):", m.name)
            metric: SingleRunMetric = cls()
            params = m.get("params", {})
            results["metrics"][m.name] = metric.run_once(
                preds.latents,
                reconstructions=preds.reconstructions,
                originals=preds.originals,
                **params,
            )
            print(results["metrics"][m.name])

    for clustering in clusterings_cfg:
        method = clustering["method"]
        cparams = clustering.get("params", {})
        print(f"Clustering using {method} with params: {cparams}")
        cluster_labels = _cluster_latents(preds.latents, method, cparams)

        # Save t-SNE plot for this clustering
        if tsne_coords is not None:
            plot_path = os.path.join(out_dir, "plots", f"tsne_{method}.png")
            try:
                save_tsne_plot(
                    tsne_coords,
                    cluster_labels,
                    out_file=plot_path,
                    title=f"t-SNE colored by {method}",
                )
                print(f"Saved t-SNE plot to {plot_path}")
            except Exception as e:
                print(f"Warning: failed to save t-SNE plot for {method}: {e}")

        for m in single_run_cfg:
            cls = SINGLE_METRICS[m.name]
            if not getattr(cls, "requires_clustering", True):
                continue  # already computed once

            print("Running single run metric:", m.name, "with", method)
            metric: SingleRunMetric = cls()
            params = m.get("params", {})
            key = f"{m.name}:{method}"
            results["metrics"][key] = metric.run_once(
                preds.latents,
                cluster_labels=cluster_labels,
                reconstructions=preds.reconstructions,
                originals=preds.originals,
                **params,
            )
            print(results["metrics"][key])
    
    print("Single run metrics done")
    for m in metrics_cfg.get("multi_run", []):
        print("Running multi run metric:", m.name)
        cls = MULTI_METRICS[m.name]
        metric: MultiRunMetric = cls(predictor, **m.get("params", {}))
        results["metrics"][m.name] = metric.run_once(
            preds.latents, points=preds.originals, **m.get("params", {})
        )
    print("Multi run metrics done")
    merge_and_save(out_dir, train_cfg, eval_cfg, results["metrics"], checkpoint_path=checkpoint)
    print("Results saved")
    return results

if __name__ == "__main__":
    run("configs/eval_configs/eval_expl.yaml")