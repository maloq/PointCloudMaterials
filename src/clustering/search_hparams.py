from __future__ import annotations

# ==================== SPEED-ORIENTED DROP-IN VERSION ====================
# Key changes vs your original:
# - Parallelize HDBSCAN grid across cores (joblib, processes)
# - Fast scoring pass (sampled silhouette/CH/DB; compute DBCV only for top-K)
# - Float32 pipeline + randomized PCA for speed
# - Faster kNN-based epsilon scale using a sample
# - Control BLAS/numba threads to avoid oversubscription

# --- set thread env before importing heavy libs ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# Global cap for joblib parallelism; can be overridden via env var N_JOBS
MAX_N_JOBS = int(os.environ.get("N_JOBS", "8"))

import math
import datetime
import warnings
from typing import Sequence, Optional, Dict, Any, List, Tuple

import numpy as np

import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# optional parallel runtime
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# optional UMAP
try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# ------------------------ scoring helpers ----------------------------------

def _try_dbcv(X: np.ndarray, labels: np.ndarray, metric: str) -> float | float("nan"):
    try:
        from hdbscan.validity import validity_index  # type: ignore
    except Exception:
        try:
            validity_index = hdbscan.validity_index  # type: ignore[attr-defined]
        except Exception:
            return float("nan")
    if len(set(labels) - {-1}) == 0:
        return float("nan")
    try:
        return float(validity_index(X, labels, metric=metric))
    except Exception:
        return float("nan")


def _scores_for_labels(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str,
    *,
    sample_size: Optional[int] = 20000,
    sample_random_state: int = 42,
    compute_dbcv: bool = False,
) -> Dict[str, float]:
    frac_noise = float(np.mean(labels == -1))
    mask = labels != -1
    Xc = X[mask]
    labelsc = labels[mask]
    uniq = np.unique(labelsc)
    n_clusters = int(len(uniq))

    sil = ch = db = float("nan")
    if n_clusters >= 2 and len(Xc) >= 2 * n_clusters:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if sample_size is not None and len(Xc) > sample_size:
                    sil = float(
                        silhouette_score(
                            Xc, labelsc, metric=metric,
                            sample_size=sample_size, random_state=sample_random_state
                        )
                    )
                else:
                    sil = float(silhouette_score(Xc, labelsc, metric=metric))
            except Exception:
                pass
            try:
                ch = float(calinski_harabasz_score(Xc, labelsc))
            except Exception:
                pass
            try:
                db = float(davies_bouldin_score(Xc, labelsc))
            except Exception:
                pass

    dbcv = _try_dbcv(X, labels, metric) if compute_dbcv else float("nan")

    def _scale_pm1_to_01(x): return (x + 1.0) / 2.0
    def _squash_positive(x): return x / (x + 1.0)

    base_candidates: List[float] = []
    if not math.isnan(dbcv): base_candidates.append(_scale_pm1_to_01(dbcv))
    if not math.isnan(sil):  base_candidates.append(_scale_pm1_to_01(sil))
    if not math.isnan(ch):   base_candidates.append(_squash_positive(ch))
    if not math.isnan(db):   base_candidates.append(1.0 / (1.0 + db))

    base = float(np.nan) if len(base_candidates) == 0 else float(np.nanmax(base_candidates))
    return {
        "dbcv": dbcv,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "base_01": base,
        "frac_noise": frac_noise,
        "n_clusters": float(n_clusters),
    }


def _default_hdbscan_grids(X: np.ndarray, eps_scale_k: int = 5) -> Tuple[List[int], List[Optional[int]], List[float]]:
    N = len(X)
    mcs_raw = sorted({max(5, int(N * p)) for p in (0.005, 0.01, 0.02, 0.05, 0.1)})
    mcs_grid = [m for m in mcs_raw if m < N] or [min(N - 1, 5)]
    try:
        k = min(max(2, eps_scale_k), max(2, N - 1))
        # sample up to 50k points for epsilon scaling
        sample_N = min(N, 50000)
        if N > sample_N:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=sample_N, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xs)
        dists, _ = nn.kneighbors(Xs, return_distance=True)
        scale = float(np.median(dists[:, -1]))
        eps_grid = sorted({max(0.0, float(e)) for e in [0.0, 0.25 * scale, 0.5 * scale, 1.0 * scale, 2.0 * scale]})
    except Exception:
        eps_grid = [0.0]
    ms_template = [None, 1]
    return mcs_grid, ms_template, eps_grid


# ------------------------ core HDBSCAN search (parallel, 2-pass scoring) ---

def _grid_search_hdbscan_on_split(
    fit_X: np.ndarray,
    score_X: np.ndarray,
    metric: str = "euclidean",
    min_cluster_size_grid: Optional[Sequence[int]] = None,
    min_samples_grid: Optional[Sequence[Optional[int]]] = None,
    cluster_selection_epsilon_grid: Optional[Sequence[float]] = None,
    noise_penalty: float = 0.5,
    min_clusters: int = 1,
    max_clusters: Optional[int] = None,
    eps_scale_k: int = 5,
    *,
    n_jobs: int = MAX_N_JOBS,
    backend: str = "loky",
    topk_for_dbcv: int = 5,
    score_sample_size: Optional[int] = 20000,
    score_random_state: int = 42,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:

    auto_mcs, auto_ms_template, auto_eps = _default_hdbscan_grids(fit_X, eps_scale_k=eps_scale_k)
    mcs_grid = list(min_cluster_size_grid) if min_cluster_size_grid is not None else auto_mcs
    eps_grid = list(cluster_selection_epsilon_grid) if cluster_selection_epsilon_grid is not None else auto_eps
    ms_grid_template = list(min_samples_grid) if min_samples_grid is not None else auto_ms_template

    # Build param list
    param_list: List[Tuple[int, Optional[int], float]] = []
    for mcs in mcs_grid:
        ms_grid = ms_grid_template.copy()
        if mcs // 2 not in [ms for ms in ms_grid if ms is not None]: ms_grid.append(mcs // 2)
        if mcs not in [ms for ms in ms_grid if ms is not None]:      ms_grid.append(mcs)
        for ms in ms_grid:
            for eps in eps_grid:
                param_list.append((int(mcs), (None if ms is None else int(ms)), float(eps)))

    def _one_trial(mcs: int, ms: Optional[int], eps: float) -> Dict[str, Any]:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=None if ms is None else ms,
                cluster_selection_epsilon=eps,
                prediction_data=True,
                metric=metric,
                core_dist_n_jobs=1,  # single-threaded inside each worker
            ).fit(fit_X)
            labels, _ = hdbscan.approximate_predict(clusterer, score_X)
        except Exception as e:
            return {
                "min_cluster_size": mcs,
                "min_samples": ms,
                "cluster_selection_epsilon": eps,
                "metric": metric,
                "error": repr(e),
                "score": float("nan"),
                "base_01": float("nan"),
                "n_clusters": 0,
                "frac_noise": float("nan"),
                "dbcv": float("nan"),
                "silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan"),
            }

        s = _scores_for_labels(
            score_X, labels, metric,
            sample_size=score_sample_size,
            sample_random_state=score_random_state,
            compute_dbcv=False,  # fast pass
        )
        ncl = int(s["n_clusters"])
        if ncl < min_clusters or (max_clusters is not None and ncl > max_clusters):
            penalized = 0.0
        else:
            base = s["base_01"]
            penalized = 0.0 if math.isnan(base) else base * (1.0 - noise_penalty * s["frac_noise"])
            penalized = float(max(0.0, min(1.0, penalized)))

        return {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "cluster_selection_epsilon": eps,
            "metric": metric,
            "n_clusters": ncl,
            "frac_noise": s["frac_noise"],
            "dbcv": s["dbcv"],  # nan in fast pass
            "silhouette": s["silhouette"],
            "calinski_harabasz": s["calinski_harabasz"],
            "davies_bouldin": s["davies_bouldin"],
            "base_01": s["base_01"],
            "score": penalized,
        }

    if _HAS_JOBLIB and n_jobs != 1 and len(param_list) > 1:
        trials = Parallel(n_jobs=min(MAX_N_JOBS, int(n_jobs)), backend=backend, prefer="processes", max_nbytes=None)(
            delayed(_one_trial)(mcs, ms, eps) for (mcs, ms, eps) in param_list
        )
    else:
        trials = [_one_trial(mcs, ms, eps) for (mcs, ms, eps) in param_list]

    # Refine: compute DBCV on top-K
    ranked = sorted([t for t in trials if not math.isnan(t["score"])], key=lambda t: t["score"], reverse=True)
    finalists = ranked[:max(0, min(topk_for_dbcv, len(ranked)))]
    for t in finalists:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(t["min_cluster_size"]),
                min_samples=None if t["min_samples"] is None else int(t["min_samples"]),
                cluster_selection_epsilon=float(t["cluster_selection_epsilon"]),
                prediction_data=True,
                metric=metric,
                core_dist_n_jobs=1,
            ).fit(fit_X)
            labels, _ = hdbscan.approximate_predict(clusterer, score_X)
            s = _scores_for_labels(
                score_X, labels, metric,
                compute_dbcv=True,
                sample_size=score_sample_size,
                sample_random_state=score_random_state
            )
            t.update({k: s[k] for k in ("dbcv", "silhouette", "calinski_harabasz", "davies_bouldin", "base_01")})
            ncl = int(s["n_clusters"])
            if ncl < min_clusters or (max_clusters is not None and ncl > max_clusters):
                penalized = 0.0
            else:
                base = s["base_01"]
                penalized = 0.0 if math.isnan(base) else base * (1.0 - noise_penalty * s["frac_noise"])
                penalized = float(max(0.0, min(1.0, penalized)))
            t["score"] = penalized
        except Exception:
            pass

    # Select best
    best = None
    for t in trials:
        s = float(t["score"])
        if math.isnan(s): 
            continue
        if best is None or s > best[0]:
            best = (s, {
                "min_cluster_size": int(t["min_cluster_size"]),
                "min_samples": (None if t["min_samples"] is None else int(t["min_samples"])),
                "cluster_selection_epsilon": float(t["cluster_selection_epsilon"]),
                "metric": metric,
            })
    if best is None:
        raise RuntimeError("HDBSCAN search failed for all parameter combinations.")
    best_score, best_params = best
    return best_params, best_score, trials


# ------------------------ reducer helpers ----------------------------------

def _fit_transform_reducer(
    spec: Dict[str, Any],
    fit_latents: np.ndarray,
    all_latents: np.ndarray,
    fit_idx: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    kind = spec.get("kind", None)
    if kind is None:
        return (spec, all_latents, fit_latents)

    if kind == "pca":
        safe_nc = min(int(spec["n_components"]), fit_latents.shape[1])
        if safe_nc < 1:
            raise ValueError("PCA n_components < 1 after safety check.")
        reducer = PCA(
            n_components=safe_nc,
            whiten=bool(spec["whiten"]),
            random_state=int(spec["random_state"]),
            svd_solver="randomized",  # faster on large data
        )
        reducer.fit(fit_latents)
        all_red = reducer.transform(all_latents).astype(np.float32, copy=False)
        fit_red = all_red[fit_idx] if fit_idx is not None else all_red
        return (spec, all_red, fit_red)

    if kind == "umap":
        if not _HAS_UMAP:
            raise ImportError("UMAP requested but not installed.")
        umap_kwargs: Dict[str, Any] = {
            "n_components": int(spec["n_components"]),
            "low_memory": True,
        }
        if "n_neighbors" in spec and spec["n_neighbors"] is not None:
            umap_kwargs["n_neighbors"] = int(spec["n_neighbors"])
        if "min_dist" in spec and spec["min_dist"] is not None:
            umap_kwargs["min_dist"] = float(spec["min_dist"])
        if "random_state" in spec and spec["random_state"] is not None:
            umap_kwargs["random_state"] = int(spec["random_state"])
        reducer = umap.UMAP(**umap_kwargs)
        reducer.fit(fit_latents)
        all_red = reducer.transform(all_latents).astype(np.float32, copy=False)
        fit_red = all_red[fit_idx] if fit_idx is not None else all_red
        return (spec, all_red, fit_red)

    raise ValueError(f"Unknown reducer kind: {kind}")


# ------------------------ master search: adds dim reduction ----------------

def search_hdbscan_with_dim_reduction(
    train_latents: np.ndarray,
    eval_latents: np.ndarray,
    # subsampling (same semantics as your pipeline)
    subsample_size: int | None = None,
    subsample_random_state: int = 42,
    # HDBSCAN grid (same metric used in scoring)
    hdbscan_metric: str = "euclidean",
    min_cluster_size_grid: Optional[Sequence[int]] = None,
    min_samples_grid: Optional[Sequence[Optional[int]]] = None,
    cluster_selection_epsilon_grid: Optional[Sequence[float]] = None,
    # dimensionality-reduction grids
    include_none: bool = True,
    # PCA
    pca_n_components_grid: Sequence[int] = (16, 32, 64),
    pca_whiten_grid: Sequence[bool] = (False,),
    pca_random_state: int = 42,
    # UMAP
    use_umap: bool = True,
    umap_n_components_grid: Sequence[int] = (2, 16, 32),
    umap_n_neighbors_grid: Optional[Sequence[int]] = None,
    umap_min_dist_grid: Optional[Sequence[float]] = None,
    umap_random_state: Optional[int] = None,
    # scoring controls
    noise_penalty: float = 0.5,
    min_clusters: int = 1,
    max_clusters: Optional[int] = None,
    eps_scale_k: int = 5,
    # output controls
    return_trials: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]] | None]:
    """
    Compare 'no reduction', PCA, and UMAP (if available) while tuning HDBSCAN.
    Returns best full config dict ready to plug into your predict_clusters.
    """

    # 0) concat + dtype for speed
    all_latents = np.ascontiguousarray(np.concatenate([train_latents, eval_latents], axis=0), dtype=np.float32)

    # 1) subsample split
    if subsample_size is not None:
        if subsample_size > len(all_latents):
            raise ValueError(f"subsample_size {subsample_size} > total {len(all_latents)}")
        rng = np.random.default_rng(subsample_random_state)
        fit_idx = rng.choice(len(all_latents), size=subsample_size, replace=False)
        fit_latents = all_latents[fit_idx]
    else:
        fit_idx = None
        fit_latents = all_latents

    # 2) build reducer candidates
    reducer_specs: List[Dict[str, Any]] = []

    if include_none:
        reducer_specs.append({"kind": None})

    # PCA candidates
    for nc in pca_n_components_grid:
        for w in pca_whiten_grid:
            reducer_specs.append({
                "kind": "pca",
                "n_components": int(nc),
                "whiten": bool(w),
                "random_state": int(pca_random_state),
            })

    # UMAP candidates
    if use_umap and _HAS_UMAP:
        for nc in umap_n_components_grid:
            neighbor_values = list(umap_n_neighbors_grid) if umap_n_neighbors_grid is not None else [None]
            min_dist_values = list(umap_min_dist_grid) if umap_min_dist_grid is not None else [None]
            for nn in neighbor_values:
                for md in min_dist_values:
                    spec: Dict[str, Any] = {
                        "kind": "umap",
                        "n_components": int(nc),
                    }
                    if nn is not None:
                        spec["n_neighbors"] = int(nn)
                    if md is not None:
                        spec["min_dist"] = float(md)
                    if umap_random_state is not None:
                        spec["random_state"] = int(umap_random_state)
                    reducer_specs.append(spec)
    elif use_umap and not _HAS_UMAP and verbose:
        print("UMAP not available; skipping its candidates.")

    all_trials: List[Dict[str, Any]] = []
    best_overall: Tuple[float, Dict[str, Any]] | None = None

    # 3) evaluate each reducer candidate (possibly in parallel)
    def _eval_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        try:
            spec, all_red, fit_red = _fit_transform_reducer(spec, fit_latents, all_latents, fit_idx)
        except Exception as e:
            rec = {
                "dim_reduction": spec.get("kind", None),
                **{k: v for k, v in spec.items() if k != "kind"},
                "error": f"reducer_failed: {repr(e)}",
                "score": float("nan"),
            }
            return {"best_hdb": None, "best_score": float("nan"), "trials": [rec], "spec": spec}

        best_hdb, best_score, trials = _grid_search_hdbscan_on_split(
            fit_X=fit_red,
            score_X=all_red,
            metric=hdbscan_metric,
            min_cluster_size_grid=min_cluster_size_grid,
            min_samples_grid=min_samples_grid,
            cluster_selection_epsilon_grid=cluster_selection_epsilon_grid,
            noise_penalty=noise_penalty,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            eps_scale_k=eps_scale_k,
            n_jobs=MAX_N_JOBS,   # parallel across HDBSCAN params (capped)
            backend="loky",
            topk_for_dbcv=5,
            score_sample_size=20000,
            score_random_state=42,
        )
        for t in trials:
            t.update({
                "dim_reduction": spec.get("kind", None),
                **{k: v for k, v in spec.items() if k != "kind"},
            })
        return {"best_hdb": best_hdb, "best_score": best_score, "trials": trials, "spec": spec}

    if _HAS_JOBLIB and len(reducer_specs) > 1:
        results = Parallel(n_jobs=MAX_N_JOBS, backend="loky", prefer="processes", max_nbytes=None)(
            delayed(_eval_spec)(spec) for spec in reducer_specs
        )
    else:
        results = [_eval_spec(spec) for spec in reducer_specs]

    for r in results:
        all_trials.extend(r["trials"])
        if r["best_hdb"] is None:
            continue
        spec = r["spec"]
        kind = spec.get("kind", None)
        best_hdb, best_score = r["best_hdb"], r["best_score"]
        if best_overall is None or best_score > best_overall[0]:
            best_pack = {
                # reducer side
                "dim_reduction": (None if kind is None else kind),
                "n_components": (spec.get("n_components", None)),
                "pca_whiten": (spec.get("whiten", False) if kind == "pca" else False),
                "umap_n_neighbors": (spec.get("n_neighbors", None) if kind == "umap" else None),
                "umap_min_dist": (spec.get("min_dist", None) if kind == "umap" else None),
                "umap_random_state": (spec.get("random_state", None) if kind == "umap" else None),
                # hdbscan side
                "hdbscan_min_cluster_size": best_hdb["min_cluster_size"],
                "hdbscan_min_samples": best_hdb["min_samples"],
                "hdbscan_cluster_selection_epsilon": best_hdb["cluster_selection_epsilon"],
                "hdbscan_metric": best_hdb["metric"],
                # echo subsampling for convenience
                "subsample_size": subsample_size,
                "subsample_random_state": subsample_random_state,
            }
            best_overall = (best_score, best_pack)

    if best_overall is None:
        raise RuntimeError("No valid reducer/HDBSCAN combination succeeded.")

    best_score, best_config = best_overall
    return best_config, best_score, (all_trials if return_trials else None)


# ------------------------ example script entrypoint ------------------------

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from src.eval_pipeline.predict_functions import save_latents_to_file, load_model_for_inference, load_latents
    from src.clustering.run_clustering import predict_clusters

    folder = "output/spd/2025-08-10_23-07-22"
    train_files = ["166ps", "170ps", "174ps", "177ps", "240ps"]
    eval_files = ["175ps"]
    train_latents, train_points, train_originals, train_labels, train_coords = load_latents(folder, train_files)
    eval_latents, eval_points, eval_originals, eval_labels, eval_coords = load_latents(folder, eval_files)

    train_coords = np.squeeze(train_coords)
    eval_coords = np.squeeze(eval_coords)

    print(train_coords.shape)
    print(eval_coords.shape)

    best_cfg, score, trials = search_hdbscan_with_dim_reduction(
        train_latents, eval_latents,
        subsample_size=None,                # or an int, to match your pipeline
        subsample_random_state=42,
        hdbscan_metric="euclidean",
        min_cluster_size_grid=[15, 25, 50, 100],
        min_samples_grid=[None, 1, 8, 16, 32],
        cluster_selection_epsilon_grid=[0.0, 0.1, 0.2],
        # Dim-reduction grids (tweak as you like)
        include_none=True,
        pca_n_components_grid=(16, 32, 64),
        pca_whiten_grid=(False,),
        use_umap=True,
        umap_n_components_grid=(2, 10, 32),
        umap_n_neighbors_grid= None,
        umap_min_dist_grid= None,
        # Scoring preferences
        noise_penalty=0.5,
        min_clusters=1,
        max_clusters=None,
        verbose=False,
    )

    print("Best config (reducer + HDBSCAN):", best_cfg, "score:", score)

    labels_out = predict_clusters(
        train_latents, eval_latents, eval_coords,
        algorithm="hdbscan",
        # reducer
        dim_reduction=best_cfg["dim_reduction"],
        n_components=(best_cfg["n_components"] or 32),
        pca_whiten=best_cfg["pca_whiten"],
        umap_n_neighbors=(best_cfg["umap_n_neighbors"] or 15),
        umap_min_dist=(best_cfg["umap_min_dist"] or 0.1),
        umap_random_state=(best_cfg["umap_random_state"] or 42),
        # subsampling (optional)
        subsample_size=best_cfg["subsample_size"],
        subsample_random_state=best_cfg["subsample_random_state"],
        # hdbscan
        hdbscan_min_cluster_size=best_cfg["hdbscan_min_cluster_size"],
        hdbscan_min_samples=best_cfg["hdbscan_min_samples"],
        hdbscan_cluster_selection_epsilon=best_cfg["hdbscan_cluster_selection_epsilon"],
        hdbscan_metric=best_cfg["hdbscan_metric"],
    )
    # save labels_out to file
    out_folder = "output/clustering"
    os.makedirs(out_folder, exist_ok=True)
    np.savez_compressed(f"{out_folder}/labels_out_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz", labels=labels_out)