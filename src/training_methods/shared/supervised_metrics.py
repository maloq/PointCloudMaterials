"""Evaluate cached embeddings; callers own logging and cache lifetime."""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC, SVC

from src.utils.evaluation_metrics import (
    compute_cluster_metrics,
    compute_embedding_quality_metrics,
    ensure_kmeansplusplus,
    finite_float,
    primary_kmeansplusplus_hungarian_key,
    stabilize_class_metric_keys,
)
from .supervised_features import (
    _collect_split_supervised_features,
    _collect_rotated_split_supervised_features,
    _get_stage_dataset,
    cache_limit_for_stage,
)


def _format_label_histogram(labels: np.ndarray, *, max_entries: int = 10) -> str:
    arr = np.asarray(labels).reshape(-1)
    if arr.size == 0:
        return "[]"
    unique, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    parts: list[str] = []
    for idx in order[:max_entries]:
        try:
            label_text = str(int(unique[idx]))
        except (TypeError, ValueError):
            label_text = str(unique[idx])
        parts.append(f"{label_text}:{int(counts[idx])}")
    if unique.size > max_entries:
        parts.append(f"...(+{int(unique.size - max_entries)} classes)")
    return "[" + ", ".join(parts) + "]"


def _validate_cached_supervised_arrays(
    stage: str,
    latents: np.ndarray,
    labels: np.ndarray | None,
    encoder_features: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    lat = _flatten_features(latents)
    y = None if labels is None else np.asarray(labels).reshape(-1)
    enc = None if encoder_features is None else _flatten_features(encoder_features)

    if y is not None and y.shape[0] != lat.shape[0]:
        raise RuntimeError(
            "Cached metric arrays have mismatched latent/label rows: "
            f"stage='{stage}', latents={lat.shape[0]}, labels={y.shape[0]}."
        )
    if enc is not None and enc.shape[0] != lat.shape[0]:
        raise RuntimeError(
            "Cached metric arrays have mismatched latent/encoder rows:"
            f" stage='{stage}', latents={lat.shape[0]},"
            f" encoder_features={enc.shape[0]}."
        )

    bad_rows = ~np.isfinite(lat).all(axis=1)
    if y is not None and np.issubdtype(y.dtype, np.floating):
        bad_rows |= ~np.isfinite(y)
    if enc is not None:
        bad_rows |= ~np.isfinite(enc).all(axis=1)

    if bad_rows.any():
        raise ValueError(
            f"Non-finite cached metric rows: stage={stage!r}, "
            f"bad_rows={int(bad_rows.sum())}/{lat.shape[0]}."
        )
    return lat, y, enc


def _flatten_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _prepare_features_and_labels(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = _flatten_features(features)
    y = np.asarray(labels).reshape(-1)
    x, y, _ = _validate_cached_supervised_arrays("linear_svm", x, y, None)
    return x, y


def _compute_linear_svm_accuracy(
    features: np.ndarray, labels: np.ndarray
) -> float | None:
    x, y = _prepare_features_and_labels(features, labels)
    if x.shape[0] <= 1:
        return None

    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        return None

    min_class = int(counts.min())
    if min_class < 2:
        return None
    cv = max(2, min(5, min_class))

    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(dual="auto", random_state=42, max_iter=20000),
    )
    try:
        scores = cross_val_score(clf, x, y, cv=cv, scoring="accuracy", n_jobs=1)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"Linear SVM cross-val failed: {exc}")
        return None
    if scores.size == 0:
        return None
    return float(np.mean(scores))


def _compute_linear_svm_train_to_test_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    svm_c: float = 0.018,
) -> float | None:
    x_train, y_train = _prepare_features_and_labels(train_features, train_labels)
    x_test, y_test = _prepare_features_and_labels(test_features, test_labels)
    if x_train.shape[0] <= 1 or x_test.shape[0] <= 0:
        return None

    unique_train = np.unique(y_train)
    if unique_train.size < 2:
        return None

    unique_test = np.unique(y_test)
    if unique_test.size < 1:
        return None

    # SVM evaluation protocol aligned with the official RI-MAE code.
    clf = SVC(C=float(svm_c), kernel="linear")
    try:
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(f"Linear SVM train-to-test eval failed: {exc}")
        return None
    if pred.shape[0] == 0:
        return None
    return float(np.mean(pred == y_test))


def _sample_probe_rows(
    latents: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if max_samples <= 0 or int(latents.shape[0]) <= max_samples:
        return latents
    rng = np.random.default_rng(int(seed))
    indices = np.sort(
        rng.choice(int(latents.shape[0]), size=max_samples, replace=False)
    )
    return latents[indices]


def _resolve_probe_k(module, stage: str, labels: np.ndarray | None) -> int:
    configured = getattr(module, "probe_k", None)
    if configured is not None:
        k = int(configured)
    else:
        if labels is None:
            raise ValueError(
                "Probe metrics for unlabeled data need an explicit cluster"
                f" count. stage='{stage}' has no cached class_id labels. Set"
                " probe_k to the number of KMeans clusters to probe, or set"
                " enable_probe_metrics=false."
            )
        stage_l = str(stage).lower()
        if stage_l in {"val", "test"}:
            inferred = _resolve_hungarian_eval_k(module, stage_l, labels)
        else:
            inferred = _infer_label_cluster_count(labels)
        if inferred is None:
            raise ValueError(
                "Probe cluster count could not be inferred from "
                f"stage='{stage}' labels. Set probe_k explicitly."
            )
        k = int(inferred)

    if k < 2:
        raise ValueError(f"probe_k must be >= 2, got {k} for stage='{stage}'.")
    return k


def _probe_cluster_entropy(assignments: np.ndarray, *, k: int) -> float:
    counts = np.bincount(
        np.asarray(assignments, dtype=np.int64), minlength=int(k)
    ).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        raise RuntimeError("KMeans probe produced zero assigned samples.")
    probs = counts[counts > 0.0] / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(int(k)))


def _resolve_probe_best_k_range(
    module,
    *,
    fixed_k: int,
    sample_count: int,
    stage: str,
) -> tuple[int, int]:
    min_k = int(getattr(module, "probe_best_k_min", 2))
    configured_max = getattr(module, "probe_best_k_max", None)
    if configured_max is None:
        max_k = min(max(int(fixed_k), 12), 32, int(sample_count) - 1)
    else:
        max_k = int(configured_max)
    max_k = min(max_k, int(sample_count) - 1)
    if max_k < min_k:
        raise ValueError(
            "Probe best-k range is empty: "
            f"stage='{stage}', probe_best_k_min={min_k}, "
            f"probe_best_k_max={configured_max}, effective_max={max_k}, "
            f"samples={int(sample_count)}."
        )
    return min_k, max_k


def _compute_probe_best_num_clusters(
    features: np.ndarray,
    *,
    min_k: int,
    max_k: int,
    silhouette_samples: int,
    kmeans_n_init: int,
    seed: int,
    eps: float,
    stage: str,
) -> tuple[float, list[tuple[int, float]]]:
    best_k = int(min_k)
    best_score = -float("inf")
    silhouette_by_k: list[tuple[int, float]] = []
    sample_size = min(int(silhouette_samples), int(features.shape[0]))
    for candidate_k in range(int(min_k), int(max_k) + 1):
        model = KMeans(
            n_clusters=candidate_k,
            init="k-means++",
            n_init=kmeans_n_init,
            random_state=seed + candidate_k,
        )
        assignments = model.fit_predict(features)
        if assignments.shape[0] != features.shape[0]:
            raise RuntimeError(
                "KMeans best-k probe returned an unexpected assignment count:"
                f" stage='{stage}', k={candidate_k},"
                f" expected={features.shape[0]}, got={assignments.shape[0]}."
            )
        unique_assignments = np.unique(assignments)
        if unique_assignments.size < 2 or unique_assignments.size >= features.shape[0]:
            score = -1.0
        else:
            score = float(
                silhouette_score(
                    features,
                    assignments,
                    metric="euclidean",
                    sample_size=sample_size,
                    random_state=seed + candidate_k,
                )
            )
        if not np.isfinite(score):
            raise RuntimeError(
                "KMeans best-k probe produced a non-finite silhouette score: "
                f"stage='{stage}', k={candidate_k}, score={score!r}."
            )
        silhouette_by_k.append((int(candidate_k), score))
        if score > best_score + eps:
            best_score = score
            best_k = int(candidate_k)
    return float(best_k), silhouette_by_k


def _compute_probe_metrics(
    module,
    latents: np.ndarray,
    labels: np.ndarray | None,
    stage: str,
) -> tuple[dict[str, float], list[tuple[int, float]]]:
    x = np.asarray(latents, dtype=np.float32).reshape(latents.shape[0], -1)
    if x.shape[0] < 3:
        raise ValueError(
            "Need at least 3 cached samples for probe metrics, got"
            f" {x.shape[0]} for stage='{stage}'."
        )
    if x.shape[1] < 1:
        raise ValueError(
            f"Probe metrics received zero-dimensional latents for stage='{stage}'."
        )

    seed = int(getattr(module, "probe_seed", getattr(module, "cluster_acc_seed", 0)))
    max_samples = int(getattr(module, "probe_max_samples", 8192) or 0)
    x = _sample_probe_rows(x, max_samples=max_samples, seed=seed)
    if not np.isfinite(x).all():
        raise RuntimeError(
            "Probe metrics received non-finite latents after cache"
            f" validation: stage='{stage}', shape={tuple(x.shape)}."
        )

    k = _resolve_probe_k(module, stage, labels)
    if int(x.shape[0]) <= k:
        raise ValueError(
            "Need more cached probe samples than clusters, got"
            f" samples={int(x.shape[0])}, probe_k={k}, stage='{stage}'."
            " Increase cache/probe sample limits."
        )

    target_variance = 0.95
    pca_max_components = int(getattr(module, "probe_pca_max_components", 64))
    silhouette_samples = int(getattr(module, "probe_silhouette_samples", 5000))
    kmeans_seed_count = int(getattr(module, "probe_kmeans_seed_count", 3))
    kmeans_n_init = int(getattr(module, "probe_kmeans_n_init", 5))
    eps = float(getattr(module, "probe_eps", 1e-12))

    x_centered = x - x.mean(axis=0, keepdims=True)
    rank = min(int(x_centered.shape[0]) - 1, int(x_centered.shape[1]))
    if rank < 1:
        raise ValueError(
            f"PCA probe rank must be >= 1, got rank={rank}, stage='{stage}', "
            f"shape={tuple(x_centered.shape)}."
        )

    if float(np.var(x_centered, dtype=np.float64)) <= eps:
        return {
            "pca95": 0.0,
            "effective_rank": 0.0,
            "silhouette": 0.0,
            "stability": 0.0,
            "balance": 0.0,
            "best_k": 1.0,
        }, [(1, 0.0)]

    pca = PCA(n_components=rank, svd_solver="full", random_state=seed)
    projected = pca.fit_transform(x_centered).astype(np.float32, copy=False)
    explained = np.nan_to_num(
        np.asarray(pca.explained_variance_ratio_, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if explained.size == 0 or float(explained.sum()) <= eps:
        components_95 = 0
    else:
        cumulative = np.cumsum(explained)
        components_95 = int(
            np.searchsorted(cumulative, target_variance, side="left") + 1
        )
        components_95 = min(components_95, int(explained.size))
    explained_sum = float(explained.sum())
    if explained_sum <= eps:
        effective_rank = 0.0
    else:
        variance_distribution = explained / explained_sum
        nonzero_variance = variance_distribution[variance_distribution > eps]
        effective_rank = float(
            np.exp(-np.sum(nonzero_variance * np.log(nonzero_variance)))
        )

    pca_dims = min(pca_max_components, int(projected.shape[1]))
    features = normalize(projected[:, :pca_dims], norm="l2", axis=1)
    if not np.isfinite(features).all():
        raise RuntimeError(
            "Probe PCA/L2 normalization produced non-finite values:"
            f" stage='{stage}', projected_shape={tuple(projected.shape)},"
            f" pca_dims={pca_dims}."
        )
    feature_norms = np.linalg.norm(features, axis=1)
    if float(feature_norms.max()) <= eps:
        return {
            "pca95": float(components_95),
            "effective_rank": effective_rank,
            "silhouette": 0.0,
            "stability": 0.0,
            "balance": 0.0,
            "best_k": 1.0,
        }, [(1, 0.0)]

    best_k_min, best_k_max = _resolve_probe_best_k_range(
        module,
        fixed_k=k,
        sample_count=int(features.shape[0]),
        stage=stage,
    )
    best_num_clusters, silhouette_by_k = _compute_probe_best_num_clusters(
        features,
        min_k=best_k_min,
        max_k=best_k_max,
        silhouette_samples=silhouette_samples,
        kmeans_n_init=kmeans_n_init,
        seed=seed,
        eps=eps,
        stage=stage,
    )

    models = []
    assignments_by_seed = []
    for run_idx in range(kmeans_seed_count):
        run_seed = seed + run_idx
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=kmeans_n_init,
            random_state=run_seed,
        )
        assignments = model.fit_predict(features)
        if assignments.shape[0] != features.shape[0]:
            raise RuntimeError(
                "KMeans probe returned an unexpected assignment count:"
                f" stage='{stage}', expected={features.shape[0]},"
                f" got={assignments.shape[0]}."
            )
        models.append(model)
        assignments_by_seed.append(assignments)

    primary_model = models[0]
    primary_assignments = assignments_by_seed[0]
    unique_primary = np.unique(primary_assignments)
    entropy = _probe_cluster_entropy(primary_assignments, k=k)

    if unique_primary.size < 2 or unique_primary.size >= features.shape[0]:
        silhouette = 0.0
    else:
        sample_size = min(int(silhouette_samples), int(features.shape[0]))
        silhouette = float(
            silhouette_score(
                features,
                primary_assignments,
                metric="euclidean",
                sample_size=sample_size,
                random_state=seed,
            )
        )

    pairwise_ari = []
    degenerate_seed = False
    for i in range(len(assignments_by_seed)):
        if np.unique(assignments_by_seed[i]).size < 2:
            degenerate_seed = True
        for j in range(i + 1, len(assignments_by_seed)):
            pairwise_ari.append(
                adjusted_rand_score(assignments_by_seed[i], assignments_by_seed[j])
            )
    if not pairwise_ari:
        raise RuntimeError(
            "KMeans seed stability probe needs at least two KMeans runs. "
            f"Got probe_kmeans_seed_count={kmeans_seed_count}."
        )
    seed_ari = 0.0 if degenerate_seed else float(np.mean(pairwise_ari))

    metrics = {
        "pca95": float(components_95),
        "effective_rank": effective_rank,
        "silhouette": silhouette,
        "stability": seed_ari,
        "balance": entropy,
        "best_k": best_num_clusters,
    }
    for name, value in metrics.items():
        if not np.isfinite(float(value)):
            raise RuntimeError(
                f"Probe metric '{name}' is non-finite for stage='{stage}': {value!r}."
            )
    return metrics, silhouette_by_k


def _compute_rotated_test_accuracy_metrics(
    module,
    *,
    canonical_acc: float,
    canonical_nmi: float | None,
    canonical_ari: float | None,
    canonical_hungarian_eval_k: int | None,
) -> dict[str, float]:
    if not bool(getattr(module, "enable_test_so3_metrics", True)):
        return {}

    rotation_runs = int(getattr(module, "test_so3_rotation_runs", 5))
    if rotation_runs < 1:
        raise ValueError(f"test_so3_rotation_runs must be >= 1, got {rotation_runs}.")
    rotation_seed = int(getattr(module, "test_so3_rotation_seed", 12345))
    sample_limit = cache_limit_for_stage(module, "test")

    run_acc_values: list[float] = []
    run_nmi_values: list[float] = []
    run_ari_values: list[float] = []
    resolved_k = canonical_hungarian_eval_k
    for run_idx in range(rotation_runs):
        run_features, run_labels = _collect_rotated_split_supervised_features(
            module,
            "test",
            max_samples=sample_limit,
            rotation_seed_base=rotation_seed,
            rotation_run_idx=run_idx,
        )
        run_k = _resolve_hungarian_eval_k(module, "test", run_labels)
        if run_k is None:
            raise ValueError(
                "Could not infer Hungarian evaluation k for rotated test"
                " metrics; ensure class labels are present and contain at"
                " least two classes."
            )
        if resolved_k is None:
            resolved_k = run_k
        elif int(resolved_k) != int(run_k):
            raise ValueError(
                "Inconsistent class count across canonical and rotated test"
                f" metrics: canonical k={resolved_k}, rotated k={run_k} (run"
                f" {run_idx + 1})."
            )

        run_metrics = (
            compute_cluster_metrics(
                run_features,
                run_labels,
                stage="test",
                hungarian_eval_k=int(resolved_k),
                acc_eval_methods=["kmeans++"],
                acc_eval_runs=1,
                acc_eval_runs_by_method={},
                acc_random_seed=rotation_seed + run_idx,
            )
            or {}
        )
        run_acc_key = primary_kmeansplusplus_hungarian_key(run_metrics, int(resolved_k))
        if run_acc_key is None:
            raise RuntimeError(
                "Rotated test metrics are missing kmeans++ Hungarian ACC. "
                f"Available keys: {sorted(run_metrics.keys())}."
            )
        run_acc = finite_float(run_metrics.get(run_acc_key))
        if run_acc is None:
            raise RuntimeError(
                f"Rotated test metric '{run_acc_key}' is not finite:"
                f" {run_metrics.get(run_acc_key)!r}."
            )
        run_nmi = finite_float(run_metrics.get("NMI"))
        if run_nmi is None:
            raise RuntimeError(
                "Rotated test metrics are missing finite NMI. "
                f"Available keys: {sorted(run_metrics.keys())}."
            )
        run_ari = finite_float(run_metrics.get("ARI"))
        if run_ari is None:
            raise RuntimeError(
                "Rotated test metrics are missing finite ARI. "
                f"Available keys: {sorted(run_metrics.keys())}."
            )
        run_acc_values.append(run_acc)
        run_nmi_values.append(run_nmi)
        run_ari_values.append(run_ari)

    if not run_acc_values:
        raise RuntimeError(
            "Rotated test metrics produced no valid ACC values across SO(3) runs."
        )
    if not run_nmi_values:
        raise RuntimeError(
            "Rotated test metrics produced no valid NMI values across SO(3) runs."
        )
    if not run_ari_values:
        raise RuntimeError(
            "Rotated test metrics produced no valid ARI values across SO(3) runs."
        )

    if resolved_k is None:
        raise RuntimeError("Resolved k is None after rotated test metric evaluation.")

    acc_arr = np.asarray(run_acc_values, dtype=np.float64)
    nmi_arr = np.asarray(run_nmi_values, dtype=np.float64)
    ari_arr = np.asarray(run_ari_values, dtype=np.float64)
    rotated_mean = float(acc_arr.mean())
    rotated_nmi_mean = float(nmi_arr.mean())
    rotated_ari_mean = float(ari_arr.mean())
    metrics: dict[str, float] = {
        "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED": rotated_mean,
        "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_STD": float(acc_arr.std()),
        "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_MIN": float(acc_arr.min()),
        "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_MAX": float(acc_arr.max()),
        "ACC_KMEANS_PLUSPLUS_HUNGARIAN_ROTATED_RUNS": float(acc_arr.size),
        "NMI_ROTATED": rotated_nmi_mean,
        "NMI_ROTATED_STD": float(nmi_arr.std()),
        "NMI_ROTATED_MIN": float(nmi_arr.min()),
        "NMI_ROTATED_MAX": float(nmi_arr.max()),
        "NMI_ROTATED_RUNS": float(nmi_arr.size),
        "ARI_ROTATED": rotated_ari_mean,
        "ARI_ROTATED_STD": float(ari_arr.std()),
        "ARI_ROTATED_MIN": float(ari_arr.min()),
        "ARI_ROTATED_MAX": float(ari_arr.max()),
        "ARI_ROTATED_RUNS": float(ari_arr.size),
        "SO3_ACCURACY_KMEANS_PLUSPLUS_HUNGARIAN": rotated_mean,
        "SO3_VS_CANONICAL_ACC_DELTA": rotated_mean - float(canonical_acc),
        "SO3_ROTATION_RUNS": float(acc_arr.size),
    }
    if canonical_nmi is not None:
        metrics["SO3_VS_CANONICAL_NMI_DELTA"] = rotated_nmi_mean - float(canonical_nmi)
    if canonical_ari is not None:
        metrics["SO3_VS_CANONICAL_ARI_DELTA"] = rotated_ari_mean - float(canonical_ari)
    if abs(float(canonical_acc)) > 1e-12:
        metrics["SO3_VS_CANONICAL_ACC_RATIO"] = rotated_mean / float(canonical_acc)
    return metrics


def _compute_train_to_test_svm_accuracy(
    module,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float | None:
    if not bool(getattr(module, "enable_train_split_svm_test_metric", True)):
        return None

    max_train_samples = int(getattr(module, "train_split_svm_max_samples", 0) or 0)
    if max_train_samples <= 0:
        max_train_samples = int(getattr(module, "max_supervised_samples", 8192) or 8192)

    train_features, train_labels = _collect_split_supervised_features(
        module,
        "train",
        max_samples=max_train_samples,
    )
    if train_features is None or train_labels is None:
        return None

    svm_c = float(getattr(module, "train_split_svm_c", 0.018))
    return _compute_linear_svm_train_to_test_accuracy(
        train_features,
        train_labels,
        test_features,
        test_labels,
        svm_c=svm_c,
    )


def _infer_label_cluster_count(labels: np.ndarray) -> int | None:
    if labels is None:
        return None
    arr = np.asarray(labels).reshape(-1)
    if arr.size == 0:
        return None
    k = int(np.unique(arr).size)
    return k if k > 1 else None


def _infer_dataset_class_count(module, stage: str) -> int | None:
    dataset = _get_stage_dataset(module, stage)
    if dataset is None:
        return None

    class_names = getattr(dataset, "class_names", None)
    if class_names is not None:
        if hasattr(class_names, "items"):
            k = len(dict(class_names))
            return k if k > 1 else None
        if hasattr(class_names, "__len__") and not isinstance(
            class_names, (str, bytes)
        ):
            k = len(class_names)
            return k if k > 1 else None

    num_classes = getattr(dataset, "num_classes", None)
    if callable(num_classes):
        value = num_classes()
    else:
        value = num_classes
    if value is not None:
        try:
            k = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {stage}_dataset.num_classes={value!r}; expected"
                " integer-like value."
            ) from None
        return k if k > 1 else None
    return None


def _resolve_hungarian_eval_k(module, stage: str, labels: np.ndarray) -> int | None:
    stage_l = str(stage).lower()
    if stage_l not in {"val", "test"}:
        return None

    observed_k = _infer_label_cluster_count(labels)
    dataset_k = _infer_dataset_class_count(module, stage_l)

    if dataset_k is not None:
        if observed_k is not None and observed_k != dataset_k:
            raise ValueError(
                "Hungarian ACC class-count mismatch: observed"
                f" {observed_k} unique labels in cached {stage_l} embeddings,"
                f" but dataset reports {dataset_k} classes. This typically"
                " means sampling limits dropped classes; increase"
                f" {'max_test_samples' if stage_l == 'test' else 'max_supervised_samples'} or"
                " disable that limit."
            )
        inferred_k = dataset_k
    else:
        inferred_k = observed_k

    configured = (
        getattr(module, "val_cluster_eval_k", None) if stage_l == "val" else None
    )
    if (
        configured is not None
        and inferred_k is not None
        and int(configured) != int(inferred_k)
    ):
        raise ValueError(
            "Hungarian ACC k must match the class count. Configured"
            f" val_cluster_eval_k={int(configured)}, inferred"
            f" classes={int(inferred_k)} for stage='{stage_l}'. Set"
            f" val_cluster_eval_k=null (recommended) or to {int(inferred_k)}."
        )
    return inferred_k


def evaluate_supervised_features(
    module, stage, latents, labels, encoder_features, *, enabled
):
    """Return class, probe and embedding metrics without emitting log records.

    Optional SO(3) and train-to-test SVM diagnostics collect their additional
    split features through supervised_features; the remaining metrics use only
    the supplied arrays.
    """
    stage_l = stage.lower()
    groups = {}
    if "class" in enabled and labels is not None:
        if stage_l == "val":
            acc_methods = ensure_kmeansplusplus(
                list(getattr(module, "val_cluster_acc_methods", []))
            )
            acc_runs = int(getattr(module, "val_cluster_acc_runs", 1))
            acc_runs_by_method = getattr(module, "val_cluster_acc_runs_by_method", {})
        elif stage_l == "test":
            acc_methods = ensure_kmeansplusplus(
                list(getattr(module, "test_cluster_acc_methods", ["kmeans++"]))
            )
            acc_runs = int(getattr(module, "test_cluster_acc_runs", 1))
            acc_runs_by_method = getattr(module, "test_cluster_acc_runs_by_method", {})
        else:
            acc_methods = []
            acc_runs = 1
            acc_runs_by_method = {}

        hungarian_eval_k = _resolve_hungarian_eval_k(module, stage_l, labels)
        if stage_l in {"val", "test"} and hungarian_eval_k is not None:
            if int(latents.shape[0]) < int(hungarian_eval_k):
                raise RuntimeError(
                    "Insufficient finite samples for Hungarian ACC"
                    f" evaluation: stage='{stage_l}',"
                    f" samples={int(latents.shape[0])},"
                    f" hungarian_eval_k={int(hungarian_eval_k)}. Increase"
                    " supervised cache limits or inspect non-finite latent"
                    " rows."
                )
        metrics = (
            compute_cluster_metrics(
                latents,
                labels,
                stage,
                hungarian_eval_k=hungarian_eval_k,
                acc_eval_methods=acc_methods,
                acc_eval_runs=max(1, acc_runs),
                acc_eval_runs_by_method=acc_runs_by_method,
                acc_random_seed=int(getattr(module, "cluster_acc_seed", 0)),
            )
            or {}
        )
        if stage_l == "test" and hungarian_eval_k is not None:
            canonical_acc_key = primary_kmeansplusplus_hungarian_key(
                metrics, hungarian_eval_k
            )
            if canonical_acc_key is None:
                raise RuntimeError(
                    "Canonical test metrics are missing kmeans++ Hungarian"
                    " ACC after clustering evaluation. stage='test',"
                    f" samples={int(latents.shape[0])},"
                    f" latent_dim={int(latents.shape[1])},"
                    f" unique_labels={int(np.unique(labels).size)},"
                    f" label_histogram={_format_label_histogram(labels)},"
                    f" hungarian_eval_k={int(hungarian_eval_k)},"
                    f" acc_methods={list(acc_methods)},"
                    f" acc_runs={max(1, int(acc_runs))},"
                    f" acc_runs_by_method={acc_runs_by_method},"
                    f" available_keys={sorted(metrics.keys())}. This usually"
                    " indicates clustering failures in all ACC runs."
                )
            canonical_acc = finite_float(metrics.get(canonical_acc_key))
            if canonical_acc is None:
                raise RuntimeError(
                    f"Canonical test metric '{canonical_acc_key}' is not"
                    f" finite: {metrics.get(canonical_acc_key)!r}."
                )
            metrics["ACC_KMEANS_PLUSPLUS_HUNGARIAN_CANONICAL"] = canonical_acc
            canonical_nmi = finite_float(metrics.get("NMI"))
            canonical_ari = finite_float(metrics.get("ARI"))
            metrics.update(
                _compute_rotated_test_accuracy_metrics(
                    module,
                    canonical_acc=canonical_acc,
                    canonical_nmi=canonical_nmi,
                    canonical_ari=canonical_ari,
                    canonical_hungarian_eval_k=int(hungarian_eval_k),
                )
            )
        metrics = stabilize_class_metric_keys(
            metrics,
            hungarian_eval_k=(hungarian_eval_k if stage_l in {"val", "test"} else None),
        )
        if bool(getattr(module, "enable_svm_accuracy", True)):
            if stage_l in {"val", "test"} and encoder_features is not None:
                svm_acc = _compute_linear_svm_accuracy(encoder_features, labels)
                if svm_acc is not None:
                    metrics["ENCODER_LINEAR_SVM_ACCURACY"] = svm_acc
            if stage_l == "test":
                svm_eval_features = (
                    encoder_features if encoder_features is not None else latents
                )
                train_to_test_svm = _compute_train_to_test_svm_accuracy(
                    module,
                    svm_eval_features,
                    labels,
                )
                if train_to_test_svm is not None:
                    metrics["ENCODER_LINEAR_SVM_TRAIN_TO_TEST_ACCURACY"] = (
                        train_to_test_svm
                    )
        groups["class"] = {name.lower(): value for name, value in metrics.items()}

    if "probe" in enabled:
        probe_metrics, _ = _compute_probe_metrics(module, latents, labels, stage_l)
        groups["probe"] = probe_metrics

    if "embedding" in enabled:
        if labels is None:
            raise ValueError(
                "enable_embedding_metrics=true requires class_id labels, but"
                f" no labels were cached for stage='{stage_l}'. Disable"
                " enable_embedding_metrics for unlabeled data."
            )
        emb_metrics = compute_embedding_quality_metrics(
            latents, labels, include_expensive=(stage == "test")
        )
        groups["embedding"] = emb_metrics

    return groups
