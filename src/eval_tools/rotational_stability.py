from __future__ import annotations

"""Rotational‑robustness benchmark – **config‑driven & easily extensible**.

The module is split into three conceptual layers:

1. **Core utilities** – geometry helpers, sampling, clustering, etc.  *Rarely
   changed.*
2. **Predictors** – plug‑ins that map point clouds → fixed‑size
   representations (latent vectors **or** integer labels).  New models are
   added simply by decorating a class with ``@register_predictor``.
3. **Entry points** – ``run(cfg)`` executes the benchmark for a YAML/``dict``
   config; the *only* command‑line usage left is an *optional* ``python ‑m
   rotational_robustness_benchmark config.yml`` convenience wrapper.

Example **YAML** (drop this next to your script)::

    # benchmark.yml
    snapshot_files:
      - datasets/Al/inherent_configurations_off/175ps.off

    predictor:
      type: autoencoder            # "autoencoder" | "soap_pca" | …your model…
      checkpoint: output/2025‑06‑16/16‑39‑19/model.ckpt
      cuda_device: 0

    benchmark:
      n_centres: null              # keep all
      sphere_radius: 7.4
      n_points: 128
      n_rotations: 5
      metric: ari                  # "ari" | "fraction"
      align: false
      rng_seed: 42
      n_clusters: 6
      batch_size: 4096

Run with::

    import rotational_robustness_benchmark as rrb
    rrb.run("benchmark.yml")

or directly::

    python -m rotational_robustness_benchmark benchmark.yml

Adding a new predictor is **one small class**::

    @register_predictor("my_model")
    class MyPredictor(Predictor):
        def __init__(self, some_param: str):
            self._model = load_my_model(some_param)

        def predict(self, pcs: np.ndarray) -> np.ndarray:
            return self._model(pcs)  # shape (B, d)

That's it – the registry + config take care of the rest.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import abc
import importlib
import sys

import numpy as np
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# External scientific stack
# -----------------------------------------------------------------------------
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

import torch


from src.data_utils.prepare_data import read_off_file, process_sample
from src.data_utils.data_load import pc_normalize
from src.autoencoder.eval_autoencoder import (
    autoencoder_predict_latent,
    load_model_and_config,
)
from src.SOAP.predict_soap_pca import soap_pca_predict_latent, fit_soap_pca

# Optional: maximise matmul precision for Ampere GPUs and newer
torch.set_float32_matmul_precision("high")

# =============================================================================
# 0.  Generic helper utilities (unchanged from the original script)
# =============================================================================


def _sample_rotations(n: int, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """Return *n* SO(3) rotation matrices.  Index 0 is identity."""
    if n < 1:
        raise ValueError("n_rotations must be ≥ 1 (identity included).")
    rng = np.random.default_rng(rng)
    mats = list(R.random(n - 1, random_state=rng).as_matrix())
    mats.insert(0, np.eye(3))
    return mats


def _hungarian_relabel(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Map *pred* labels on to *ref* labels via maximal overlap."""
    ref_labels, ref_inv = np.unique(ref, return_inverse=True)
    pred_labels, pred_inv = np.unique(pred, return_inverse=True)

    contingency = np.zeros((ref_labels.size, pred_labels.size), dtype=int)
    np.add.at(contingency, (ref_inv, pred_inv), 1)

    cost = contingency.max() - contingency  # maximise overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {pred_labels[j]: ref_labels[i] for i, j in zip(row_ind, col_ind)}
    return np.vectorize(lambda lbl: mapping.get(lbl, lbl))(pred)


def _fraction_unchanged(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(ref == pred))


# -----------------------------------------------------------------------------
# 1.  Centre‑selection helpers
# -----------------------------------------------------------------------------


def _pick_centre_indices(
    points: np.ndarray,
    radius: float,
    n_max: Optional[int] = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick centre atoms as **nearest atoms to the nodes of a regular cubic grid**."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spacing = radius
    coords = [np.arange(mins[d], maxs[d] + 1e-6, spacing) for d in range(3)]
    grid = (
        np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(points.dtype)
    )

    tree = KDTree(points)
    _, nearest_idx = tree.query(grid)
    centre_idx = np.unique(nearest_idx)

    if n_max is not None and centre_idx.size > n_max:
        rng = np.random.default_rng(rng)
        centre_idx = rng.choice(centre_idx, size=n_max, replace=False)

    return centre_idx



def _filter_edge_centres(
    points: np.ndarray,
    centre_idx: np.ndarray,
    radius: float,
    factor: float = 1.5,
) -> np.ndarray:
    """Drop centres that are closer than *factor·radius* to any box face."""
    if centre_idx.size == 0:
        return centre_idx

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    margin = factor * radius

    centres = points[centre_idx]
    inside = np.all((centres >= mins + margin) & (centres <= maxs - margin), axis=1)
    return centre_idx[inside]


# -----------------------------------------------------------------------------
# 2.  Local‑environment sampling
# -----------------------------------------------------------------------------


def _extract_samples(
    points: np.ndarray,
    centre_idx: np.ndarray,
    radius: float,
    n_points: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return list of local point clouds and mask of *valid* centres."""
    tree = KDTree(points)
    centres = points[centre_idx]
    samples: List[np.ndarray] = []
    valid_mask = np.ones(len(centres), dtype=bool)
    for i, c in enumerate(centres):
        struct, _, _ = process_sample(points, tree, c, radius, n_points)
        if struct is None:
            valid_mask[i] = False
            continue
        samples.append(pc_normalize(struct))
    return samples, valid_mask


# =============================================================================
# 3.  Predictor plug‑in system – add your models here
# =============================================================================


class Predictor(abc.ABC):
    """Minimal interface all *predictors* must implement."""

    @abc.abstractmethod
    def predict(self, pcs: np.ndarray | List[np.ndarray]) -> np.ndarray:
        """Return latent vectors **or** integer labels for *pcs* (B, N, 3)."""

    # Optional – override if your model needs access to *all* snapshots first
    def fit(self, snapshots: List[np.ndarray]) -> None:  # noqa: D401 – imperative
        """(Re‑)train / pre‑compute statistics for this predictor, if needed."""
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Predictor registry & helper decorator
# ─────────────────────────────────────────────────────────────────────────────
_PREDICTOR_REGISTRY: Dict[str, Type[Predictor]] = {}


def register_predictor(name: str):
    """Decorator: ``@register_predictor("foo")`` adds *foo* → class mapping."""

    def _decorator(cls: Type[Predictor]):
        if name in _PREDICTOR_REGISTRY:
            raise KeyError(f"Predictor '{name}' already registered")
        _PREDICTOR_REGISTRY[name] = cls
        cls.__registry_name__ = name  # type: ignore[attr-defined]
        return cls

    return _decorator


# -----------------------------------------------------------------------------
# 3.1  Autoencoder latent predictor
# -----------------------------------------------------------------------------


@register_predictor("autoencoder")
class AutoencoderPredictor(Predictor):
    """Wrapper around a trained *Point‑Net‑style* autoencoder."""

    def __init__(self, checkpoint: str, cuda_device: int = 0):
        self.model, _cfg, self.device = load_model_and_config(checkpoint, cuda_device)

    # Autoencoder already supports batched input → just delegate
    def predict(self, pcs: np.ndarray | List[np.ndarray]) -> np.ndarray:  # type: ignore[override]
        return autoencoder_predict_latent(pcs, model=self.model, device=self.device)


# -----------------------------------------------------------------------------
# 3.2  SOAP + PCA predictor
# -----------------------------------------------------------------------------


@register_predictor("soap_pca")
class SOAPPredictor(Predictor):
    """Density‑based SOAP descriptor compressed by PCA."""

    def __init__(self, species: str, r_cut: float = 7.4, n_max: int = 8, l_max: int = 6, sigma: float = 0.2, pca_components: int = 16):
        from ase.atoms import Atoms  # Imported lazily – avoids heavy deps unless used

        self.species = species
        # Store descriptor parameters – actual construction deferred to *fit*
        self._soap_params = dict(r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma)
        self._pca_components = pca_components
        self._soap_desc = None  # set in *fit*
        self._pca_model = None

    # The SOAP pipeline needs to *see* the full dataset to centre PCA
    def fit(self, snapshots: List[np.ndarray]) -> None:  # noqa: D401 – imperative style
        soap_desc, pca_model = fit_soap_pca(
            snapshots,
            species=self.species,
            soap_params=self._soap_params,
            n_components=self._pca_components,
        )
        self._soap_desc, self._pca_model = soap_desc, pca_model

    def predict(self, pcs: np.ndarray | List[np.ndarray]) -> np.ndarray:  # type: ignore[override]
        if self._soap_desc is None or self._pca_model is None:
            raise RuntimeError("SOAPPredictor.fit() must be called before predict()")
        return soap_pca_predict_latent(pcs, soap=self._soap_desc, pca=self._pca_model, species=self.species)


# =============================================================================
# 4.  Rotational‑robustness core (algorithm body unchanged, but parameterised)
# =============================================================================


def _predict_batch(
    predictor: Predictor,
    samples: List[np.ndarray],
    batch_size: Optional[int],
    desc: str,
) -> np.ndarray:
    """Utility: run *predictor* on *samples* with optional batching."""

    if batch_size is None or batch_size <= 1:
        reps = [predictor.predict(s) for s in tqdm(samples, desc=desc, leave=False)]
        reps = np.asarray(reps)
    else:
        reps_chunks: List[np.ndarray] = []
        for i in tqdm(range(0, len(samples), batch_size), desc=desc, leave=False):
            batch = np.stack(samples[i : i + batch_size], axis=0)  # (B, N, 3)
            reps_chunks.append(predictor.predict(batch))
        reps = np.concatenate(reps_chunks, axis=0)
    return reps



def rotational_robustness(
    predictor: Predictor,
    points: np.ndarray,
    *,
    n_centres: Optional[int] = 200,
    sphere_radius: float = 6.0,
    n_points: int = 128,
    n_rotations: int = 24,
    metric: str = "ari",  # "ari" | "fraction"
    align: bool = True,
    rng_seed: int | None = 0,
    n_clusters: int = 8,
    edge_margin_factor: float = 1.5,
    batch_size: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Compute **mean** rotational‑robustness score & distribution (per rotation)."""

    rng = np.random.default_rng(rng_seed)

    # 1. Pick central atoms on a grid & drop edge‑neighbours
    centre_idx = _pick_centre_indices(points, sphere_radius, n_max=n_centres, rng=rng)
    centre_idx = _filter_edge_centres(points, centre_idx, sphere_radius, factor=edge_margin_factor)
    if centre_idx.size == 0:
        raise ValueError(
            "All candidate centres were filtered out. Adjust 'edge_margin_factor', 'n_centres' or input snapshot size."
        )

    # 2. Reference samples / predictions (no rotation)
    ref_samples, valid_mask = _extract_samples(points, centre_idx, sphere_radius, n_points)
    if not valid_mask.all():
        centre_idx = centre_idx[valid_mask]

    ref_reps = _predict_batch(predictor, ref_samples, batch_size, desc="Predicting reference samples")

    # 3. Generate rotations & compute scores
    R_mats = _sample_rotations(n_rotations, rng)
    scores: List[float] = []

    original_centres = points[centre_idx]

    for rot in tqdm(R_mats[1:], desc="Processing rotations"):  # skip identity
        pos_rot = points @ rot.T
        tree_rot = KDTree(pos_rot)
        _, new_idx = tree_rot.query(original_centres @ rot.T)
        rot_samples, _ = _extract_samples(pos_rot, new_idx, sphere_radius, n_points)
        pred_reps = _predict_batch(predictor, rot_samples, batch_size, desc="Predicting rotated samples")

        if metric == "ari":
            if ref_reps.ndim == 1:  # already labels
                ref_labels, pred_labels = ref_reps, pred_reps
            else:
                kmeans_ref = KMeans(n_clusters=n_clusters, random_state=rng_seed, n_init="auto")
                ref_labels = kmeans_ref.fit_predict(ref_reps)
                pred_labels = KMeans(n_clusters=n_clusters, random_state=rng_seed, n_init="auto").fit_predict(
                    pred_reps
                )
            score = adjusted_rand_score(ref_labels, pred_labels)
        elif metric == "fraction":
            if align:
                pred_reps = _hungarian_relabel(ref_reps, pred_reps)
            score = _fraction_unchanged(ref_reps, pred_reps)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

        scores.append(score)

    return float(np.mean(scores)), np.asarray(scores)


# =============================================================================
# 5.  Configuration – dataclasses & YAML loader
# =============================================================================


@dataclass
class BenchmarkParams:
    """Subset of *rotational_robustness* kwargs that can be set via YAML."""

    n_centres: Optional[int] = 200
    sphere_radius: float = 6.0
    n_points: int = 128
    n_rotations: int = 24
    metric: str = "ari"
    align: bool = True
    rng_seed: Optional[int] = 0
    n_clusters: int = 8
    edge_margin_factor: float = 1.5
    batch_size: Optional[int] = None


@dataclass
class Config:
    snapshot_files: List[str]
    predictor: Dict[str, Any]
    benchmark: BenchmarkParams = field(default_factory=BenchmarkParams)

    @staticmethod
    def from_yaml(path_or_str: str | Path | Dict[str, Any]) -> "Config":
        if isinstance(path_or_str, (str, Path)) and Path(path_or_str).is_file():
            with open(path_or_str, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif isinstance(path_or_str, dict):
            data = path_or_str
        else:
            raise FileNotFoundError(f"Cannot load config from: {path_or_str}")
        # *benchmark* part can be missing – defaults will fill in
        bench = BenchmarkParams(**data.get("benchmark", {}))
        return Config(
            snapshot_files=data["snapshot_files"],
            predictor=data["predictor"],
            benchmark=bench,
        )


# =============================================================================
# 6.  Public API – run() helper & __main__
# =============================================================================


def _load_points(files: List[str]) -> np.ndarray:
    """Read & concatenate all *OFF* snapshots listed in *files*."""
    all_points = [read_off_file(fp, verbose=True) for fp in files]
    return np.concatenate(all_points, axis=0)


def _build_predictor(spec: Dict[str, Any], snapshots: List[np.ndarray]) -> Predictor:
    """Instantiate *and* optionally ``fit`` a predictor from *spec*."""

    typ = spec["type"]
    cls = _PREDICTOR_REGISTRY.get(typ)
    if cls is None:
        raise KeyError(
            f"Predictor type '{typ}' is not registered. Implement it via @register_predictor and add to config."
        )

    # Remove the *type* key before passing remaining kwargs
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    predictor = cls(**kwargs)  # type: ignore[arg-type]

    # Some predictors (e.g. SOAP‑PCA) need to see samples first
    try:
        predictor.fit(snapshots)  # no‑op for predictors that don't override fit()
    except Exception as exc:  # noqa: BLE001 – propagate informative errors
        raise RuntimeError(f"Error during predictor.fit(): {exc}") from exc

    return predictor


def run(cfg: str | Path | Dict[str, Any]) -> None:
    """Convenience API: *cfg* → YAML/``dict`` → execute benchmark & print result."""
    config = Config.from_yaml(cfg)

    points = _load_points(config.snapshot_files)
    predictor = _build_predictor(config.predictor, [points])

    bench = config.benchmark
    mean, dist = rotational_robustness(
        predictor,
        points,
        n_centres=bench.n_centres,
        sphere_radius=bench.sphere_radius,
        n_points=bench.n_points,
        n_rotations=bench.n_rotations,
        metric=bench.metric,
        align=bench.align,
        rng_seed=bench.rng_seed,
        n_clusters=bench.n_clusters,
        edge_margin_factor=bench.edge_margin_factor,
        batch_size=bench.batch_size,
    )

    print(f"Rotational robustness ({config.predictor['type']}) = {mean:.3f}")


# -----------------------------------------------------------------------------
# 7.  Fallback *module‑as‑script* execution
# -----------------------------------------------------------------------------
# Allows `python -m rotational_robustness_benchmark cfg.yml` without argparse.
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    run("configs/autoencoder_80.yaml")