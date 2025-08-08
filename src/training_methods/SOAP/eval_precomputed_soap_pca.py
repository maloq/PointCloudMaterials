from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Sequence, Optional

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence[float]]

# -----------------------------------------------------------------------------
# 1) Loaders – stack precomputed SOAP-PCA features and coordinates
# -----------------------------------------------------------------------------

def load_precomputed_from_parquet(
    parquet_paths: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    num_coord_dims: int = 3,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and vertically stack precomputed feature rows and coordinates from one
    or more Parquet files. The Parquet layout is assumed to be:

      • first `num_coord_dims` columns → Cartesian coords (x, y, z)
      • remaining columns              → precomputed feature vector (e.g. SOAP-PCA)

    Returns
    -------
    (features, coords)
        features: shape (N, D)
        coords:   shape (N, num_coord_dims)
    """
    if isinstance(parquet_paths, (str, Path)):
        files = [Path(parquet_paths)]
    else:
        files = [Path(p) for p in parquet_paths]

    feat_list: List[np.ndarray] = []
    coord_list: List[np.ndarray] = []
    for p in files:
        mat = pd.read_parquet(p, engine="pyarrow").to_numpy()
        if mat.shape[1] <= num_coord_dims:
            raise ValueError(
                f"{p!r} has only {mat.shape[1]} cols but num_coord_dims={num_coord_dims}."
            )
        coords = mat[:, :num_coord_dims]
        feats = mat[:, num_coord_dims:]
        coord_list.append(coords)
        feat_list.append(feats)

    coords_all = np.vstack(coord_list).astype(dtype, copy=False)
    feats_all = np.vstack(feat_list).astype(dtype, copy=False)
    return feats_all, coords_all


def load_precomputed_from_npz(
    npz_paths: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    feature_keys: Sequence[str] = ("latents", "features"),
    coords_key: str = "coords",
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and stack from one or more .npz bundles. We will look for one of the
    feature keys ('latents' by default) and for 'coords'.

    Returns (features, coords).
    """
    if isinstance(npz_paths, (str, Path)):
        files = [Path(npz_paths)]
    else:
        files = [Path(p) for p in npz_paths]

    feat_list: List[np.ndarray] = []
    coord_list: List[np.ndarray] = []
    for p in files:
        with np.load(p) as data:
            feat_arr = None
            for k in feature_keys:
                if k in data:
                    feat_arr = data[k]
                    break
            if feat_arr is None:
                raise KeyError(
                    f"None of feature keys {feature_keys} found in {p.name}."
                )
            if coords_key not in data:
                raise KeyError(f"coords key '{coords_key}' not found in {p.name}.")
            feat_list.append(np.asarray(feat_arr))
            coord_list.append(np.asarray(data[coords_key]))
    feats_all = np.vstack(feat_list).astype(dtype, copy=False)
    coords_all = np.vstack(coord_list).astype(dtype, copy=False)
    return feats_all, coords_all


# -----------------------------------------------------------------------------
# 2) Predict-like helpers – same public contract as eval_autoencoder, but trivial
# -----------------------------------------------------------------------------

def predict_latents_from_arrays(features: np.ndarray) -> np.ndarray:
    """
    Mirror of `predict_latents(...)` but for precomputed features.
    Simply returns the provided array (shape (N, D)).
    """
    return np.asarray(features)


def predict_single_latent_from_array(feature_row: ArrayLike) -> np.ndarray:
    """
    Mirror of `predict_single_latent(...)` – returns a single latent vector (shape (D,)).
    """
    arr = np.asarray(feature_row)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D feature vector; got shape {arr.shape}")
    return arr


def predict_coords_from_arrays(coords: np.ndarray) -> np.ndarray:
    """
    Convenience accessor – returns the coordinates paired with the features.
    """
    return np.asarray(coords)


# -----------------------------------------------------------------------------
# 3) Save utility – persist a bundle for downstream clustering/evaluation
# -----------------------------------------------------------------------------

def save_latents_bundle(
    out_file: Union[str, Path],
    *,
    latents: np.ndarray,
    coords: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Path:
    """
    Save a compressed bundle with latents (+ optional coords/labels) to NPZ.
    Keys: 'latents', 'coords' (if provided), 'labels' (if provided).
    """
    out_path = Path(out_file)
    payload = {"latents": np.asarray(latents).astype(np.float32)}
    if coords is not None:
        payload["coords"] = np.asarray(coords).astype(np.float32)
    if labels is not None:
        payload["labels"] = np.asarray(labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


# -----------------------------------------------------------------------------
# 4) Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: load directly from Parquet files (coords | precomputed SOAP-PCA)
    parquet_files = [
        "datasets/Al/soap_features/166ps_selected.parquet",
        "datasets/Al/soap_features/170ps_selected.parquet",
    ]
    feats, xyz = load_precomputed_from_parquet(parquet_files, num_coord_dims=3)
    latents = predict_latents_from_arrays(feats)
    print("Loaded latents:", latents.shape, "coords:", xyz.shape)

    save_path = save_latents_bundle(
        "output/soap_pca_precomputed/latents_bundle.npz", latents=latents, coords=xyz
    )
    print(f"[✓] Saved {save_path}")