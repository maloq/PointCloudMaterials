from __future__ import annotations
from pathlib import Path
from typing import List, Iterable, Tuple, Union, Sequence, Dict, Any

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# 1.  Helper – build SOAP (wraps the one in soap_parquet.py, but self-contained)
# --------------------------------------------------------------------------
def build_soap(
    *,
    species: Sequence[Union[str, int]],
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.3,
    periodic: bool = False,
    compression_mode: str = "off",
    dtype: str = "float64",
) -> SOAP:
    """Return a configured DScribe SOAP object."""
    return SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        compression={"mode": compression_mode, "species_weighting": None},
        sparse=False,
        dtype=dtype,
    )

# --------------------------------------------------------------------------
# 2.  One-off routine – fit PCA on the whole (or a representative) data set
# --------------------------------------------------------------------------
def fit_soap_pca(
    point_clouds: Iterable[np.ndarray],
    *,
    species: str = "Al",
    soap_params: Dict[str, Any] | None = None,
    n_components: int | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Tuple[SOAP, PCA]:
    """
    Compute SOAP for **all** supplied point clouds, fit a PCA, and return
    both the SOAP descriptor and the trained PCA object.

    *point_clouds* – iterable of (N, 3) numpy arrays.
    """
    soap_params = soap_params or {}
    soap = build_soap(species=[species], **soap_params)
    rows: List[np.ndarray] = []

    for i, xyz in enumerate(point_clouds):
        # ------------------------------------------------------------------
        # Skip empty point clouds early (can happen when filtering neighbours)
        # ------------------------------------------------------------------
        if len(xyz) == 0:
            if verbose:
                print(f"  skipping empty point cloud #{i} (no atoms)")
            continue
        atoms = Atoms(symbols=[species] * len(xyz), positions=xyz)
        # SOAP for all centres → (N_atoms, n_features); store them all
        all_soap_vectors = soap.create(atoms, n_jobs=n_jobs)
        rows.append(all_soap_vectors)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  processed {i+1:,} point clouds")

    if not rows:
        raise ValueError(
            "No valid (non-empty) point clouds were supplied – cannot fit PCA."
        )
    X = np.vstack(rows)  # shape (n_samples, n_features)
    if verbose:
        print(f"Fitting PCA on {X.shape[0]:,} samples × {X.shape[1]:,} dims …")

    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    pca.fit(X)
    return soap, pca

# --------------------------------------------------------------------------
# 3.  Inference routine – same public contract as autoencoder_predict_latent
# --------------------------------------------------------------------------
def soap_pca_predict_latent(
    pc: Union[np.ndarray, "torch.Tensor"],
    *,
    soap: SOAP,
    pca: PCA,
    species: str = "Al",
    aggregate: str = "mean",
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Parameters
    ----------
    pc
        Either a single point cloud ``(N, 3)`` or a batch ``(B, N, 3)``.
        Accepts NumPy arrays or torch tensors.
    soap / pca
        Objects returned by :func:`fit_soap_pca`.
    aggregate
        How to collapse atom-wise SOAP to a fixed length vector.
        Currently only ``"mean"`` is implemented.
    Returns
    -------
    np.ndarray
        If *pc* is 2-D → shape ``(d,)``; if 3-D → ``(B, d)``.
        Here ``d = pca.n_components_`` (or full SOAP dim if PCA kept all).
    """
    import torch  # local import avoids unconditional dependency

    # ------------------------------------------------------------------
    # Normalise input shape to (B, N, 3)
    # ------------------------------------------------------------------
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = np.asarray(pc, dtype=np.float64)

    if pc.ndim == 2:                      # (N, 3) → (1, N, 3)
        pc = pc[None, ...]
        squeeze_back = True
    elif pc.ndim == 3:                    # already batch
        squeeze_back = False
    else:
        raise ValueError(f"Expected shape (N,3) or (B,N,3); got {pc.shape}")

    # ------------------------------------------------------------------
    # SOAP + aggregation for every item in the batch
    # ------------------------------------------------------------------
    batch_vecs: List[np.ndarray] = []
    for xyz in pc:                        # iterate over batch dimension
        if len(xyz) == 0:
            raise ValueError("Input point cloud is empty – cannot compute SOAP descriptor.")
        atoms = Atoms(symbols=[species] * len(xyz), positions=xyz)
        per_atom = soap.create(atoms, n_jobs=n_jobs)

        if aggregate == "mean":
            descriptor = per_atom.mean(axis=0)          # (F,)
        else:
            raise NotImplementedError(f"aggregate={aggregate!r}")

        batch_vecs.append(descriptor)

    X = np.vstack(batch_vecs)             # (B, F)

    # ------------------------------------------------------------------
    # PCA transform
    # ------------------------------------------------------------------
    reps = pca.transform(X)               # (B, d)

    return reps[0] if squeeze_back else reps



if __name__ == "__main__":
    """
    Run a couple of simple checks when this file is executed directly.

    These are *not* exhaustive unit tests – just a quick way to see whether the
    plumbing between build_soap / fit_soap_pca / soap_pca_predict_latent works.
    They deliberately use very small hyper-parameters so that they finish within
    a few seconds on a CPU-only machine.
    """
    import torch

    rng = np.random.default_rng(0)

    # Create a small batch of random point clouds (B = 4, N = 30)
    batch_pc = rng.random((4, 30, 3)) * 5.0  # positions in an arbitrary 5 Å box

    # Fit SOAP + PCA on the batch (tiny parameters → fast)
    soap_params = dict(r_cut=3.0, n_max=2, l_max=2, sigma=0.3)
    soap, pca = fit_soap_pca(
        batch_pc,
        species="Al",
        soap_params=soap_params,
        n_components=2,
        verbose=False,
    )

    print(f"Fitted PCA keeps {pca.n_components_} components")

    # ------------------------------------------------------------------
    # 1) Single point cloud, NumPy input
    # ------------------------------------------------------------------
    rep_np = soap_pca_predict_latent(batch_pc[0], soap=soap, pca=pca)
    assert rep_np.shape == (pca.n_components_,)
    print(" NumPy input → OK", rep_np)

    # ------------------------------------------------------------------
    # 2) Single point cloud, torch.Tensor input
    # ------------------------------------------------------------------
    rep_torch = soap_pca_predict_latent(
        torch.from_numpy(batch_pc[1]), soap=soap, pca=pca
    )
    assert rep_torch.shape == (pca.n_components_,)
    print(" Torch input → OK", rep_torch)

    # ------------------------------------------------------------------
    # 3) Batched point clouds
    # ------------------------------------------------------------------
    reps_batch = soap_pca_predict_latent(batch_pc, soap=soap, pca=pca)
    assert reps_batch.shape == (batch_pc.shape[0], pca.n_components_)
    print(" Batched input → OK, shape:", reps_batch.shape)

    print("All smoke tests passed ✔")