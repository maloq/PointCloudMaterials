from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _matrix_batches(values: np.ndarray, batch_size: int):
    for start in range(0, int(values.shape[0]), int(batch_size)):
        yield np.asarray(values[start : start + batch_size], dtype=np.float64)


@dataclass
class LinearVAMP:
    """Numerically stable two-sided linear VAMP estimator.

    Covariances are accumulated in float64 in two passes. Near-null covariance
    directions are removed before ridge-regularized whitening, avoiding the large
    amplification caused by whitening redundant encoder dimensions.
    """

    regularization: float = 1.0e-6
    eigenvalue_cutoff: float = 1.0e-10
    covariance_batch_size: int = 65536

    mean0_: np.ndarray | None = None
    mean1_: np.ndarray | None = None
    whitening0_: np.ndarray | None = None
    whitening1_: np.ndarray | None = None
    left_singular_vectors_: np.ndarray | None = None
    right_singular_vectors_: np.ndarray | None = None
    singular_values_: np.ndarray | None = None
    covariance_eigenvalues0_: np.ndarray | None = None
    covariance_eigenvalues1_: np.ndarray | None = None
    ridge0_: float | None = None
    ridge1_: float | None = None
    n_samples_: int | None = None

    def fit(self, z0: np.ndarray, z1: np.ndarray) -> "LinearVAMP":
        present = np.asarray(z0)
        future = np.asarray(z1)
        if present.ndim != 2 or future.ndim != 2 or present.shape != future.shape:
            raise ValueError(
                "LinearVAMP expects equally shaped 2D arrays Z0 and Ztau, "
                f"got {present.shape} and {future.shape}."
            )
        if present.shape[0] < 2:
            raise ValueError(f"LinearVAMP needs at least two pairs, got {present.shape[0]}.")
        if self.regularization < 0.0 or self.eigenvalue_cutoff < 0.0:
            raise ValueError(
                "regularization and eigenvalue_cutoff must be non-negative, "
                f"got {self.regularization} and {self.eigenvalue_cutoff}."
            )
        if int(self.covariance_batch_size) <= 0:
            raise ValueError(
                f"covariance_batch_size must be > 0, got {self.covariance_batch_size}."
            )

        n_samples = int(present.shape[0])
        sum0 = np.zeros(present.shape[1], dtype=np.float64)
        sum1 = np.zeros(future.shape[1], dtype=np.float64)
        for batch0, batch1 in zip(
            _matrix_batches(present, self.covariance_batch_size),
            _matrix_batches(future, self.covariance_batch_size),
            strict=True,
        ):
            sum0 += batch0.sum(axis=0)
            sum1 += batch1.sum(axis=0)
        mean0 = sum0 / float(n_samples)
        mean1 = sum1 / float(n_samples)

        feature_dim = int(present.shape[1])
        c00 = np.zeros((feature_dim, feature_dim), dtype=np.float64)
        c11 = np.zeros_like(c00)
        c01 = np.zeros_like(c00)
        for batch0, batch1 in zip(
            _matrix_batches(present, self.covariance_batch_size),
            _matrix_batches(future, self.covariance_batch_size),
            strict=True,
        ):
            centered0 = batch0 - mean0
            centered1 = batch1 - mean1
            c00 += centered0.T @ centered0
            c11 += centered1.T @ centered1
            c01 += centered0.T @ centered1
        c00 /= float(n_samples)
        c11 /= float(n_samples)
        c01 /= float(n_samples)
        c00 = 0.5 * (c00 + c00.T)
        c11 = 0.5 * (c11 + c11.T)

        whitening0, eig0, ridge0 = self._whitener(c00, name="C00")
        whitening1, eig1, ridge1 = self._whitener(c11, name="C11")
        koopman = whitening0.T @ c01 @ whitening1
        left, singular_values, right_t = np.linalg.svd(koopman, full_matrices=False)

        self.mean0_ = mean0
        self.mean1_ = mean1
        self.whitening0_ = whitening0
        self.whitening1_ = whitening1
        self.left_singular_vectors_ = left
        self.right_singular_vectors_ = right_t.T
        self.singular_values_ = singular_values
        self.covariance_eigenvalues0_ = eig0
        self.covariance_eigenvalues1_ = eig1
        self.ridge0_ = ridge0
        self.ridge1_ = ridge1
        self.n_samples_ = n_samples
        return self

    def _whitener(
        self,
        covariance: np.ndarray,
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        largest = float(eigenvalues[0])
        if largest <= 0.0:
            raise np.linalg.LinAlgError(f"{name} has no positive-variance directions.")
        threshold = float(self.eigenvalue_cutoff) * largest
        retained = eigenvalues > threshold
        if not np.any(retained):
            raise np.linalg.LinAlgError(
                f"{name} retained no covariance directions at relative cutoff "
                f"{self.eigenvalue_cutoff}."
            )
        ridge = float(self.regularization) * largest
        scales = 1.0 / np.sqrt(eigenvalues[retained] + ridge)
        whitening = eigenvectors[:, retained] * scales[None, :]
        return whitening, eigenvalues, ridge

    @property
    def rank(self) -> int:
        self._require_fitted()
        assert self.singular_values_ is not None
        return int(self.singular_values_.size)

    def left_singular_functions(self, values: np.ndarray, dimension: int | None = None) -> np.ndarray:
        self._require_fitted()
        assert self.mean0_ is not None
        assert self.whitening0_ is not None
        assert self.left_singular_vectors_ is not None
        dim = self._resolve_dimension(dimension)
        return (
            (np.asarray(values, dtype=np.float64) - self.mean0_)
            @ self.whitening0_
            @ self.left_singular_vectors_[:, :dim]
        )

    def right_singular_functions(self, values: np.ndarray, dimension: int | None = None) -> np.ndarray:
        self._require_fitted()
        assert self.mean1_ is not None
        assert self.whitening1_ is not None
        assert self.right_singular_vectors_ is not None
        dim = self._resolve_dimension(dimension)
        return (
            (np.asarray(values, dtype=np.float64) - self.mean1_)
            @ self.whitening1_
            @ self.right_singular_vectors_[:, :dim]
        )

    def transform(self, values: np.ndarray, dimension: int | None = None) -> np.ndarray:
        """Return present-state kinetic-map coordinates scaled by singular values."""
        dim = self._resolve_dimension(dimension)
        assert self.singular_values_ is not None
        functions = self.left_singular_functions(values, dim)
        return functions * self.singular_values_[None, :dim]

    def _resolve_dimension(self, dimension: int | None) -> int:
        self._require_fitted()
        assert self.singular_values_ is not None
        dim = int(self.singular_values_.size) if dimension is None else int(dimension)
        if dim <= 0 or dim > int(self.singular_values_.size):
            raise ValueError(
                f"dimension must be in [1, {self.singular_values_.size}], got {dim}."
            )
        return dim

    def _require_fitted(self) -> None:
        if self.singular_values_ is None:
            raise RuntimeError("LinearVAMP is not fitted.")

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            regularization=np.asarray(self.regularization, dtype=np.float64),
            eigenvalue_cutoff=np.asarray(self.eigenvalue_cutoff, dtype=np.float64),
            covariance_batch_size=np.asarray(self.covariance_batch_size, dtype=np.int64),
            mean0=self.mean0_,
            mean1=self.mean1_,
            whitening0=self.whitening0_,
            whitening1=self.whitening1_,
            left_singular_vectors=self.left_singular_vectors_,
            right_singular_vectors=self.right_singular_vectors_,
            singular_values=self.singular_values_,
            covariance_eigenvalues0=self.covariance_eigenvalues0_,
            covariance_eigenvalues1=self.covariance_eigenvalues1_,
            ridge0=np.asarray(self.ridge0_, dtype=np.float64),
            ridge1=np.asarray(self.ridge1_, dtype=np.float64),
            n_samples=np.asarray(self.n_samples_, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearVAMP":
        with np.load(Path(path), allow_pickle=False) as payload:
            model = cls(
                regularization=float(payload["regularization"]),
                eigenvalue_cutoff=float(payload["eigenvalue_cutoff"]),
                covariance_batch_size=int(payload["covariance_batch_size"]),
            )
            model.mean0_ = payload["mean0"].copy()
            model.mean1_ = payload["mean1"].copy()
            model.whitening0_ = payload["whitening0"].copy()
            model.whitening1_ = payload["whitening1"].copy()
            model.left_singular_vectors_ = payload["left_singular_vectors"].copy()
            model.right_singular_vectors_ = payload["right_singular_vectors"].copy()
            model.singular_values_ = payload["singular_values"].copy()
            model.covariance_eigenvalues0_ = payload["covariance_eigenvalues0"].copy()
            model.covariance_eigenvalues1_ = payload["covariance_eigenvalues1"].copy()
            model.ridge0_ = float(payload["ridge0"])
            model.ridge1_ = float(payload["ridge1"])
            model.n_samples_ = int(payload["n_samples"])
        return model


__all__ = ["LinearVAMP"]
