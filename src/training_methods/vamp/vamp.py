from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_score_name(score: str) -> str:
    score_name = str(score).strip().upper()
    if score_name in {"2", "VAMP2"}:
        return "VAMP2"
    if score_name in {"E", "VAMPE"}:
        return "VAMPE"
    raise ValueError(
        "score must be one of ['VAMP2', 'VAMPE'], "
        f"got {score!r}."
    )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"Expected a square matrix for symmetrization, got shape={tuple(arr.shape)}."
        )
    return 0.5 * (arr + arr.T)


@dataclass(frozen=True)
class CovarianceEstimate:
    mean_0: np.ndarray
    mean_t: np.ndarray
    c00: np.ndarray
    c01: np.ndarray
    c11: np.ndarray
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "feature_dim": int(self.c00.shape[0]),
        }


def estimate_covariances(x0: np.ndarray, x1: np.ndarray) -> CovarianceEstimate:
    data_0 = np.asarray(x0, dtype=np.float64)
    data_t = np.asarray(x1, dtype=np.float64)
    if data_0.ndim != 2 or data_t.ndim != 2:
        raise ValueError(
            "VAMP covariance estimation expects 2D feature matrices, "
            f"got x0.shape={tuple(data_0.shape)}, x1.shape={tuple(data_t.shape)}."
        )
    if data_0.shape != data_t.shape:
        raise ValueError(
            "Instantaneous and lagged feature matrices must have the same shape. "
            f"Got x0.shape={tuple(data_0.shape)}, x1.shape={tuple(data_t.shape)}."
        )
    sample_count = int(data_0.shape[0])
    if sample_count <= 1:
        raise ValueError(
            f"Need at least two lagged samples to estimate covariances, got {sample_count}."
        )

    mean_0 = data_0.mean(axis=0)
    mean_t = data_t.mean(axis=0)
    centered_0 = data_0 - mean_0[None, :]
    centered_t = data_t - mean_t[None, :]

    inv_count = 1.0 / float(sample_count)
    c00 = inv_count * centered_0.T @ centered_0
    c01 = inv_count * centered_0.T @ centered_t
    c11 = inv_count * centered_t.T @ centered_t
    return CovarianceEstimate(
        mean_0=mean_0,
        mean_t=mean_t,
        c00=_symmetrize(c00),
        c01=np.asarray(c01, dtype=np.float64),
        c11=_symmetrize(c11),
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class WhiteningModel:
    inverse_sqrt: np.ndarray
    eigenvalues: np.ndarray
    retained_eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    retained_mask: np.ndarray
    rank: int
    epsilon: float
    cutoff: float


def regularized_inverse_sqrt(
    matrix: np.ndarray,
    *,
    epsilon: float,
    eigenvalue_cutoff: float | None,
) -> WhiteningModel:
    cov = _symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    cutoff = float(epsilon if eigenvalue_cutoff is None else eigenvalue_cutoff)
    retained_mask = eigvals > cutoff
    if not np.any(retained_mask):
        raise np.linalg.LinAlgError(
            "No covariance eigenvalues survived the whitening cutoff. "
            f"cutoff={cutoff}, epsilon={epsilon}, eigvals_min={float(eigvals.min())}, "
            f"eigvals_max={float(eigvals.max())}."
        )
    retained_vals = eigvals[retained_mask]
    retained_vecs = eigvecs[:, retained_mask]
    inv_sqrt = retained_vecs @ np.diag(1.0 / np.sqrt(retained_vals + float(epsilon))) @ retained_vecs.T
    inv_sqrt = _symmetrize(inv_sqrt)
    return WhiteningModel(
        inverse_sqrt=inv_sqrt,
        eigenvalues=eigvals,
        retained_eigenvalues=retained_vals,
        eigenvectors=eigvecs,
        retained_mask=retained_mask,
        rank=int(retained_vals.size),
        epsilon=float(epsilon),
        cutoff=cutoff,
    )


def _small_spd_inverse_sqrt(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    cov = _symmetrize(matrix)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    if np.any(eigvals < -10.0 * float(epsilon)):
        raise np.linalg.LinAlgError(
            "Encountered a strongly indefinite matrix while computing a VAMP score. "
            f"min_eigenvalue={float(eigvals.min())}, epsilon={float(epsilon)}."
        )
    clipped = np.clip(eigvals, a_min=float(epsilon), a_max=None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(clipped)) @ eigvecs.T
    return _symmetrize(inv_sqrt)


class ManualVAMP:
    def __init__(
        self,
        *,
        lagtime: int | None = None,
        epsilon: float = 1.0e-6,
        eigenvalue_cutoff: float | None = None,
        scaling: str | None = None,
        dim: int | None = None,
    ) -> None:
        self.lagtime = None if lagtime is None else int(lagtime)
        self.epsilon = float(epsilon)
        self.eigenvalue_cutoff = None if eigenvalue_cutoff is None else float(eigenvalue_cutoff)
        self.scaling = None if scaling is None else str(scaling).strip().lower()
        if self.scaling not in {None, "", "kinetic_map"}:
            raise ValueError(
                "scaling must be None or 'kinetic_map', "
                f"got {scaling!r}."
            )
        self.dim = None if dim is None else int(dim)
        if self.dim is not None and self.dim <= 0:
            raise ValueError(f"dim must be > 0 when provided, got {self.dim}.")

        self.mean_0: np.ndarray | None = None
        self.mean_t: np.ndarray | None = None
        self.c00: np.ndarray | None = None
        self.c01: np.ndarray | None = None
        self.c11: np.ndarray | None = None
        self.sample_count: int | None = None

        self.whitening_0: WhiteningModel | None = None
        self.whitening_t: WhiteningModel | None = None
        self.half_weighted_koopman: np.ndarray | None = None
        self.left_singular_vectors: np.ndarray | None = None
        self.right_singular_vectors: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.left_coefficients: np.ndarray | None = None
        self.right_coefficients: np.ndarray | None = None
        self.koopman_matrix: np.ndarray | None = None
        self._fitted = False

    @property
    def feature_dim(self) -> int:
        if self.c00 is None:
            raise RuntimeError("ManualVAMP has not been fitted yet.")
        return int(self.c00.shape[0])

    @property
    def model_dim(self) -> int:
        if self.singular_values is None:
            raise RuntimeError("ManualVAMP has not been fitted yet.")
        return int(self.singular_values.shape[0])

    def fit(self, x0: np.ndarray, x1: np.ndarray) -> "ManualVAMP":
        covariances = estimate_covariances(x0, x1)
        return self.fit_from_covariances(covariances)

    def fit_from_covariances(self, covariances: CovarianceEstimate) -> "ManualVAMP":
        self.mean_0 = np.asarray(covariances.mean_0, dtype=np.float64)
        self.mean_t = np.asarray(covariances.mean_t, dtype=np.float64)
        self.c00 = _symmetrize(covariances.c00)
        self.c01 = np.asarray(covariances.c01, dtype=np.float64)
        self.c11 = _symmetrize(covariances.c11)
        self.sample_count = int(covariances.sample_count)

        self.whitening_0 = regularized_inverse_sqrt(
            self.c00,
            epsilon=self.epsilon,
            eigenvalue_cutoff=self.eigenvalue_cutoff,
        )
        self.whitening_t = regularized_inverse_sqrt(
            self.c11,
            epsilon=self.epsilon,
            eigenvalue_cutoff=self.eigenvalue_cutoff,
        )

        self.half_weighted_koopman = np.linalg.multi_dot(
            [self.whitening_0.inverse_sqrt, self.c01, self.whitening_t.inverse_sqrt]
        )
        u, s, vt = np.linalg.svd(self.half_weighted_koopman, full_matrices=False)
        if self.dim is not None:
            keep = min(int(self.dim), int(s.shape[0]))
            u = u[:, :keep]
            s = s[:keep]
            vt = vt[:keep]
        self.left_singular_vectors = np.asarray(u, dtype=np.float64)
        self.right_singular_vectors = np.asarray(vt.T, dtype=np.float64)
        self.singular_values = np.asarray(s, dtype=np.float64)
        self.left_coefficients = self.whitening_0.inverse_sqrt @ self.left_singular_vectors
        self.right_coefficients = self.whitening_t.inverse_sqrt @ self.right_singular_vectors

        c00_inv = self.whitening_0.inverse_sqrt @ self.whitening_0.inverse_sqrt
        self.koopman_matrix = c00_inv @ self.c01
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ManualVAMP must be fitted before use.")

    def _resolve_dim(self, dim: int | None) -> int:
        self._check_fitted()
        if dim is None:
            return int(self.model_dim)
        resolved = int(dim)
        if resolved <= 0:
            raise ValueError(f"dim must be > 0, got {resolved}.")
        return min(resolved, int(self.model_dim))

    def _resolve_scaling(self, scaling: str | None) -> str | None:
        if scaling is None:
            scaling = self.scaling
        if scaling in {None, ""}:
            return None
        scaling_name = str(scaling).strip().lower()
        if scaling_name != "kinetic_map":
            raise ValueError(
                "scaling must be None or 'kinetic_map', "
                f"got {scaling!r}."
            )
        return scaling_name

    def score(
        self,
        *,
        score: str = "VAMP2",
        covariances: CovarianceEstimate | None = None,
        dim: int | None = None,
    ) -> float:
        self._check_fitted()
        score_name = _canonical_score_name(score)
        resolved_dim = self._resolve_dim(dim)
        left = self.left_coefficients[:, :resolved_dim]
        right = self.right_coefficients[:, :resolved_dim]
        singular_values = self.singular_values[:resolved_dim]

        if covariances is None:
            if self.mean_0 is None or self.mean_t is None or self.c00 is None or self.c01 is None or self.c11 is None:
                raise RuntimeError("ManualVAMP internal covariance state is incomplete.")
            covariances = CovarianceEstimate(
                mean_0=self.mean_0,
                mean_t=self.mean_t,
                c00=self.c00,
                c01=self.c01,
                c11=self.c11,
                sample_count=int(self.sample_count),
            )

        if score_name == "VAMPE":
            sigma_diag = np.diag(singular_values)
            first_term = 2.0 * np.trace(sigma_diag @ left.T @ covariances.c01 @ right)
            second_term = np.trace(
                sigma_diag
                @ left.T
                @ covariances.c00
                @ left
                @ sigma_diag
                @ right.T
                @ covariances.c11
                @ right
            )
            return float(np.real_if_close(first_term - second_term))

        a = _small_spd_inverse_sqrt(
            left.T @ covariances.c00 @ left,
            epsilon=self.epsilon,
        )
        b = left.T @ covariances.c01 @ right
        c = _small_spd_inverse_sqrt(
            right.T @ covariances.c11 @ right,
            epsilon=self.epsilon,
        )
        abc = a @ b @ c
        return float(np.linalg.norm(abc, ord="fro") ** 2)

    def transform_instantaneous(
        self,
        data: np.ndarray,
        *,
        dim: int | None = None,
        scaling: str | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        centered = np.asarray(data, dtype=np.float64) - self.mean_0[None, :]
        resolved_dim = self._resolve_dim(dim)
        projected = centered @ self.left_coefficients[:, :resolved_dim]
        if self._resolve_scaling(scaling) == "kinetic_map":
            projected = projected * self.singular_values[:resolved_dim][None, :]
        return np.asarray(projected, dtype=np.float64)

    def transform_timelagged(
        self,
        data: np.ndarray,
        *,
        dim: int | None = None,
        scaling: str | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        centered = np.asarray(data, dtype=np.float64) - self.mean_t[None, :]
        resolved_dim = self._resolve_dim(dim)
        projected = centered @ self.right_coefficients[:, :resolved_dim]
        if self._resolve_scaling(scaling) == "kinetic_map":
            projected = projected * self.singular_values[:resolved_dim][None, :]
        return np.asarray(projected, dtype=np.float64)

    def implied_timescales(self, *, lagtime: int | None = None) -> dict[str, np.ndarray]:
        self._check_fitted()
        if lagtime is None:
            if self.lagtime is None:
                raise ValueError(
                    "lagtime must be provided when the model was created without a lagtime."
                )
            lagtime = int(self.lagtime)
        lag = float(lagtime)
        eigenvalues = np.linalg.eigvals(self.koopman_matrix)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = np.asarray(eigenvalues[order], dtype=np.complex128)
        abs_lambda = np.abs(eigenvalues)
        timescales = np.full(abs_lambda.shape, np.nan, dtype=np.float64)
        valid = (abs_lambda > 0.0) & (abs_lambda < 1.0)
        timescales[valid] = -lag / np.log(abs_lambda[valid])
        return {
            "eigenvalues": eigenvalues,
            "timescales": timescales,
        }

    def to_metadata(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "lagtime": None if self.lagtime is None else int(self.lagtime),
            "epsilon": float(self.epsilon),
            "eigenvalue_cutoff": (
                None if self.eigenvalue_cutoff is None else float(self.eigenvalue_cutoff)
            ),
            "scaling": None if self.scaling in {None, ""} else str(self.scaling),
            "dim": None if self.dim is None else int(self.dim),
            "sample_count": int(self.sample_count),
            "feature_dim": int(self.feature_dim),
            "model_dim": int(self.model_dim),
            "rank_0": int(self.whitening_0.rank),
            "rank_t": int(self.whitening_t.rank),
        }

    def save(self, path: str | Path) -> Path:
        self._check_fitted()
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved,
            mean_0=self.mean_0,
            mean_t=self.mean_t,
            c00=self.c00,
            c01=self.c01,
            c11=self.c11,
            singular_values=self.singular_values,
            left_singular_vectors=self.left_singular_vectors,
            right_singular_vectors=self.right_singular_vectors,
            left_coefficients=self.left_coefficients,
            right_coefficients=self.right_coefficients,
            half_weighted_koopman=self.half_weighted_koopman,
            koopman_matrix=self.koopman_matrix,
            whitening_0_inverse_sqrt=self.whitening_0.inverse_sqrt,
            whitening_t_inverse_sqrt=self.whitening_t.inverse_sqrt,
            whitening_0_eigenvalues=self.whitening_0.eigenvalues,
            whitening_t_eigenvalues=self.whitening_t.eigenvalues,
            whitening_0_retained_mask=self.whitening_0.retained_mask.astype(np.int8),
            whitening_t_retained_mask=self.whitening_t.retained_mask.astype(np.int8),
        )
        meta = self.to_metadata()
        with resolved.with_suffix(resolved.suffix + ".meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "ManualVAMP":
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Saved ManualVAMP model does not exist: {resolved}")
        meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Saved ManualVAMP metadata does not exist: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(resolved) as payload:
            model = cls(
                lagtime=metadata.get("lagtime", None),
                epsilon=float(metadata["epsilon"]),
                eigenvalue_cutoff=metadata.get("eigenvalue_cutoff", None),
                scaling=metadata.get("scaling", None),
                dim=metadata.get("dim", None),
            )
            model.mean_0 = np.asarray(payload["mean_0"], dtype=np.float64)
            model.mean_t = np.asarray(payload["mean_t"], dtype=np.float64)
            model.c00 = np.asarray(payload["c00"], dtype=np.float64)
            model.c01 = np.asarray(payload["c01"], dtype=np.float64)
            model.c11 = np.asarray(payload["c11"], dtype=np.float64)
            model.sample_count = int(metadata["sample_count"])

            whitening_0_retained_mask = np.asarray(
                payload["whitening_0_retained_mask"], dtype=np.int8
            ).astype(bool, copy=False)
            whitening_t_retained_mask = np.asarray(
                payload["whitening_t_retained_mask"], dtype=np.int8
            ).astype(bool, copy=False)
            model.whitening_0 = WhiteningModel(
                inverse_sqrt=np.asarray(payload["whitening_0_inverse_sqrt"], dtype=np.float64),
                eigenvalues=np.asarray(payload["whitening_0_eigenvalues"], dtype=np.float64),
                retained_eigenvalues=np.asarray(
                    payload["whitening_0_eigenvalues"], dtype=np.float64
                )[whitening_0_retained_mask],
                eigenvectors=np.empty((model.feature_dim, model.feature_dim), dtype=np.float64),
                retained_mask=whitening_0_retained_mask,
                rank=int(metadata["rank_0"]),
                epsilon=float(metadata["epsilon"]),
                cutoff=float(
                    metadata["epsilon"]
                    if metadata.get("eigenvalue_cutoff", None) is None
                    else metadata["eigenvalue_cutoff"]
                ),
            )
            model.whitening_t = WhiteningModel(
                inverse_sqrt=np.asarray(payload["whitening_t_inverse_sqrt"], dtype=np.float64),
                eigenvalues=np.asarray(payload["whitening_t_eigenvalues"], dtype=np.float64),
                retained_eigenvalues=np.asarray(
                    payload["whitening_t_eigenvalues"], dtype=np.float64
                )[whitening_t_retained_mask],
                eigenvectors=np.empty((model.feature_dim, model.feature_dim), dtype=np.float64),
                retained_mask=whitening_t_retained_mask,
                rank=int(metadata["rank_t"]),
                epsilon=float(metadata["epsilon"]),
                cutoff=float(
                    metadata["epsilon"]
                    if metadata.get("eigenvalue_cutoff", None) is None
                    else metadata["eigenvalue_cutoff"]
                ),
            )
            model.singular_values = np.asarray(payload["singular_values"], dtype=np.float64)
            model.left_singular_vectors = np.asarray(
                payload["left_singular_vectors"], dtype=np.float64
            )
            model.right_singular_vectors = np.asarray(
                payload["right_singular_vectors"], dtype=np.float64
            )
            model.left_coefficients = np.asarray(
                payload["left_coefficients"], dtype=np.float64
            )
            model.right_coefficients = np.asarray(
                payload["right_coefficients"], dtype=np.float64
            )
            model.half_weighted_koopman = np.asarray(
                payload["half_weighted_koopman"], dtype=np.float64
            )
            model.koopman_matrix = np.asarray(payload["koopman_matrix"], dtype=np.float64)
        model._fitted = True
        return model
