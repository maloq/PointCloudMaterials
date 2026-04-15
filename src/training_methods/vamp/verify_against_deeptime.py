from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import orthogonal_procrustes

from src.training_methods.vamp.common import (
    TrajectoryEmbeddings,
    build_frame_splits,
    build_lagged_pairs,
    log_progress,
    resolve_frame_window,
)
from src.training_methods.vamp.config import load_vamp_config, resolve_path
from src.training_methods.vamp.vamp import ManualVAMP, estimate_covariances


def _import_deeptime():
    try:
        from deeptime.covariance import Covariance
        from deeptime.decomposition import VAMP
    except ImportError as exc:
        raise ModuleNotFoundError(
            "deeptime is required for VAMP verification. "
            "Install it in the active environment and rerun."
        ) from exc
    return Covariance, VAMP


def _aligned_deeptime_score(model: Any, *, score: str, test_model: Any | None = None) -> float:
    if test_model is None:
        value = float(model.score(score))
    else:
        value = float(model.score(score, test_model))
    if bool(getattr(model.cov, "data_mean_removed", False)):
        value -= 1.0
    return value


def _orthogonal_alignment_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(
            "reference and candidate transforms must share the same shape, "
            f"got reference={tuple(ref.shape)}, candidate={tuple(cand.shape)}."
        )
    if ref.ndim != 2:
        raise ValueError(
            f"Expected transform matrices with shape (samples, dim), got {tuple(ref.shape)}."
        )
    if ref.shape[1] == 0:
        return 0.0
    ref_centered = ref - ref.mean(axis=0, keepdims=True)
    cand_centered = cand - cand.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(cand_centered, ref_centered)
    aligned = cand_centered @ rotation
    denom = np.linalg.norm(ref_centered)
    if denom <= 0.0:
        return float(np.linalg.norm(aligned))
    return float(np.linalg.norm(ref_centered - aligned) / denom)


def compare_manual_model_against_deeptime(
    *,
    train_x0: np.ndarray,
    train_x1: np.ndarray,
    manual_model: ManualVAMP,
    val_x0: np.ndarray | None = None,
    val_x1: np.ndarray | None = None,
    scaling: str | None = None,
) -> dict[str, Any]:
    _, VAMP = _import_deeptime()
    deeptime_model = VAMP(
        dim=int(manual_model.model_dim),
        epsilon=float(manual_model.epsilon),
        scaling=scaling,
    ).fit((train_x0, train_x1)).fetch_model()

    compare_dim = min(int(manual_model.model_dim), int(deeptime_model.dim))
    manual_singular = np.asarray(manual_model.singular_values[:compare_dim], dtype=np.float64)
    deeptime_singular = np.asarray(deeptime_model.singular_values[:compare_dim], dtype=np.float64)

    manual_train_vamp2 = float(manual_model.score(score="VAMP2", dim=compare_dim))
    manual_train_vampe = float(manual_model.score(score="VAMPE", dim=compare_dim))
    deeptime_train_vamp2 = _aligned_deeptime_score(deeptime_model, score="VAMP2")
    deeptime_train_vampe = _aligned_deeptime_score(deeptime_model, score="VAMPE")

    manual_transform = manual_model.transform_instantaneous(
        train_x0,
        dim=compare_dim,
        scaling=scaling,
    )
    deeptime_transform = np.asarray(deeptime_model.transform(train_x0), dtype=np.float64)
    left_alignment_error = _orthogonal_alignment_error(manual_transform, deeptime_transform)

    timelagged_obs = getattr(deeptime_model, "timelagged_obs", None)
    if timelagged_obs is None or not callable(timelagged_obs):
        raise AttributeError(
            "deeptime CovarianceKoopmanModel is missing a callable timelagged_obs, "
            "so right singular-function verification cannot proceed."
        )
    manual_transform_t = manual_model.transform_timelagged(
        train_x1,
        dim=compare_dim,
        scaling=scaling,
    )
    deeptime_transform_t = np.asarray(timelagged_obs(train_x1), dtype=np.float64)
    right_alignment_error = _orthogonal_alignment_error(
        manual_transform_t,
        deeptime_transform_t,
    )

    report: dict[str, Any] = {
        "compare_dim": int(compare_dim),
        "manual_singular_values": manual_singular.tolist(),
        "deeptime_singular_values": deeptime_singular.tolist(),
        "max_abs_singular_value_diff": float(np.max(np.abs(manual_singular - deeptime_singular))),
        "train_vamp2_manual": manual_train_vamp2,
        "train_vamp2_deeptime": deeptime_train_vamp2,
        "train_vamp2_abs_diff": float(abs(manual_train_vamp2 - deeptime_train_vamp2)),
        "train_vampe_manual": manual_train_vampe,
        "train_vampe_deeptime": deeptime_train_vampe,
        "train_vampe_abs_diff": float(abs(manual_train_vampe - deeptime_train_vampe)),
        "left_transform_alignment_error": left_alignment_error,
        "right_transform_alignment_error": right_alignment_error,
    }

    if val_x0 is not None and val_x1 is not None:
        if val_x0.shape != val_x1.shape:
            raise ValueError(
                "Validation instantaneous and lagged matrices must have the same shape, "
                f"got val_x0.shape={tuple(val_x0.shape)}, val_x1.shape={tuple(val_x1.shape)}."
            )
        val_deeptime_model = VAMP(
            dim=int(compare_dim),
            epsilon=float(manual_model.epsilon),
            scaling=scaling,
        ).fit((val_x0, val_x1)).fetch_model()
        cov_val_manual = estimate_covariances(val_x0, val_x1)
        manual_val_vamp2 = float(
            manual_model.score(score="VAMP2", dim=compare_dim, covariances=cov_val_manual)
        )
        manual_val_vampe = float(
            manual_model.score(score="VAMPE", dim=compare_dim, covariances=cov_val_manual)
        )
        deeptime_val_vamp2 = _aligned_deeptime_score(
            deeptime_model,
            score="VAMP2",
            test_model=val_deeptime_model,
        )
        deeptime_val_vampe = _aligned_deeptime_score(
            deeptime_model,
            score="VAMPE",
            test_model=val_deeptime_model,
        )
        report.update(
            {
                "val_vamp2_manual": manual_val_vamp2,
                "val_vamp2_deeptime": deeptime_val_vamp2,
                "val_vamp2_abs_diff": float(abs(manual_val_vamp2 - deeptime_val_vamp2)),
                "val_vampe_manual": manual_val_vampe,
                "val_vampe_deeptime": deeptime_val_vampe,
                "val_vampe_abs_diff": float(abs(manual_val_vampe - deeptime_val_vampe)),
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the manual VAMP implementation against deeptime using a VAMP config."
    )
    parser.add_argument("config", help="Config name inside configs/vamp/ or a YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path, base_dir = load_vamp_config(args.config)
    verify_cfg = cfg.verify
    embeddings_path = resolve_path(verify_cfg.embeddings_path, base_dir=base_dir)
    if embeddings_path is None:
        raise ValueError("verify.embeddings_path must be set in the VAMP config.")
    lag = int(verify_cfg.lag)
    scaling = str(getattr(verify_cfg, "scaling", "kinetic_map"))
    output_dir_text = resolve_path(getattr(verify_cfg, "output_dir", None), base_dir=base_dir)
    output_text = resolve_path(getattr(verify_cfg, "output", None), base_dir=base_dir)
    start = time.perf_counter()
    log_progress(
        "verify_against_deeptime",
        f"loading embeddings from {Path(embeddings_path).expanduser().resolve()} using config={config_path.name}",
    )
    embeddings = TrajectoryEmbeddings.load(embeddings_path)
    window = resolve_frame_window(
        embeddings.frame_count,
        window=str(getattr(verify_cfg, "window", "full")),
        frame_start=getattr(verify_cfg, "frame_start", None),
        frame_stop=getattr(verify_cfg, "frame_stop", None),
    )
    splits = build_frame_splits(
        window,
        train_fraction=float(getattr(verify_cfg, "train_fraction", 0.6)),
        val_fraction=float(getattr(verify_cfg, "val_fraction", 0.2)),
    )
    log_progress(
        "verify_against_deeptime",
        f"building lagged pairs for tau={lag} with window={window.to_dict()}",
    )
    pair_blocks = build_lagged_pairs(embeddings, lag=lag, splits=splits)
    train_pairs = pair_blocks["train"]
    val_pairs = pair_blocks["val"]
    log_progress(
        "verify_against_deeptime",
        (
            f"pair counts: train={train_pairs.pair_count}, val={val_pairs.pair_count}; "
            f"fitting manual reference model"
        ),
    )

    manual_model = ManualVAMP(
        lagtime=lag,
        epsilon=float(getattr(verify_cfg, "epsilon", 1.0e-6)),
        eigenvalue_cutoff=getattr(verify_cfg, "eigenvalue_cutoff", None),
        scaling=None if scaling == "none" else scaling,
        dim=getattr(verify_cfg, "dim", None),
    ).fit(train_pairs.x0, train_pairs.x1)

    log_progress("verify_against_deeptime", "comparing manual model against deeptime")
    report = compare_manual_model_against_deeptime(
        train_x0=train_pairs.x0,
        train_x1=train_pairs.x1,
        val_x0=val_pairs.x0,
        val_x1=val_pairs.x1,
        manual_model=manual_model,
        scaling=None if scaling == "none" else scaling,
    )
    report.update(
        {
            "config_path": str(config_path),
            "embedding_path": str(Path(embeddings_path).expanduser().resolve()),
            "lag": lag,
            "window": window.to_dict(),
            "splits": {name: split.to_dict() for name, split in splits.items()},
        }
    )

    if output_dir_text is not None:
        verification_root = Path(output_dir_text).expanduser().resolve()
        verification_root.mkdir(parents=True, exist_ok=True)
        artifacts_dir = verification_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / f"verify_tau{lag:03d}.json"
    elif output_text is not None:
        output_path = Path(output_text).expanduser().resolve()
    else:
        embedding_path = Path(embeddings_path).expanduser().resolve()
        if embedding_path.parent.name == "artifacts" and embedding_path.parent.parent.name == "embeddings":
            verification_root = embedding_path.parent.parent.parent / "verification"
            verification_root.mkdir(parents=True, exist_ok=True)
            artifacts_dir = verification_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            output_path = artifacts_dir / f"verify_tau{lag:03d}.json"
        else:
            output_path = embedding_path.with_name(f"verify_deeptime_tau{lag:03d}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    elapsed = time.perf_counter() - start
    log_progress("verify_against_deeptime", f"wrote {output_path} in {elapsed:.1f}s")

    if getattr(verify_cfg, "max_score_diff", None) is not None:
        threshold = float(verify_cfg.max_score_diff)
        score_diffs = [
            float(report["train_vamp2_abs_diff"]),
            float(report["train_vampe_abs_diff"]),
        ]
        if "val_vamp2_abs_diff" in report:
            score_diffs.append(float(report["val_vamp2_abs_diff"]))
        if "val_vampe_abs_diff" in report:
            score_diffs.append(float(report["val_vampe_abs_diff"]))
        if max(score_diffs) > threshold:
            raise SystemExit(
                f"deeptime score comparison failed: max_diff={max(score_diffs)} > {threshold}"
            )

    if getattr(verify_cfg, "max_transform_error", None) is not None:
        threshold = float(verify_cfg.max_transform_error)
        transform_errors = [
            float(report["left_transform_alignment_error"]),
            float(report["right_transform_alignment_error"]),
        ]
        if max(transform_errors) > threshold:
            raise SystemExit(
                "deeptime transform comparison failed: "
                f"max_error={max(transform_errors)} > {threshold}"
            )


if __name__ == "__main__":
    main()
