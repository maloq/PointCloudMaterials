from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.training_methods.vamp.common import (
    TrajectoryEmbeddings,
    build_frame_splits,
    build_lagged_pairs,
    ensure_dir,
    log_progress,
    parse_int_list,
    resolve_frame_window,
    save_json,
)
from src.training_methods.vamp.config import load_vamp_config, resolve_path
from src.training_methods.vamp.vamp import ManualVAMP, estimate_covariances
from src.training_methods.vamp.verify_against_deeptime import compare_manual_model_against_deeptime


def _default_output_dir(embedding_path: str | Path) -> Path:
    resolved = Path(embedding_path).expanduser().resolve()
    if resolved.parent.name == "artifacts" and resolved.parent.parent.name == "embeddings":
        return resolved.parent.parent.parent / "fit"
    return resolved.parent / "vamp_fit"


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows were provided for CSV output: {resolved}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return resolved


def _plot_score_diagnostics(rows: list[dict[str, Any]], out_path: str | Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise ValueError("Cannot plot lag diagnostics because no lag was fitted successfully.")
    lags = np.asarray([int(row["lag"]) for row in ok_rows], dtype=np.int64)
    train_vamp2 = np.asarray([float(row["train_vamp2"]) for row in ok_rows], dtype=np.float64)
    val_vamp2 = np.asarray([float(row["val_vamp2"]) for row in ok_rows], dtype=np.float64)
    val_vampe = np.asarray([float(row["val_vampe"]) for row in ok_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(lags, train_vamp2, marker="o", label="train VAMP-2")
    ax.plot(lags, val_vamp2, marker="o", label="val VAMP-2")
    ax.plot(lags, val_vampe, marker="o", label="val VAMP-E")
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("score")
    ax.set_title("Lag Diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_singular_values(rows: list[dict[str, Any]], out_path: str | Path, max_modes: int) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise ValueError("Cannot plot singular values because no lag was fitted successfully.")
    lags = np.asarray([int(row["lag"]) for row in ok_rows], dtype=np.int64)
    singular_values = np.full((len(ok_rows), max_modes), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(ok_rows):
        values = json.loads(str(row["singular_values"]))
        take = min(max_modes, len(values))
        singular_values[row_idx, :take] = np.asarray(values[:take], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    for mode_idx in range(singular_values.shape[1]):
        if not np.any(np.isfinite(singular_values[:, mode_idx])):
            continue
        ax.plot(
            lags,
            singular_values[:, mode_idx],
            marker="o",
            label=f"mode {mode_idx + 1}",
        )
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("singular value")
    ax.set_title("Leading Singular Values")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _plot_implied_timescales(rows: list[dict[str, Any]], out_path: str | Path, max_modes: int) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise ValueError("Cannot plot implied timescales because no lag was fitted successfully.")
    lags = np.asarray([int(row["lag"]) for row in ok_rows], dtype=np.int64)
    timescales = np.full((len(ok_rows), max_modes), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(ok_rows):
        values = json.loads(str(row["leading_timescales"]))
        take = min(max_modes, len(values))
        timescales[row_idx, :take] = np.asarray(values[:take], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    for mode_idx in range(timescales.shape[1]):
        if not np.any(np.isfinite(timescales[:, mode_idx])):
            continue
        ax.plot(
            lags,
            timescales[:, mode_idx],
            marker="o",
            label=f"timescale {mode_idx + 1}",
        )
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("implied timescale (frames)")
    ax.set_title("Implied Timescales")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def _run_ck_test(models_by_lag: dict[int, ManualVAMP], preferred_base_lag: int | None) -> dict[str, Any]:
    if len(models_by_lag) < 2:
        return {
            "status": "not_available",
            "reason": "Need at least two fitted lags to run a Chapman-Kolmogorov-style comparison.",
        }

    available_lags = sorted(int(lag) for lag in models_by_lag)
    candidate_base_lags = [preferred_base_lag] if preferred_base_lag in models_by_lag else []
    candidate_base_lags.extend(
        lag for lag in available_lags if lag not in candidate_base_lags
    )

    base_lag = None
    for lag in candidate_base_lags:
        if any(other > lag and other % lag == 0 for other in available_lags):
            base_lag = int(lag)
            break
    if base_lag is None:
        return {
            "status": "not_available",
            "reason": (
                "No lag had an integer multiple among the fitted lag grid, "
                f"available_lags={available_lags}."
            ),
        }

    base_model = models_by_lag[base_lag]
    comparisons: list[dict[str, Any]] = []
    for lag in available_lags:
        if lag == base_lag or lag % base_lag != 0:
            continue
        step_multiple = lag // base_lag
        predicted = np.linalg.matrix_power(base_model.koopman_matrix, step_multiple)
        observed = models_by_lag[lag].koopman_matrix
        denom = max(float(np.linalg.norm(observed, ord="fro")), 1.0e-12)
        relative_error = float(np.linalg.norm(observed - predicted, ord="fro") / denom)
        comparisons.append(
            {
                "base_lag": int(base_lag),
                "target_lag": int(lag),
                "step_multiple": int(step_multiple),
                "relative_frobenius_error": relative_error,
            }
        )

    if not comparisons:
        return {
            "status": "not_available",
            "reason": (
                "No integer-multiple lag pairs were available after selecting the base lag, "
                f"base_lag={base_lag}, available_lags={available_lags}."
            ),
        }
    return {
        "status": "ok",
        "base_lag": int(base_lag),
        "comparisons": comparisons,
    }


def _plot_ck_test(ck_summary: dict[str, Any], out_path: str | Path) -> None:
    if ck_summary.get("status") != "ok":
        return
    comparisons = ck_summary["comparisons"]
    x = np.asarray([int(entry["target_lag"]) for entry in comparisons], dtype=np.int64)
    y = np.asarray([float(entry["relative_frobenius_error"]) for entry in comparisons], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=180)
    ax.plot(x, y, marker="o")
    ax.set_xlabel("target lag (frames)")
    ax.set_ylabel("relative CK error")
    ax.set_title(f"CK Test vs base lag {int(ck_summary['base_lag'])}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path).expanduser().resolve(), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a manual VAMP model on frozen local-structure embeddings using a VAMP config."
    )
    parser.add_argument("config", help="Config name inside configs/vamp/ or a YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path, base_dir = load_vamp_config(args.config)
    fit_cfg = cfg.fit
    embeddings_path = resolve_path(fit_cfg.embeddings_path, base_dir=base_dir)
    if embeddings_path is None:
        raise ValueError("fit.embeddings_path must be set in the VAMP config.")
    output_dir_text = resolve_path(getattr(fit_cfg, "output_dir", None), base_dir=base_dir)
    output_dir = ensure_dir(
        _default_output_dir(embeddings_path) if output_dir_text is None else output_dir_text
    )
    figures_dir = ensure_dir(output_dir / "figures")
    artifacts_dir = ensure_dir(output_dir / "artifacts")
    models_dir = ensure_dir(output_dir / "models")
    epsilon = float(getattr(fit_cfg, "epsilon", 1.0e-6))
    eigenvalue_cutoff = getattr(fit_cfg, "eigenvalue_cutoff", None)
    scaling = str(getattr(fit_cfg, "scaling", "kinetic_map"))
    verify_deeptime = bool(getattr(fit_cfg, "verify_deeptime", False))
    projection_dim_requested = int(getattr(fit_cfg, "projection_dim", 6))
    dim = getattr(fit_cfg, "dim", None)
    overall_start = time.perf_counter()
    log_progress(
        "fit_vamp",
        f"loading embeddings from {Path(embeddings_path).expanduser().resolve()} using config={config_path.name}",
    )
    embeddings = TrajectoryEmbeddings.load(embeddings_path)
    log_progress(
        "fit_vamp",
        (
            f"embedding artifact ready: frames={embeddings.frame_count}, atoms={embeddings.num_atoms}, "
            f"latent_dim={embeddings.latent_dim}, output_dir={output_dir}"
        ),
    )
    window = resolve_frame_window(
        embeddings.frame_count,
        window=str(getattr(fit_cfg, "window", "full")),
        frame_start=getattr(fit_cfg, "frame_start", None),
        frame_stop=getattr(fit_cfg, "frame_stop", None),
    )
    splits = build_frame_splits(
        window,
        train_fraction=float(getattr(fit_cfg, "train_fraction", 0.6)),
        val_fraction=float(getattr(fit_cfg, "val_fraction", 0.2)),
    )
    lag_values = parse_int_list(getattr(fit_cfg, "lags", "1,2,4,8,12,16,24,32"))
    split_summary = {name: split.to_dict() for name, split in splits.items()}
    log_progress(
        "fit_vamp",
        (
            f"window={window.to_dict()} "
            f"splits={split_summary} "
            f"lags={lag_values}"
        ),
    )

    rows: list[dict[str, Any]] = []
    models_by_lag: dict[int, ManualVAMP] = {}
    deeptime_reports: dict[str, Any] = {}

    for lag_idx, lag in enumerate(lag_values, start=1):
        lag = int(lag)
        lag_start = time.perf_counter()
        log_progress("fit_vamp", f"fitting lag {lag_idx}/{len(lag_values)}: tau={lag}")
        lag_row: dict[str, Any] = {
            "lag": int(lag),
            "status": "ok",
            "window": json.dumps(window.to_dict()),
        }
        try:
            pair_blocks = build_lagged_pairs(embeddings, lag=lag, splits=splits)
            train_pairs = pair_blocks["train"]
            val_pairs = pair_blocks["val"]
            test_pairs = pair_blocks["test"]
            log_progress(
                "fit_vamp",
                (
                    f"tau={lag}: train_pairs={train_pairs.pair_count}, "
                    f"val_pairs={val_pairs.pair_count}, test_pairs={test_pairs.pair_count}"
                ),
            )

            cov_train = estimate_covariances(train_pairs.x0, train_pairs.x1)
            cov_val = estimate_covariances(val_pairs.x0, val_pairs.x1)
            cov_test = estimate_covariances(test_pairs.x0, test_pairs.x1)

            model = ManualVAMP(
                lagtime=lag,
                epsilon=epsilon,
                eigenvalue_cutoff=eigenvalue_cutoff,
                scaling=None if scaling == "none" else scaling,
                dim=dim,
            ).fit_from_covariances(cov_train)
            models_by_lag[int(lag)] = model

            train_vamp2 = model.score(score="VAMP2", covariances=cov_train)
            val_vamp2 = model.score(score="VAMP2", covariances=cov_val)
            val_vampe = model.score(score="VAMPE", covariances=cov_val)
            test_vamp2 = model.score(score="VAMP2", covariances=cov_test)
            test_vampe = model.score(score="VAMPE", covariances=cov_test)

            eigenanalysis = model.implied_timescales(lagtime=lag)
            leading_timescales = np.asarray(eigenanalysis["timescales"], dtype=np.float64)
            finite_timescales = leading_timescales[np.isfinite(leading_timescales)]
            leading_timescales_list = (
                finite_timescales[: min(6, finite_timescales.size)].tolist()
                if finite_timescales.size > 0
                else []
            )

            lag_model_path = models_dir / f"lag_{lag:03d}_model.npz"
            model.save(lag_model_path)

            lag_row.update(
                {
                    "train_pair_count": int(train_pairs.pair_count),
                    "val_pair_count": int(val_pairs.pair_count),
                    "test_pair_count": int(test_pairs.pair_count),
                    "rank_0": int(model.whitening_0.rank),
                    "rank_t": int(model.whitening_t.rank),
                    "train_vamp2": float(train_vamp2),
                    "val_vamp2": float(val_vamp2),
                    "val_vampe": float(val_vampe),
                    "test_vamp2": float(test_vamp2),
                    "test_vampe": float(test_vampe),
                    "singular_values": json.dumps(
                        [float(v) for v in np.asarray(model.singular_values, dtype=np.float64).tolist()]
                    ),
                    "leading_timescales": json.dumps(
                        [float(v) for v in leading_timescales_list]
                    ),
                    "model_path": str(lag_model_path),
                }
            )

            if verify_deeptime:
                log_progress("fit_vamp", f"tau={lag}: comparing manual model against deeptime")
                try:
                    deeptime_report = compare_manual_model_against_deeptime(
                        train_x0=train_pairs.x0,
                        train_x1=train_pairs.x1,
                        val_x0=val_pairs.x0,
                        val_x1=val_pairs.x1,
                        manual_model=model,
                        scaling=None if scaling == "none" else scaling,
                    )
                    deeptime_reports[str(lag)] = deeptime_report
                    lag_row.update(
                        {
                            "deeptime_status": "ok",
                            "deeptime_max_abs_singular_value_diff": float(
                                deeptime_report["max_abs_singular_value_diff"]
                            ),
                            "deeptime_train_vamp2_abs_diff": float(
                                deeptime_report["train_vamp2_abs_diff"]
                            ),
                            "deeptime_val_vamp2_abs_diff": float(
                                deeptime_report["val_vamp2_abs_diff"]
                            ),
                            "deeptime_left_transform_alignment_error": float(
                                deeptime_report["left_transform_alignment_error"]
                            ),
                            "deeptime_right_transform_alignment_error": float(
                                deeptime_report["right_transform_alignment_error"]
                            ),
                        }
                    )
                except Exception as exc:
                    lag_row.update(
                        {
                            "deeptime_status": "failed",
                            "deeptime_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            lag_elapsed = time.perf_counter() - lag_start
            log_progress(
                "fit_vamp",
                (
                    f"tau={lag} complete in {lag_elapsed:.1f}s: "
                    f"val_vampe={float(lag_row['val_vampe']):.6f}, "
                    f"train_vamp2={float(lag_row['train_vamp2']):.6f}, "
                    f"status={lag_row['status']}"
                ),
            )
        except Exception as exc:
            lag_row.update(
                {
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            lag_elapsed = time.perf_counter() - lag_start
            log_progress(
                "fit_vamp",
                f"tau={lag} skipped after {lag_elapsed:.1f}s: {lag_row['reason']}",
            )
        rows.append(lag_row)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError(
            "No lag in the requested grid could be fitted successfully. "
            f"lags={lag_values}, window={window.to_dict()}."
        )

    selected_row = max(ok_rows, key=lambda row: float(row["val_vampe"]))
    selected_lag = int(selected_row["lag"])
    selected_model = models_by_lag[selected_lag]
    projection_dim = min(projection_dim_requested, int(selected_model.model_dim))
    if projection_dim <= 0:
        raise ValueError(
            f"projection_dim must resolve to > 0, got {projection_dim}."
        )

    flat_embeddings = embeddings.invariant_embeddings.reshape(-1, embeddings.latent_dim)
    projections = selected_model.transform_instantaneous(
        flat_embeddings,
        dim=projection_dim,
        scaling=None if scaling == "none" else scaling,
    ).astype(np.float32, copy=False)
    projection_grid = projections.reshape(
        embeddings.frame_count,
        embeddings.num_atoms,
        projection_dim,
    )
    projection_path = artifacts_dir / "selected_projections.npz"
    np.savez_compressed(
        projection_path,
        instantaneous_projection=projection_grid,
        frame_indices=embeddings.frame_indices,
        timesteps=embeddings.timesteps,
        atom_ids=embeddings.atom_ids,
    )

    selected_model_path = models_dir / "selected_model.npz"
    selected_model.save(selected_model_path)

    ck_summary = _run_ck_test(models_by_lag, preferred_base_lag=selected_lag)
    if ck_summary.get("status") == "ok":
        _plot_ck_test(ck_summary, figures_dir / "ck_test.png")

    diagnostics_csv_path = _write_csv(rows, artifacts_dir / "lag_diagnostics.csv")
    diagnostics_json_path = save_json({"rows": rows}, artifacts_dir / "lag_diagnostics.json")
    deeptime_verification_path = None
    if verify_deeptime:
        deeptime_verification_path = save_json(deeptime_reports, artifacts_dir / "deeptime_verification.json")

    score_plot_path = figures_dir / "score_diagnostics.png"
    singular_values_plot_path = figures_dir / "singular_values_vs_lag.png"
    implied_timescales_plot_path = figures_dir / "implied_timescales_vs_lag.png"
    _plot_score_diagnostics(rows, score_plot_path)
    _plot_singular_values(rows, singular_values_plot_path, max_modes=min(6, projection_dim))
    _plot_implied_timescales(rows, implied_timescales_plot_path, max_modes=min(6, projection_dim))

    summary_path = artifacts_dir / "summary.json"
    summary = {
        "config_path": str(config_path),
        "embedding_path": str(Path(embeddings_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "figures_dir": str(figures_dir),
        "models_dir": str(models_dir),
        "window": window.to_dict(),
        "splits": {name: split.to_dict() for name, split in splits.items()},
        "lags_requested": [int(v) for v in lag_values],
        "lags_fitted": [int(row["lag"]) for row in ok_rows],
        "selected_lag": int(selected_lag),
        "selection_metric": "val_vampe",
        "selected_row": selected_row,
        "selected_model_path": str(selected_model_path),
        "selected_projection_path": str(projection_path),
        "projection_dim": int(projection_dim),
        "projection_scaling": None if scaling == "none" else scaling,
        "ck_test": ck_summary,
        "verify_deeptime": verify_deeptime,
        "lag_diagnostics_csv": str(diagnostics_csv_path),
        "lag_diagnostics_json": str(diagnostics_json_path),
        "deeptime_verification_path": (
            None if deeptime_verification_path is None else str(deeptime_verification_path)
        ),
        "figure_paths": {
            "score_diagnostics": str(score_plot_path),
            "singular_values_vs_lag": str(singular_values_plot_path),
            "implied_timescales_vs_lag": str(implied_timescales_plot_path),
            "ck_test": (
                None if ck_summary.get("status") != "ok" else str(figures_dir / "ck_test.png")
            ),
        },
    }
    save_json(summary, summary_path)
    total_elapsed = time.perf_counter() - overall_start
    log_progress(
        "fit_vamp",
        (
            f"selected lag={selected_lag} with val_vampe={float(selected_row['val_vampe']):.6f}. "
            f"Artifacts written to {output_dir} in {total_elapsed:.1f}s"
        ),
    )


if __name__ == "__main__":
    main()
