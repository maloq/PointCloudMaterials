#!/usr/bin/env python3
"""Random-search sweep for VN PointNet VICReg augmentation + hyperparameters.

This script launches multiple `train_contrastive.py` runs with Hydra overrides,
collects a target metric, and ranks trials.

Example:
  python scripts/sweep_vn_pointnet_vicreg.py --num-trials 24 --devices "[0,1]"

The default objective is `val/class/encoder_linear_svm_accuracy` (maximum),
parsed from training logs or recovered from checkpoint callback state.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    # Core optimizer/schedule.
    "learning_rate": [5e-4, 8e-4, 1.2e-3, 1.5e-3, 2.0e-3],
    "decay_rate": [5e-5, 1e-4, 2e-4],
    "scheduler_min_lr": [1e-6, 2e-6, 5e-6],
    "warmup_epochs": [8, 12, 16],
    # VN-PointNet architecture knobs.
    "latent_size": [96, 144],
    "encoder.kwargs.n_knn": [12, 20, 28],
    "encoder.kwargs.feature_transform": [True, False],
    "encoder.kwargs.hidden_dim1": [192, 256, 320],
    "encoder.kwargs.hidden_dim2": [768, 1024, 1280],
    # VICReg balance terms.
    "vicreg_sim_coeff": [15.0, 25.0, 35.0],
    "vicreg_std_coeff": [15.0, 25.0, 35.0],
    "vicreg_cov_coeff": [0.5, 1.0, 2.0],
    "vicreg_radial_beta1": [0.5, 1.0, 1.5],
    "vicreg_radial_beta2": [0.05, 0.1, 0.2],
    # View augmentations (the main focus).
    "vicreg_jitter_std": [0.0, 0.01, 0.02, 0.04],
    "vicreg_drop_ratio": [0.05, 0.12, 0.2, 0.3],
    "vicreg_neighbor_view_mode": ["both", "second", "first"],
    "vicreg_neighbor_k": [4, 8, 12],
    "vicreg_neighbor_max_relative_distance": [0.05, 0.1, 0.2],
    "vicreg_rotation_mode": ["full"],
    "vicreg_strain_std": [0.0, 0.02, 0.04, 0.08],
    "vicreg_occlusion_mode": ["none", "mixed", "slab", "cone"],
    "vicreg_occlusion_view": ["both", "second"],
    "vicreg_occlusion_slab_frac": [0.2, 0.3, 0.4],
    "vicreg_occlusion_cone_deg": [15.0, 25.0, 35.0],
    # Invariant head complexity.
    "vicreg_invariant_max_factor": [2.0, 3.0, 4.0],
    "vicreg_invariant_use_third_order": [True, False],
}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SAFE_OVERRIDE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")
DEFAULT_TRAINER_METRIC = "val/class/encoder_linear_svm_accuracy"
DEFAULT_EPOCHS = 30
MODELNET40_TRAINSET_SIZE = 9843
MIN_MAX_SAMPLES = (MODELNET40_TRAINSET_SIZE + 1) // 2
DEFAULT_MAX_SAMPLES = 5000
if DEFAULT_MAX_SAMPLES < MIN_MAX_SAMPLES:
    raise ValueError(
        "DEFAULT_MAX_SAMPLES must be at least half of the ModelNet40 train split: "
        f"DEFAULT_MAX_SAMPLES={DEFAULT_MAX_SAMPLES}, MIN_MAX_SAMPLES={MIN_MAX_SAMPLES}."
    )


@dataclass
class TrialResult:
    trial_id: int
    run_dir: str
    status: str
    return_code: int
    duration_sec: float
    score: float | None
    score_source: str | None
    error: str | None
    params: dict[str, Any]
    command: list[str]

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "trial_id": self.trial_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "return_code": self.return_code,
            "duration_sec": round(self.duration_sec, 3),
            "score": self.score,
            "score_source": self.score_source,
            "error": self.error or "",
        }
        for key, value in self.params.items():
            row[f"param::{key}"] = value
        return row


def _bool_to_override(value: bool) -> str:
    return "true" if value else "false"


def _override_value(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_to_override(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (int,)):
        return str(value)
    text = str(value)
    if SAFE_OVERRIDE_TOKEN_RE.fullmatch(text) is not None:
        return text
    # Quote strings that contain Hydra grammar control characters (for example '=' in ckpt names).
    return json.dumps(text)


def _force_override(key: str, value: Any) -> str:
    """Return a Hydra override that works for both existing and missing keys."""
    if not isinstance(key, str) or key.strip() == "":
        raise ValueError(f"Override key must be a non-empty string, got {key!r}")
    return f"++{key}={_override_value(value)}"


def _sample_unique_params(
    *,
    search_space: dict[str, list[Any]],
    rng: random.Random,
    num_trials: int,
) -> list[dict[str, Any]]:
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")

    for key, values in search_space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search-space entry '{key}' must be a non-empty list, got {values!r}"
            )

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(200, 20 * num_trials)

    while len(trials) < num_trials:
        if max_attempts <= 0:
            raise RuntimeError(
                "Could not sample enough unique trials from the provided search space. "
                f"Requested={num_trials}, sampled={len(trials)}."
            )
        max_attempts -= 1

        params = {key: rng.choice(values) for key, values in search_space.items()}
        _validate_sampled_params(params)
        signature = json.dumps(params, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(params)
    return trials


def _validate_sampled_params(params: dict[str, Any]) -> None:
    latent_size = int(params.get("latent_size", 96))
    if latent_size % 3 != 0:
        raise ValueError(
            f"Invalid sampled latent_size={latent_size}; VN PointNet requires divisibility by 3."
        )

    hidden_dim1 = int(params.get("encoder.kwargs.hidden_dim1", 256))
    hidden_dim2 = int(params.get("encoder.kwargs.hidden_dim2", 1024))
    if hidden_dim2 < hidden_dim1:
        raise ValueError(
            f"Invalid sampled hidden dims: hidden_dim2 ({hidden_dim2}) must be >= hidden_dim1 ({hidden_dim1})."
        )

    drop_ratio = float(params.get("vicreg_drop_ratio", 0.0))
    if not (0.0 <= drop_ratio < 1.0):
        raise ValueError(f"Invalid vicreg_drop_ratio={drop_ratio}; expected [0, 1).")

    occlusion_slab = float(params.get("vicreg_occlusion_slab_frac", 0.3))
    if not (0.0 <= occlusion_slab <= 1.0):
        raise ValueError(
            f"Invalid vicreg_occlusion_slab_frac={occlusion_slab}; expected [0, 1]."
        )


def _load_search_space(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return dict(DEFAULT_SEARCH_SPACE)
    if not path.exists():
        raise FileNotFoundError(f"Search-space file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"Search-space file must be a JSON object mapping keys->list, got {type(data)!r}."
        )
    out: dict[str, list[Any]] = {}
    for key, values in data.items():
        if not isinstance(key, str) or key.strip() == "":
            raise ValueError(f"Search-space key must be a non-empty string, got {key!r}.")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search-space entry '{key}' must be a non-empty list, got {values!r}."
            )
        out[key] = values
    return out


def _latest_checkpoint_under(root: Path) -> Path | None:
    newest: Path | None = None
    newest_mtime = float("-inf")
    for ckpt in root.rglob("*.ckpt"):
        try:
            mtime = ckpt.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = ckpt
    return newest


def _resolve_init_checkpoint(init_checkpoint_arg: str, repo_root: Path) -> Path:
    raw = str(init_checkpoint_arg).strip()
    if raw == "":
        raise ValueError("--init-checkpoint must be a non-empty path or 'auto'.")

    if raw.lower() == "auto":
        output_root = repo_root / "output"
        if not output_root.exists():
            raise FileNotFoundError(
                f"Cannot auto-resolve checkpoint: output directory does not exist: {output_root}"
            )
        latest = _latest_checkpoint_under(output_root)
        if latest is None:
            raise FileNotFoundError(
                "Cannot auto-resolve checkpoint: no '*.ckpt' files found under "
                f"{output_root}. Pass --init-checkpoint PATH explicitly."
            )
        return latest.resolve()

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"--init-checkpoint path does not exist: {path}")
    if path.suffix.lower() != ".ckpt":
        raise ValueError(
            f"--init-checkpoint must point to a .ckpt file, got: {path}"
        )
    return path


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_metric_from_text(text: str, metric_name: str) -> float | None:
    cleaned = _strip_ansi(text)
    pattern = re.compile(
        rf"{re.escape(metric_name)}(?:\s*[:=]\s*|\s+)([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(cleaned)
    if not matches:
        return None
    return float(matches[-1])


def _parse_last_canonical_acc_metric_from_text(
    text: str,
) -> tuple[float | None, str | None]:
    cleaned = _strip_ansi(text)
    pattern = re.compile(
        r"(test/class/acc_kmeans_plusplus_hungarian_canonical(?:_k\d+)?)"
        r"(?:\s*[:=]\s*|\s+)([+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(cleaned)
    if not matches:
        return None, None
    metric_name, value = matches[-1]
    return float(value), metric_name


def _read_checkpoint_monitored_score(
    *,
    run_dir: Path,
    monitor_name: str,
) -> tuple[float | None, str | None]:
    ckpt_paths = sorted(
        run_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpt_paths:
        return None, None

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Failed to import torch while reading checkpoint-monitored metric "
            f"'{monitor_name}' from {run_dir}: {exc}"
        ) from exc

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        callbacks = ckpt.get("callbacks")
        if not isinstance(callbacks, dict):
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has invalid callbacks payload: "
                f"expected dict, got {type(callbacks)!r}."
            )
        for callback_name, state in callbacks.items():
            if not isinstance(state, dict):
                continue
            monitor = str(state.get("monitor", "")).strip()
            if monitor != monitor_name:
                continue

            score = state.get("best_model_score")
            score_field = "best_model_score"
            if score is None:
                score = state.get("current_score")
                score_field = "current_score"
            if score is None:
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' callback '{callback_name}' monitors "
                    f"'{monitor_name}' but does not provide best_model_score/current_score."
                )

            if hasattr(score, "item"):
                score_value = float(score.item())
            else:
                score_value = float(score)
            source = f"{ckpt_path}:callbacks[{callback_name}].{score_field}"
            return score_value, source
    return None, None


def _read_metric_from_run_dir(
    *,
    run_dir: Path,
    objective: str,
    trainer_metric_name: str,
) -> tuple[float | None, str | None]:
    if objective == "analysis_canonical_acc":
        analysis_file = run_dir / "analysis" / "analysis_metrics.json"
        if analysis_file.exists():
            data = json.loads(analysis_file.read_text())
            value = (
                data.get("test_phase", {})
                .get("canonical", {})
                .get("accuracy")
            )
            if value is None:
                # Backward-compatibility: canonical ACC now comes from Trainer.test metrics
                # rather than post-analysis test_phase payload.
                value = None
            else:
                return float(value), f"{analysis_file}:test_phase.canonical.accuracy"
        candidates = [run_dir / "train_contrastive.log", run_dir / "sweep_driver.log"]
        for path in candidates:
            if not path.exists():
                continue
            text = path.read_text(errors="replace")
            parsed, metric_name = _parse_last_canonical_acc_metric_from_text(text)
            if parsed is not None and metric_name is not None:
                return parsed, f"{path}:{metric_name}"
        if analysis_file.exists():
            raise ValueError(
                "analysis_canonical_acc objective could not be resolved. "
                "Expected either legacy analysis metric "
                f"'{analysis_file}:test_phase.canonical.accuracy' "
                "or test log metric matching "
                "'test/class/acc_kmeans_plusplus_hungarian_canonical'."
            )
        return None, None

    # objective == trainer_test_metric
    candidates = [run_dir / "train_contrastive.log", run_dir / "sweep_driver.log"]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        value = _parse_metric_from_text(text, trainer_metric_name)
        if value is not None:
            return value, f"{path}:{trainer_metric_name}"

    # Some validation-only metrics (e.g. val/class/encoder_linear_svm_accuracy) are not guaranteed to
    # appear in progress-bar logs. Recover monitored value from checkpoint state.
    if str(trainer_metric_name).startswith("val/"):
        ckpt_value, ckpt_source = _read_checkpoint_monitored_score(
            run_dir=run_dir,
            monitor_name=trainer_metric_name,
        )
        if ckpt_value is not None:
            return ckpt_value, ckpt_source
    return None, None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def _write_csv(path: Path, records: list[TrialResult]) -> None:
    rows = [r.to_row() for r in records]
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            all_keys.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def _build_command(
    *,
    python_exe: str,
    train_script: Path,
    config_name: str,
    run_dir: Path,
    trial_id: int,
    params: dict[str, Any],
    epochs_override: int | None,
    max_samples: int,
    init_checkpoint: Path,
    devices_override: str,
    batch_size: int,
    accumulate_grad_batches: int,
    extra_overrides: list[str],
    wandb_mode: str,
    trainer_metric_name: str,
    metric_direction: str,
) -> list[str]:
    effective_batch = int(batch_size) * int(accumulate_grad_batches)
    if effective_batch < 1:
        raise ValueError(
            f"Invalid effective batch {effective_batch} from "
            f"batch_size={batch_size}, accumulate_grad_batches={accumulate_grad_batches}."
        )

    metric_name = str(trainer_metric_name).strip()
    metric_dir = str(metric_direction).strip().lower()
    if metric_dir not in {"min", "max"}:
        raise ValueError(
            f"metric_direction must be 'min' or 'max', got {metric_direction!r}."
        )

    supervised_metrics_enabled = metric_name.startswith("val/class/") or metric_name.startswith("test/class/")

    command = [
        python_exe,
        str(train_script),
        "--config-name",
        config_name,
        f"hydra.run.dir={run_dir}",
        "encoder.name=PnE_VN",
        f"experiment_name=VICREG_MODELNET40_vn_pointnet_sweep_t{trial_id:03d}",
        f"devices={devices_override}",
        f"wandb_mode={wandb_mode}",
        f"++init_from_checkpoint={_override_value(str(init_checkpoint))}",
        "++init_from_checkpoint_strict=false",
        "++run_test_after_training=false",
        "++run_post_training_analysis=false",
        f"++enable_supervised_metrics={_bool_to_override(supervised_metrics_enabled)}",
        "++check_val_every_n_epoch=1",
        "++num_sanity_val_steps=0",
        f"++max_samples={max_samples}",
        "auto_batch_size_search=false",
        "auto_batch_size_adjust_accumulate=false",
        f"batch_size={batch_size}",
        f"accumulate_grad_batches={accumulate_grad_batches}",
        f"auto_batch_size_target_effective_batch={effective_batch}",
        # Keep any residual analysis knobs lightweight.
        "analysis_hdbscan_enabled=false",
        "analysis_tsne_max_samples=2000",
    ]
    if metric_name.startswith("val/"):
        command.append(f"++checkpoint_monitor={_override_value(metric_name)}")
        command.append(f"++checkpoint_mode={metric_dir}")

    if epochs_override is not None:
        if epochs_override < 1:
            raise ValueError(f"--epochs must be >= 1, got {epochs_override}")
        command.append(f"epochs={int(epochs_override)}")

    for key in sorted(params.keys()):
        value = params[key]
        command.append(_force_override(key, value))

    for override in extra_overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --extra-override '{override}'. Expected KEY=VALUE."
            )
        command.append(override)

    return command


def _run_trial(
    *,
    command: list[str],
    repo_root: Path,
    run_dir: Path,
    trial_id: int,
    objective: str,
    trainer_metric_name: str,
    continue_on_error: bool,
    dry_run: bool,
    params: dict[str, Any],
) -> TrialResult:
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {run_dir}. "
            "Choose a different output root or remove stale trial folders."
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "sweep_driver.log"
    start = time.perf_counter()
    return_code = 0
    status = "ok"
    error: str | None = None

    if dry_run:
        status = "dry_run"
        log_path.write_text("DRY RUN\n" + " ".join(command) + "\n")
    else:
        with log_path.open("w") as log_handle:
            log_handle.write("COMMAND:\n")
            log_handle.write(" ".join(command) + "\n\n")
            log_handle.flush()
            proc = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
            return_code = int(proc.returncode)
            if return_code != 0:
                status = "failed"
                error = (
                    f"Training command exited with code {return_code}. "
                    f"See {log_path} for details."
                )
                if not continue_on_error:
                    raise RuntimeError(error)

    duration = time.perf_counter() - start
    score, source = _read_metric_from_run_dir(
        run_dir=run_dir,
        objective=objective,
        trainer_metric_name=trainer_metric_name,
    )

    if status == "ok" and score is None:
        status = "missing_metric"
        error = (
            "Run finished but target metric was not found in expected outputs "
            f"(objective={objective})."
        )

    return TrialResult(
        trial_id=trial_id,
        run_dir=str(run_dir),
        status=status,
        return_code=return_code,
        duration_sec=duration,
        score=score,
        score_source=source,
        error=error,
        params=dict(params),
        command=list(command),
    )


def _select_best(
    records: list[TrialResult],
    *,
    metric_direction: str,
) -> TrialResult | None:
    scored = [r for r in records if r.score is not None]
    if not scored:
        return None
    direction = str(metric_direction).strip().lower()
    if direction not in {"max", "min"}:
        raise ValueError(
            f"metric_direction must be 'max' or 'min', got {metric_direction!r}"
        )
    scored.sort(key=lambda r: float(r.score), reverse=(direction == "max"))
    return scored[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search VN PointNet VICReg augmentation/hyperparameters by launching "
            "multiple train_contrastive.py runs."
        )
    )
    parser.add_argument("--num-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config-name",
        type=str,
        default="vicreg_vn_modelnet40.yaml",
        help="Hydra config name passed to train_contrastive.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/modelnet40_vicreg/vn_pointnet_sweep"),
        help="Root directory where sweep folders are created.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epoch count per trial.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            "Training subset size override. Must be >= half of ModelNet40 train "
            f"({MIN_MAX_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="auto",
        help="Warm-start checkpoint path, or 'auto' to use newest .ckpt under output/.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="[0,1]",
        help="Hydra override for devices (example: \"[0,1]\").",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["offline", "online", "disabled"],
        default="disabled",
        help="W&B mode override for sweep runs (default: disabled).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Micro-batch size override applied to all trials.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=40,
        help="Gradient accumulation override applied to all trials.",
    )
    parser.add_argument(
        "--objective",
        choices=["trainer_test_metric", "analysis_canonical_acc"],
        default="trainer_test_metric",
        help=(
            "Metric source used for ranking. "
            "'trainer_test_metric' parses train logs; "
            "'analysis_canonical_acc' reads legacy analysis_metrics.json "
            "or falls back to test/class/acc_kmeans_plusplus_hungarian_canonical in logs."
        ),
    )
    parser.add_argument(
        "--trainer-metric-name",
        type=str,
        default=DEFAULT_TRAINER_METRIC,
        help="Metric key to parse when objective=trainer_test_metric.",
    )
    parser.add_argument(
        "--metric-direction",
        choices=["max", "min"],
        default="max",
        help=(
            "How to rank parsed trainer metrics "
            "(default: max for val/class/encoder_linear_svm_accuracy)."
        ),
    )
    parser.add_argument(
        "--search-space-json",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default search space (key -> list).",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override(s), repeatable. Example: --extra-override vicreg_enabled=true",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Continue sweep when an individual trial fails (default: enabled).",
    )
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately when a trial fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate trials and commands without launching training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "src/training_methods/contrastive_learning/train_contrastive.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.accumulate_grad_batches < 1:
        raise ValueError(
            f"--accumulate-grad-batches must be >= 1, got {args.accumulate_grad_batches}"
        )
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.max_samples < MIN_MAX_SAMPLES:
        raise ValueError(
            f"--max-samples must be >= {MIN_MAX_SAMPLES} (half of ModelNet40 train split "
            f"{MODELNET40_TRAINSET_SIZE}), got {args.max_samples}."
        )
    if args.objective == "analysis_canonical_acc":
        raise ValueError(
            "analysis_canonical_acc objective is disabled for fast sweep mode because "
            "post-training analysis is skipped. Use trainer metric objective instead."
        )

    init_checkpoint = _resolve_init_checkpoint(args.init_checkpoint, repo_root)

    search_space = _load_search_space(args.search_space_json)
    rng = random.Random(args.seed)
    trials = _sample_unique_params(
        search_space=search_space,
        rng=rng,
        num_trials=int(args.num_trials),
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = (repo_root / args.output_root / stamp).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "num_trials": int(args.num_trials),
        "config_name": str(args.config_name),
        "objective": str(args.objective),
        "trainer_metric_name": str(args.trainer_metric_name),
        "metric_direction": str(args.metric_direction),
        "batch_size": int(args.batch_size),
        "accumulate_grad_batches": int(args.accumulate_grad_batches),
        "max_samples": int(args.max_samples),
        "devices": str(args.devices),
        "wandb_mode": str(args.wandb_mode),
        "epochs_override": None if args.epochs is None else int(args.epochs),
        "init_checkpoint": str(init_checkpoint),
        "search_space": search_space,
        "extra_overrides": list(args.extra_override),
    }
    _write_json(sweep_dir / "sweep_manifest.json", manifest)

    print(f"Sweep directory: {sweep_dir}")
    print(f"Trials: {len(trials)}")
    print(f"Objective: {args.objective}")
    if args.objective == "trainer_test_metric":
        print(f"Trainer metric name: {args.trainer_metric_name}")
        print(f"Metric direction: {args.metric_direction}")
    print(f"Warm-start checkpoint: {init_checkpoint}")
    supervised_metrics_enabled = str(args.trainer_metric_name).startswith(
        ("val/class/", "test/class/")
    )
    print(
        "Sweep overrides: "
        f"epochs={args.epochs}, max_samples={args.max_samples}, "
        "run_test_after_training=false, run_post_training_analysis=false, "
        f"enable_supervised_metrics={_bool_to_override(supervised_metrics_enabled)}"
    )
    if args.dry_run:
        print("DRY RUN enabled: no training will be launched.")

    records: list[TrialResult] = []
    for i, params in enumerate(trials, start=1):
        trial_dir = sweep_dir / f"trial_{i:03d}"
        command = _build_command(
            python_exe=sys.executable,
            train_script=train_script,
            config_name=args.config_name,
            run_dir=trial_dir,
            trial_id=i,
            params=params,
            epochs_override=args.epochs,
            max_samples=int(args.max_samples),
            init_checkpoint=init_checkpoint,
            devices_override=args.devices,
            wandb_mode=args.wandb_mode,
            batch_size=int(args.batch_size),
            accumulate_grad_batches=int(args.accumulate_grad_batches),
            extra_overrides=list(args.extra_override),
            trainer_metric_name=str(args.trainer_metric_name),
            metric_direction=str(args.metric_direction),
        )
        print(f"[trial {i:03d}/{len(trials):03d}] launching...")
        result = _run_trial(
            command=command,
            repo_root=repo_root,
            run_dir=trial_dir,
            trial_id=i,
            objective=args.objective,
            trainer_metric_name=args.trainer_metric_name,
            continue_on_error=bool(args.continue_on_error),
            dry_run=bool(args.dry_run),
            params=params,
        )
        records.append(result)

        summary_payload = {
            "records": [
                {
                    "trial_id": r.trial_id,
                    "run_dir": r.run_dir,
                    "status": r.status,
                    "return_code": r.return_code,
                    "duration_sec": r.duration_sec,
                    "score": r.score,
                    "score_source": r.score_source,
                    "error": r.error,
                    "params": r.params,
                    "command": r.command,
                }
                for r in records
            ]
        }
        _write_json(sweep_dir / "results.json", summary_payload)
        _write_csv(sweep_dir / "results.csv", records)

        if result.score is None:
            print(
                f"[trial {i:03d}] status={result.status} score=NA "
                f"return_code={result.return_code}"
            )
        else:
            print(
                f"[trial {i:03d}] status={result.status} score={result.score:.6f} "
                f"source={result.score_source}"
            )

    best = _select_best(records, metric_direction=args.metric_direction)
    if best is None:
        if args.dry_run:
            print("\nDry run completed. No scoring was attempted.")
            print(f"Inspect generated commands in: {sweep_dir}")
            return 0
        print("\nNo trial produced a valid score.")
        print(f"Inspect: {sweep_dir / 'results.csv'} and {sweep_dir / 'results.json'}")
        return 2

    best_payload = {
        "trial_id": best.trial_id,
        "run_dir": best.run_dir,
        "score": best.score,
        "score_source": best.score_source,
        "params": best.params,
        "command": best.command,
    }
    _write_json(sweep_dir / "best_trial.json", best_payload)

    print("\nBest trial:")
    print(f"  trial_id: {best.trial_id:03d}")
    print(f"  score: {best.score:.6f}")
    print(f"  source: {best.score_source}")
    print(f"  run_dir: {best.run_dir}")
    print(f"  params: {json.dumps(best.params, sort_keys=True)}")
    print(f"  results_csv: {sweep_dir / 'results.csv'}")

    n_failed = sum(1 for r in records if r.status == "failed")
    n_missing = sum(1 for r in records if r.status == "missing_metric")
    if n_failed > 0 or n_missing > 0:
        print(
            "\nCompleted with non-ideal trials: "
            f"failed={n_failed}, missing_metric={n_missing}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
