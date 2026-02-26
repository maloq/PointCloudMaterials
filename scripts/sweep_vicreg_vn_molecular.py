#!/usr/bin/env python3
"""Random-search hyperparameter sweep for VICReg molecular training.

This script launches multiple `train_contrastive.py` runs with Hydra overrides,
collects target metrics, and ranks trials.

By default it optimizes `val/class/acc_kmeans_plusplus_hungarian` and reads the
best checkpoint-monitored score (best over epochs, not final epoch).

Example:
  python scripts/sweep_vicreg_vn_molecular.py \
      --num-trials 20 \
      --devices "[0]" \
      --epochs 80
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


DEFAULT_CONFIG_NAME = "vicreg_vn_molecular.yaml"
DEFAULT_METRIC_NAME = "val/class/acc_kmeans_plusplus_hungarian"
DEFAULT_METRIC_FALLBACKS: list[str] = []

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    # Optimization.
    "learning_rate": [8e-4, 1.2e-3, 2.0e-3, 3.0e-3, 4.0e-3],
    "decay_rate": [1e-4, 2e-4, 4e-4],
    "scheduler_min_lr": [5e-7, 1e-6, 2e-6],
    "warmup_epochs": [6, 12, 18],
    # VICReg loss balance.
    "vicreg_sim_coeff": [15.0, 25.0, 35.0],
    "vicreg_std_coeff": [15.0, 25.0, 35.0],
    "vicreg_cov_coeff": [0.5, 1.0, 2.0],
    "vicreg_radial_beta1": [0.5, 1.0, 1.5],
    "vicreg_radial_beta2": [0.05, 0.1, 0.2],
    # View augmentation.
    "vicreg_jitter_std": [0.0, 0.005, 0.01, 0.02],
    "vicreg_drop_ratio": [0.0, 0.05, 0.1, 0.2],
    "vicreg_neighbor_view_mode": ["second", "both", "first"],
    "vicreg_neighbor_k": [4, 6, 8, 10],
    "vicreg_neighbor_max_relative_distance": [0.0, 0.05, 0.1],
    "vicreg_rotation_mode": ["none", "full"],
    "vicreg_rotation_deg": [5.0, 10.0, 20.0],
    "vicreg_strain_std": [0.0, 0.01, 0.02, 0.04],
    "vicreg_occlusion_mode": ["none", "mixed", "slab", "cone"],
    "vicreg_occlusion_view": ["second", "both"],
    "vicreg_occlusion_slab_frac": [0.1, 0.2, 0.3],
    "vicreg_occlusion_cone_deg": [10.0, 20.0, 30.0],
}


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SAFE_OVERRIDE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")


@dataclass
class TrialResult:
    trial_id: int
    run_dir: str
    status: str
    return_code: int
    duration_sec: float
    score: float | None
    score_source: str | None
    score_metric: str | None
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
            "score_metric": self.score_metric,
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
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if SAFE_OVERRIDE_TOKEN_RE.fullmatch(text) is not None:
        return text
    return json.dumps(text)


def _force_override(key: str, value: Any) -> str:
    if not isinstance(key, str) or key.strip() == "":
        raise ValueError(f"Override key must be a non-empty string, got {key!r}")
    return f"++{key}={_override_value(value)}"


def _load_search_space(path: Path | None) -> dict[str, list[Any]]:
    if path is None:
        return dict(DEFAULT_SEARCH_SPACE)

    if not path.exists():
        raise FileNotFoundError(f"Search-space file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            "Search-space JSON must be an object mapping parameter keys to lists, "
            f"got {type(data)!r}."
        )
    out: dict[str, list[Any]] = {}
    for key, values in data.items():
        if not isinstance(key, str) or key.strip() == "":
            raise ValueError(f"Search-space key must be a non-empty string, got {key!r}")
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search-space entry '{key}' must be a non-empty list, got {values!r}"
            )
        out[key] = values
    return out


def _search_space_cardinality(search_space: dict[str, list[Any]]) -> int:
    total = 1
    for key, values in search_space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Search-space entry '{key}' must be a non-empty list, got {values!r}"
            )
        total *= len(values)
    return total


def _validate_sampled_params(params: dict[str, Any]) -> None:
    learning_rate = float(params.get("learning_rate", 0.0))
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    decay_rate = float(params.get("decay_rate", 0.0))
    if decay_rate < 0:
        raise ValueError(f"decay_rate must be >= 0, got {decay_rate}")

    scheduler_min_lr = float(params.get("scheduler_min_lr", 0.0))
    if scheduler_min_lr < 0:
        raise ValueError(f"scheduler_min_lr must be >= 0, got {scheduler_min_lr}")
    if scheduler_min_lr > learning_rate:
        raise ValueError(
            f"scheduler_min_lr ({scheduler_min_lr}) cannot exceed learning_rate ({learning_rate})."
        )

    warmup_epochs = int(params.get("warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

    drop_ratio = float(params.get("vicreg_drop_ratio", 0.0))
    if not (0.0 <= drop_ratio < 1.0):
        raise ValueError(f"vicreg_drop_ratio must be in [0, 1), got {drop_ratio}")

    jitter_std = float(params.get("vicreg_jitter_std", 0.0))
    if jitter_std < 0.0:
        raise ValueError(f"vicreg_jitter_std must be >= 0, got {jitter_std}")

    neighbor_k = int(params.get("vicreg_neighbor_k", 1))
    if neighbor_k < 1:
        raise ValueError(f"vicreg_neighbor_k must be >= 1, got {neighbor_k}")

    neighbor_max_distance = float(params.get("vicreg_neighbor_max_relative_distance", 0.0))
    if neighbor_max_distance < 0.0:
        raise ValueError(
            "vicreg_neighbor_max_relative_distance must be >= 0, "
            f"got {neighbor_max_distance}"
        )

    strain_std = float(params.get("vicreg_strain_std", 0.0))
    if strain_std < 0.0:
        raise ValueError(f"vicreg_strain_std must be >= 0, got {strain_std}")

    occlusion_mode = str(params.get("vicreg_occlusion_mode", "none")).lower()
    allowed_occlusion_modes = {"none", "mixed", "slab", "cone"}
    if occlusion_mode not in allowed_occlusion_modes:
        raise ValueError(
            "vicreg_occlusion_mode must be one of "
            f"{sorted(allowed_occlusion_modes)}, got {occlusion_mode!r}"
        )

    occlusion_slab = float(params.get("vicreg_occlusion_slab_frac", 0.0))
    if not (0.0 <= occlusion_slab <= 1.0):
        raise ValueError(f"vicreg_occlusion_slab_frac must be in [0, 1], got {occlusion_slab}")

    occlusion_cone = float(params.get("vicreg_occlusion_cone_deg", 0.0))
    if not (0.0 <= occlusion_cone < 180.0):
        raise ValueError(
            f"vicreg_occlusion_cone_deg must be in [0, 180), got {occlusion_cone}"
        )

    rotation_mode = str(params.get("vicreg_rotation_mode", "none")).lower()
    if rotation_mode not in {"none", "full"}:
        raise ValueError(
            f"vicreg_rotation_mode must be 'none' or 'full', got {rotation_mode!r}"
        )
    rotation_deg = float(params.get("vicreg_rotation_deg", 0.0))
    if rotation_deg < 0.0:
        raise ValueError(f"vicreg_rotation_deg must be >= 0, got {rotation_deg}")


def _sample_unique_params(
    *,
    search_space: dict[str, list[Any]],
    rng: random.Random,
    num_trials: int,
) -> list[dict[str, Any]]:
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")

    total_unique = _search_space_cardinality(search_space)
    if num_trials > total_unique:
        raise ValueError(
            f"Requested num_trials={num_trials}, but search space only has "
            f"{total_unique} unique combinations."
        )

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(500, 30 * num_trials)

    while len(trials) < num_trials:
        if max_attempts <= 0:
            raise RuntimeError(
                "Could not sample enough unique trials from search space. "
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


def _parse_last_hungarian_metric(text: str) -> tuple[float | None, str | None]:
    cleaned = _strip_ansi(text)
    pattern = re.compile(
        r"((?:val|test)/class/acc_kmeans_plusplus_hungarian(?:_canonical)?(?:_k\d+)?)"
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
        run_dir.rglob("*.ckpt"),
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


def _read_metric_from_json(
    *,
    run_dir: Path,
    metric_names: list[str],
) -> tuple[float | None, str | None, str | None]:
    final_metrics_path = run_dir / "final_metrics.json"
    if not final_metrics_path.exists():
        return None, None, None

    data = json.loads(final_metrics_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {final_metrics_path}, got {type(data)!r}"
        )

    for metric_name in metric_names:
        if metric_name not in data:
            continue
        value = data[metric_name]
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        return parsed, f"{final_metrics_path}:{metric_name}", metric_name
    return None, None, None


def _read_metric_from_run_dir(
    *,
    run_dir: Path,
    metric_names: list[str],
) -> tuple[float | None, str | None, str | None]:
    # For val/* objectives use the checkpoint callback score so ranking uses
    # the best value observed during training, not the final epoch value.
    for metric_name in metric_names:
        metric_name_text = str(metric_name).strip()
        if not metric_name_text.startswith("val/"):
            continue
        ckpt_value, ckpt_source = _read_checkpoint_monitored_score(
            run_dir=run_dir,
            monitor_name=metric_name_text,
        )
        if ckpt_value is not None:
            return ckpt_value, ckpt_source, metric_name_text

    parsed, source, metric = _read_metric_from_json(run_dir=run_dir, metric_names=metric_names)
    if parsed is not None:
        return parsed, source, metric

    candidates = [
        run_dir / "train_contrastive.log",
        run_dir / "train.log",
        run_dir / "sweep_driver.log",
    ]

    for metric_name in metric_names:
        for path in candidates:
            if not path.exists():
                continue
            value = _parse_metric_from_text(path.read_text(errors="replace"), metric_name)
            if value is not None:
                return value, f"{path}:{metric_name}", metric_name

    for path in candidates:
        if not path.exists():
            continue
        value, parsed_metric = _parse_last_hungarian_metric(path.read_text(errors="replace"))
        if value is None or parsed_metric is None:
            continue
        return value, f"{path}:{parsed_metric}", parsed_metric

    return None, None, None


def _resolve_checkpoint_path(checkpoint_arg: str | None, repo_root: Path) -> Path | None:
    if checkpoint_arg is None:
        return None

    raw = str(checkpoint_arg).strip()
    if raw == "":
        raise ValueError("--init-checkpoint cannot be empty when provided.")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"--init-checkpoint path does not exist: {path}")
    if path.suffix.lower() != ".ckpt":
        raise ValueError(f"--init-checkpoint must point to a .ckpt file, got: {path}")
    return path


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
    devices_override: str | None,
    wandb_mode: str,
    batch_size: int | None,
    accumulate_grad_batches: int | None,
    max_samples: int | None,
    init_checkpoint: Path | None,
    primary_metric_name: str,
    metric_direction: str,
    extra_overrides: list[str],
) -> list[str]:
    command = [
        python_exe,
        str(train_script),
        "--config-name",
        config_name,
        f"hydra.run.dir={run_dir}",
        f"experiment_name=VICREG_vn_molecular_sweep_t{trial_id:03d}",
        f"wandb_mode={wandb_mode}",
        "++run_post_training_analysis=false",
        "++run_test_after_training=true",
        "++enable_supervised_metrics=true",
        "++enable_test_so3_metrics=false",
        "++check_val_every_n_epoch=1",
        "++num_sanity_val_steps=0",
    ]

    if devices_override is not None:
        command.append(f"devices={devices_override}")

    if init_checkpoint is not None:
        command.append(f"++init_from_checkpoint={_override_value(str(init_checkpoint))}")
        command.append("++init_from_checkpoint_strict=false")

    if str(primary_metric_name).startswith("val/"):
        metric_mode = str(metric_direction).strip().lower()
        if metric_mode not in {"min", "max"}:
            raise ValueError(
                f"metric_direction must be 'min' or 'max', got {metric_direction!r}."
            )
        command.append(f"++checkpoint_monitor={_override_value(primary_metric_name)}")
        command.append(f"++checkpoint_mode={metric_mode}")

    if epochs_override is not None:
        if epochs_override < 1:
            raise ValueError(f"--epochs must be >= 1, got {epochs_override}")
        command.append(f"epochs={epochs_override}")

    if batch_size is not None:
        if batch_size < 1:
            raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
        command.append(f"batch_size={batch_size}")

    if accumulate_grad_batches is not None:
        if accumulate_grad_batches < 1:
            raise ValueError(
                "--accumulate-grad-batches must be >= 1, got "
                f"{accumulate_grad_batches}"
            )
        command.append(f"accumulate_grad_batches={accumulate_grad_batches}")

    if max_samples is not None:
        if max_samples < 0:
            raise ValueError(f"--max-samples must be >= 0, got {max_samples}")
        command.append(f"max_samples={max_samples}")

    for key in sorted(params.keys()):
        command.append(_force_override(key, params[key]))

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
    metric_names: list[str],
    continue_on_error: bool,
    dry_run: bool,
    params: dict[str, Any],
) -> TrialResult:
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {run_dir}. "
            "Use a new output root or remove stale trial folders."
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
                text=True,
                check=False,
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
    score, source, metric = _read_metric_from_run_dir(
        run_dir=run_dir,
        metric_names=metric_names,
    )
    if status == "ok" and score is None:
        status = "missing_metric"
        error = (
            "Run finished but target metric was not found in expected outputs. "
            f"Checked metrics={metric_names}."
        )

    return TrialResult(
        trial_id=trial_id,
        run_dir=str(run_dir),
        status=status,
        return_code=return_code,
        duration_sec=duration,
        score=score,
        score_source=source,
        score_metric=metric,
        error=error,
        params=dict(params),
        command=list(command),
    )


def _select_best(records: list[TrialResult], *, metric_direction: str) -> TrialResult | None:
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
            "Random-search sweep for train_contrastive.py + vicreg_vn_molecular.yaml, "
            "ranked by best monitored metric (default: best val Hungarian ACC)."
        )
    )
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Hydra config name passed to train_contrastive.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/vicreg_vn_molecular/hparam_sweep"),
        help="Root directory where sweep folders are created.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for every trial.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch_size override for every trial.",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=None,
        help="Optional accumulate_grad_batches override for every trial.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max_samples override for every trial (0 means full configured dataset).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="[0]",
        help='Hydra override for devices, e.g. "[0]" or "[0,1]".',
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["offline", "online", "disabled"],
        default="disabled",
        help="W&B mode override for sweep runs (default: disabled).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional init checkpoint (.ckpt) path for weights-only initialization.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default=DEFAULT_METRIC_NAME,
        help=(
            "Primary metric key used for ranking. "
            "Default uses best checkpoint-monitored validation Hungarian ACC."
        ),
    )
    parser.add_argument(
        "--metric-fallback",
        action="append",
        default=list(DEFAULT_METRIC_FALLBACKS),
        help=(
            "Fallback metric key(s) when --metric-name is absent. "
            "Repeat for multiple keys."
        ),
    )
    parser.add_argument(
        "--metric-direction",
        choices=["max", "min"],
        default="max",
        help="How to rank metrics (default: max).",
    )
    parser.add_argument(
        "--search-space-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with search space (key -> non-empty list). "
            "When omitted, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override(s), repeatable. Example: --extra-override pose.enabled=false",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Continue sweep when a trial fails (default: enabled).",
    )
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop sweep immediately when a trial fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate trial commands without launching training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "src/training_methods/contrastive_learning/train_contrastive.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    if args.num_trials < 1:
        raise ValueError(f"--num-trials must be >= 1, got {args.num_trials}")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.accumulate_grad_batches is not None and args.accumulate_grad_batches < 1:
        raise ValueError(
            "--accumulate-grad-batches must be >= 1, got "
            f"{args.accumulate_grad_batches}"
        )
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError(f"--max-samples must be >= 0, got {args.max_samples}")

    primary_metric = str(args.metric_name).strip()
    if primary_metric == "":
        raise ValueError("--metric-name must be a non-empty string.")

    metric_names = [primary_metric]
    for fallback in args.metric_fallback:
        metric_name = str(fallback).strip()
        if metric_name == "":
            raise ValueError("--metric-fallback entries must be non-empty strings.")
        if metric_name not in metric_names:
            metric_names.append(metric_name)

    search_space_path = args.search_space_json
    search_space = _load_search_space(search_space_path)

    init_checkpoint = _resolve_checkpoint_path(args.init_checkpoint, repo_root)

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
        "metric_names": metric_names,
        "metric_direction": str(args.metric_direction),
        "checkpoint_monitor_override": (
            primary_metric if str(primary_metric).startswith("val/") else None
        ),
        "checkpoint_mode_override": (
            str(args.metric_direction).lower() if str(primary_metric).startswith("val/") else None
        ),
        "devices": str(args.devices) if args.devices is not None else None,
        "wandb_mode": str(args.wandb_mode),
        "epochs_override": None if args.epochs is None else int(args.epochs),
        "batch_size_override": None if args.batch_size is None else int(args.batch_size),
        "accumulate_grad_batches_override": (
            None if args.accumulate_grad_batches is None else int(args.accumulate_grad_batches)
        ),
        "max_samples_override": None if args.max_samples is None else int(args.max_samples),
        "init_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        "search_space_source": (
            "built_in_defaults" if search_space_path is None else str(search_space_path)
        ),
        "search_space": search_space,
        "extra_overrides": list(args.extra_override),
    }
    _write_json(sweep_dir / "sweep_manifest.json", manifest)

    print(f"Sweep directory: {sweep_dir}")
    print(f"Trials: {len(trials)}")
    print(f"Config: {args.config_name}")
    print(f"Ranking metrics (ordered): {metric_names}")
    print(f"Metric direction: {args.metric_direction}")
    if primary_metric.startswith("val/"):
        print(
            "Checkpoint selection: "
            f"monitor={primary_metric}, mode={str(args.metric_direction).lower()}"
        )
    print(f"Devices override: {args.devices}")
    print(f"W&B mode: {args.wandb_mode}")
    if init_checkpoint is not None:
        print(f"Init checkpoint: {init_checkpoint}")
    if args.dry_run:
        print("DRY RUN enabled: no training will be launched.")

    records: list[TrialResult] = []
    for i, params in enumerate(trials, start=1):
        trial_dir = sweep_dir / f"trial_{i:03d}"
        command = _build_command(
            python_exe=sys.executable,
            train_script=train_script,
            config_name=str(args.config_name),
            run_dir=trial_dir,
            trial_id=i,
            params=params,
            epochs_override=None if args.epochs is None else int(args.epochs),
            devices_override=None if args.devices is None else str(args.devices),
            wandb_mode=str(args.wandb_mode),
            batch_size=None if args.batch_size is None else int(args.batch_size),
            accumulate_grad_batches=(
                None if args.accumulate_grad_batches is None else int(args.accumulate_grad_batches)
            ),
            max_samples=None if args.max_samples is None else int(args.max_samples),
            init_checkpoint=init_checkpoint,
            primary_metric_name=primary_metric,
            metric_direction=str(args.metric_direction),
            extra_overrides=list(args.extra_override),
        )

        print(f"[trial {i:03d}/{len(trials):03d}] launching...")
        result = _run_trial(
            command=command,
            repo_root=repo_root,
            run_dir=trial_dir,
            trial_id=i,
            metric_names=metric_names,
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
                    "score_metric": r.score_metric,
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
                f"metric={result.score_metric} source={result.score_source}"
            )

    best = _select_best(records, metric_direction=str(args.metric_direction))
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
        "score_metric": best.score_metric,
        "params": best.params,
        "command": best.command,
    }
    _write_json(sweep_dir / "best_trial.json", best_payload)

    print("\nBest trial:")
    print(f"  trial_id: {best.trial_id:03d}")
    print(f"  score: {best.score:.6f}")
    print(f"  metric: {best.score_metric}")
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
