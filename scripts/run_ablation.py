#!/usr/bin/env python3
"""Utility to launch single-variable ablation studies and aggregate metrics."""

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

import sys
sys.path.append(os.getcwd())
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.training_methods.spd.train_spd import get_rundir_name, init_wandb
from src.data_utils.data_module import PointCloudDataModule

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is part of project requirements
    wandb = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_list(value: Optional[Union[List[Any], ListConfig, tuple]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_metric_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _merge_decoder_override(cfg: DictConfig, override_dict: Dict[str, Any]) -> Dict[str, Any]:
    base = OmegaConf.to_container(cfg.decoder, resolve=False)
    base_dict = deepcopy(base) if base is not None else {}
    override_copy = deepcopy(override_dict)

    override_copy.pop("label", None)
    replace_all = bool(override_copy.pop("replace", False))
    replace_kwargs = bool(override_copy.pop("replace_kwargs", False))

    if replace_all:
        merged: Dict[str, Any] = {}
    else:
        merged = base_dict if isinstance(base_dict, dict) else {}

    if "name" in override_copy and override_copy["name"] is not None:
        merged["name"] = override_copy.pop("name")

    if "kwargs" in override_copy:
        kwargs_update = override_copy.pop("kwargs") or {}
        if replace_kwargs or "kwargs" not in merged:
            merged["kwargs"] = kwargs_update
        else:
            current_kwargs = dict(merged.get("kwargs", {}))
            current_kwargs.update(kwargs_update)
            merged["kwargs"] = current_kwargs

    for key, val in override_copy.items():
        merged[key] = val

    return merged


def _value_to_display(value: Any) -> str:
    value = _normalize_value(value)
    if isinstance(value, dict):
        if "label" in value:
            return str(value["label"])
        name = value.get("name")
        if name is None:
            return json.dumps(value, sort_keys=True)
        display = str(name)
        kwargs = value.get("kwargs")
        if isinstance(kwargs, dict) and kwargs:
            items = []
            for key in sorted(kwargs):
                val = kwargs[key]
                if isinstance(val, (int, float)):
                    items.append(f"{key}={val:.6g}")
                elif isinstance(val, bool):
                    items.append(f"{key}={'true' if val else 'false'}")
                else:
                    items.append(f"{key}={val}")
            if items:
                display += " (" + ", ".join(items) + ")"
        return display
    if isinstance(value, list):
        return "[" + ", ".join(_value_to_display(v) for v in value) + "]"
    return str(value)


def _format_value_for_tag(value: Any) -> str:
    value = _normalize_value(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}".replace(".", "p").replace("-", "m")
    if isinstance(value, dict):
        if "label" in value:
            return _format_value_for_tag(value["label"])
        base = _format_value_for_tag(value.get("name", "dict"))
        kwargs = value.get("kwargs")
        if isinstance(kwargs, dict) and kwargs:
            extras = []
            for key in sorted(kwargs):
                val = kwargs[key]
                if isinstance(val, (int, float)):
                    fmt = f"{val:.6g}".replace(".", "p").replace("-", "m")
                    extras.append(f"{key}={fmt}")
                elif isinstance(val, bool):
                    extras.append(f"{key}={'true' if val else 'false'}")
                else:
                    extras.append(f"{key}={str(val).replace(' ', '_')}")
            if extras:
                base += "_" + "_".join(extras)
        return base
    if isinstance(value, list):
        return "_".join(_format_value_for_tag(v) for v in value)
    return str(value).replace(" ", "_")


def _format_value_for_path(value: Any, *, fallback: Optional[str] = None) -> str:
    normalized = _normalize_value(value)
    candidate: Optional[str] = None
    if isinstance(normalized, dict):
        for key in ("label", "name"):
            text = normalized.get(key)
            if isinstance(text, str) and text.strip():
                candidate = text.strip()
                break
    elif isinstance(normalized, str) and normalized.strip():
        candidate = normalized.strip()
    if candidate is None:
        candidate = _format_value_for_tag(normalized)

    def _sanitize(text: str) -> str:
        filtered = []
        for ch in text.strip():
            if ch.isalnum():
                filtered.append(ch)
            elif ch in {"-", "_"}:
                filtered.append(ch)
            else:
                filtered.append("-")
        safe = "".join(filtered).strip("-_")
        while "--" in safe:
            safe = safe.replace("--", "-")
        while "__" in safe:
            safe = safe.replace("__", "_")
        if len(safe) > 48:
            safe = safe[:48]
        return safe

    safe_candidate = _sanitize(candidate)
    ambiguous_tokens = {"", "value", "dict", "mapping", "list"}
    if safe_candidate in ambiguous_tokens or safe_candidate.startswith("mapping-values"):
        fallback_text = fallback or "value"
        safe_fallback = _sanitize(fallback_text)
        return safe_fallback or "value"
    return safe_candidate


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


# def _resolve_devices(cfg: DictConfig) -> Union[int, Sequence[int]]:
#     devices = cfg.get("devices")
#     if isinstance(devices, ListConfig):
#         devices = list(devices)
#     if devices in (None, 0):
#         return [0, 1]
#     if isinstance(devices, int):
#         return [devices]
#     return devices


@dataclass
class MetricSpec:
    name: str
    mode: str

    def __post_init__(self):
        mode_lower = self.mode.lower()
        if mode_lower not in {"min", "max"}:
            raise ValueError(f"Unsupported mode '{self.mode}' for metric '{self.name}' (expected 'min' or 'max').")
        self.mode = mode_lower


class MetricTracker(Callback):
    """Records per-epoch metrics and computes best values for selected targets."""

    def __init__(self, final_metrics: Iterable[str], best_specs: Iterable[MetricSpec]):
        super().__init__()
        final_list = list(final_metrics)
        best_list = list(best_specs)
        self._metric_names = set(final_list + [spec.name for spec in best_list])
        self._best_specs = {spec.name: spec for spec in best_list}
        self.history: List[Dict[str, Any]] = []
        self._best: Dict[str, Optional[Dict[str, Any]]] = {name: None for name in self._metric_names}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module  # unused
        epoch = trainer.current_epoch
        snapshot: Dict[str, Any] = {"epoch": int(epoch)}
        metrics = trainer.callback_metrics
        for name in self._metric_names:
            value = self._extract_metric(metrics, name)
            if value is None:
                continue
            snapshot[name] = value
            if "/" in name:
                snapshot[_sanitize_metric_name(name)] = value
            spec = self._best_specs.get(name)
            if spec is None:
                continue
            current_best = self._best[name]
            if current_best is None or self._is_improvement(value, current_best["value"], spec.mode):
                self._best[name] = {"epoch": int(epoch), "value": value}
        self.history.append(snapshot)

    @staticmethod
    def _is_improvement(candidate: float, incumbent: float, mode: str) -> bool:
        if incumbent is None:
            return True
        if mode == "min":
            return candidate < incumbent
        return candidate > incumbent

    @staticmethod
    def _extract_metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
        if name in metrics:
            return _to_float(metrics[name])
        alt = _sanitize_metric_name(name)
        if alt in metrics:
            return _to_float(metrics[alt])
        return None

    def last_metrics(self, names: Iterable[str]) -> Dict[str, Optional[float]]:
        names = list(names)
        result = {name: None for name in names}
        if not self.history:
            return result
        latest = self.history[-1]
        for name in names:
            if name in latest:
                result[name] = latest[name]
            else:
                alt = _sanitize_metric_name(name)
                result[name] = latest.get(alt)
        return result

    def best_metrics(self) -> Dict[str, Optional[Dict[str, Any]]]:
        return deepcopy(self._best)


def load_ablation_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError("Ablation configuration must be a DictConfig.")
    return cfg


def validate_ablation_config(cfg: DictConfig) -> None:
    required_sections = ("experiment", "variable", "metrics")
    for section in required_sections:
        if section not in cfg or cfg[section] is None:
            raise ValueError(f"Missing '{section}' section in ablation config.")

    experiment = cfg.experiment
    if "config_name" not in experiment:
        raise ValueError("experiment.config_name is required.")

    variable = cfg.variable
    if "override" not in variable:
        raise ValueError("variable.override is required.")
    if "values" not in variable or not variable.values:
        raise ValueError("variable.values must contain at least one value.")

    metrics = cfg.metrics
    if "final" not in metrics or not metrics.final:
        raise ValueError("metrics.final must list at least one metric.")
    if "best" in metrics:
        for entry in metrics.best:
            if "name" not in entry or "mode" not in entry:
                raise ValueError("Each metrics.best entry must define 'name' and 'mode'.")


def prepare_output_root(cfg: DictConfig) -> Tuple[Path, str]:
    output_cfg = cfg.output if "output" in cfg and cfg.output is not None else DictConfig({})
    root_dir = Path(output_cfg.get("root_dir", "output/ablations"))
    run_name = output_cfg.get("run_name")
    if not run_name:
        run_name = f"{cfg.variable.override.replace('.', '_')}_ablation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root_dir / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, run_name


def compose_training_config(
    exp_cfg: DictConfig,
    variable_override: str,
    value: Any,
    *,
    run_name: str,
    value_label: str,
) -> DictConfig:
    overrides = _as_list(exp_cfg.get("base_overrides", []))
    config_path_entry = exp_cfg.get("config_path", "configs")
    config_path = Path(config_path_entry)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("experiment.config_path must point inside the repository root.") from exc
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration directory '{config_path}' does not exist.")
    job_name = f"ablation_{variable_override.replace('.', '_')}"
    with initialize_config_dir(version_base=None, config_dir=str(config_path), job_name=job_name):
        cfg = compose(config_name=exp_cfg.config_name, overrides=list(overrides))
    OmegaConf.set_struct(cfg, False)
    normalized_value = _normalize_value(value)
    tag_source = normalized_value
    decoder_override_applied = False
    if variable_override == "decoder" and isinstance(normalized_value, dict):
        normalized_value = _merge_decoder_override(cfg, normalized_value)
        decoder_override_applied = True
    try:
        if decoder_override_applied:
            cfg[variable_override] = OmegaConf.create(normalized_value)
        else:
            OmegaConf.update(cfg, variable_override, normalized_value, force_add=True)
    except Exception as exc:
        raise ValueError(
            f"Failed to apply override '{variable_override}' with value {normalized_value!r}."
            " Verify that the key exists in the training config or adjust the path."
        ) from exc
    cfg.wandb_mode = "online"
    variable_sanitized = variable_override.replace(".", "_")
    experiment_label = f"ablation_{run_name}_{variable_sanitized}_{value_label}"
    cfg.experiment_name = experiment_label

    tag_value = _format_value_for_tag(tag_source)
    tags = _as_list(cfg.get("experiment_tags"))
    tags.append(f"abl/{variable_override}={tag_value}")
    cfg.experiment_tags = tags
    return cfg


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_history(path: Path, history: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def run_single_experiment(
    cfg: DictConfig,
    metrics_cfg: DictConfig,
    run_output_dir: Path,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[Dict[str, Any]]], str, List[Dict[str, Any]]]:
    final_metric_names = _as_list(metrics_cfg.final)
    best_specs = [MetricSpec(name=entry.name, mode=entry.mode) for entry in _as_list(metrics_cfg.get("best", []))]
    tracker = MetricTracker(final_metric_names, best_specs)

    run_output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_output_dir / "resolved_training_config.json", OmegaConf.to_container(cfg, resolve=True))

    run_dir = get_rundir_name()
    wandb_logger = init_wandb(cfg, run_dir)
    
    datamodule = PointCloudDataModule(cfg)
    model = ShapePoseDisentanglement(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor="val_loss",
        filename=f"{cfg.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    precision = cfg.get("precision", "32")
    
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator="gpu" if cfg.gpu else "cpu",
        devices=[0],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, tracker],
        log_every_n_steps=cfg.log_every_n_steps,
        precision=precision,
        benchmark=True,
    )

    try:
        trainer.fit(model, datamodule)
    finally:
        if wandb is not None:
            wandb.finish()

    final_metrics = tracker.last_metrics(final_metric_names)
    best_metrics = tracker.best_metrics()
    _write_json(run_output_dir / "final_metrics.json", final_metrics)
    _write_json(run_output_dir / "best_metrics.json", best_metrics)
    _write_history(run_output_dir / "metrics_history.jsonl", tracker.history)
    with (run_output_dir / "training_run_dir.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(run_dir))
    return final_metrics, best_metrics, run_dir, tracker.history


def aggregate_and_save_tables(
    runs: List[Dict[str, Any]],
    output_dir: Path,
    final_metric_names: Sequence[str],
    best_specs: Sequence[MetricSpec],
    variable_name: str,
) -> None:
    def fmt_metric(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None

        try:
            decimal_value = Decimal(str(value))
        except (ArithmeticError, ValueError):
            return str(value)

        if decimal_value.is_nan():
            return "nan"
        if decimal_value.is_infinite():
            return "inf" if decimal_value > 0 else "-inf"
        if decimal_value == 0:
            return "0"

        abs_value = abs(decimal_value)
        if abs_value >= 1:
            quantize_unit = Decimal("0.01")
        else:
            significant_digits = 2
            exponent = abs_value.adjusted() - significant_digits + 1
            quantize_unit = Decimal(f"1e{exponent}")

        rounded_value = decimal_value.quantize(quantize_unit, rounding=ROUND_HALF_UP)
        if rounded_value == 0:
            return "0"

        formatted = format(rounded_value.normalize(), "f")
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted or "0"

    final_rows: List[Dict[str, Any]] = []
    for entry in runs:
        variable_display = entry.get("value_display", entry["value"])
        row: Dict[str, Any] = {"variable": variable_display}
        final_metrics = entry["final_metrics"]
        for metric in final_metric_names:
            column = _sanitize_metric_name(metric)
            row[column] = fmt_metric(final_metrics.get(metric))
        final_rows.append(row)
    _write_csv(output_dir / "final_metrics.csv", final_rows)

    for spec in best_specs:
        column_value = _sanitize_metric_name(spec.name)
        rows: List[Dict[str, Any]] = []
        for entry in runs:
            best_data = entry["best_metrics"].get(spec.name)
            variable_display = entry.get("value_display", entry["value"])
            rows.append(
                {
                    "variable": variable_display,
                    f"{column_value}_value": None if best_data is None else fmt_metric(best_data.get("value")),
                    f"{column_value}_epoch": None if best_data is None else best_data.get("epoch"),
                }
            )
        filename = f"best_{column_value}.csv"
        _write_csv(output_dir / filename, rows)

    overview_payload = {
        "variable": variable_name,
        "runs": runs,
        "generated_at": datetime.now().isoformat(),
    }
    _write_json(output_dir / "summary.json", overview_payload)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.touch()
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ablation study with automated metric summaries.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the ablation study YAML configuration.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    original_wd = Path.cwd()
    if REPO_ROOT != original_wd:
        os.chdir(REPO_ROOT)

    cfg = load_ablation_config(config_path)
    validate_ablation_config(cfg)
    output_root, run_name = prepare_output_root(cfg)
    _write_json(output_root / "ablation_config_resolved.json", OmegaConf.to_container(cfg, resolve=True))

    exp_cfg = cfg.experiment
    variable_cfg = cfg.variable
    metric_cfg = cfg.metrics

    metric_specs = [MetricSpec(name=entry.name, mode=entry.mode) for entry in _as_list(metric_cfg.get("best", []))]
    final_metric_names = _as_list(metric_cfg.final)

    override_key = variable_cfg.get("override")
    if override_key is None:
        raise ValueError("variable.override must be set in the ablation config.")
    values = _as_list(variable_cfg.get("values"))

    runs: List[Dict[str, Any]] = []
    for idx, value in enumerate(values):
        value_serializable = _normalize_value(value)
        fallback_label = f"value_{idx + 1:02d}"
        value_label = _format_value_for_path(value_serializable, fallback=fallback_label)
        value_display = _value_to_display(value_serializable)
        cfg_instance = compose_training_config(
            exp_cfg,
            override_key,
            value,
            run_name=run_name,
            value_label=value_label,
        )
        run_dir = output_root / f"{override_key.replace('.', '_')}_{value_label}"
        final_metrics, best_metrics, training_dir, history = run_single_experiment(cfg_instance, metric_cfg, run_dir)
        runs.append(
            {
                "value": _json_safe(value_serializable),
                "value_display": value_display,
                "value_label": value_label,
                "final_metrics": final_metrics,
                "best_metrics": best_metrics,
                "training_dir": training_dir,
                "history_path": str((run_dir / "metrics_history.jsonl").relative_to(output_root)),
            }
        )

    aggregate_and_save_tables(runs, output_root, final_metric_names, metric_specs, override_key)

    if REPO_ROOT != original_wd:
        os.chdir(original_wd)


if __name__ == "__main__":
    main(sys.argv[1:])
