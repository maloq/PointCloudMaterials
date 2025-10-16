#!/usr/bin/env python3
"""Rebuild aggregated ablation tables from partially completed study outputs."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError as exc:  # pragma: no cover - hard dependency for schema access
    raise SystemExit(
        "The recover_ablation_results script requires the 'omegaconf' package. "
        "Install project dependencies before running it."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class MetricSpec:
    name: str
    mode: str

    def __post_init__(self) -> None:
        mode_lower = self.mode.lower()
        if mode_lower not in {"min", "max"}:
            raise ValueError(f"Unsupported mode '{self.mode}' for metric '{self.name}'.")
        self.mode = mode_lower


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


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sanitize_metric_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _format_metric_value(value: Optional[float]) -> Optional[str]:
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


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def aggregate_and_save_tables(
    runs: List[Dict[str, Any]],
    output_dir: Path,
    final_metric_names: Sequence[str],
    best_specs: Sequence[MetricSpec],
    variable_name: str,
) -> None:
    final_rows: List[Dict[str, Any]] = []
    for entry in runs:
        variable_display = entry.get("value_display", entry["value"])
        row: Dict[str, Any] = {"variable": variable_display}
        final_metrics = entry["final_metrics"]
        for metric in final_metric_names:
            column = _sanitize_metric_name(metric)
            row[column] = _format_metric_value(final_metrics.get(metric))
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
                    f"{column_value}_value": None if best_data is None else _format_metric_value(best_data.get("value")),
                    f"{column_value}_epoch": None if best_data is None else best_data.get("epoch"),
                }
            )
        _write_csv(output_dir / f"best_{column_value}.csv", rows)

    overview_payload = {
        "variable": variable_name,
        "runs": runs,
        "generated_at": datetime.now().isoformat(),
    }
    _write_json(output_dir / "summary.json", overview_payload)


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
    values_section = variable.get("values")
    if not _as_list(values_section):
        raise ValueError("variable.values must contain at least one value.")

    metrics = cfg.metrics
    if "final" not in metrics or not metrics.final:
        raise ValueError("metrics.final must list at least one metric.")
    if "best" in metrics:
        for entry in metrics.best:
            if "name" not in entry or "mode" not in entry:
                raise ValueError("Each metrics.best entry must define 'name' and 'mode'.")


def _load_config_any(path: Path) -> DictConfig:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_ablation_config(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return OmegaConf.create(payload)
    raise ValueError(f"Unsupported config extension for {path}")


def _path_candidates(root: Path) -> Iterable[Path]:
    names = ("ablation_config_resolved.json", "ablation_config.yaml", "ablation_config.yml")
    for name in names:
        candidate = root / name
        if candidate.exists():
            yield candidate


def _load_final_metrics(
    run_dir: Path,
    metric_names: Iterable[str],
    *,
    verbose: bool = False,
) -> Dict[str, Optional[float]]:
    final_path = run_dir / "final_metrics.json"
    metrics: Dict[str, Optional[float]] = {}
    if final_path.exists():
        with final_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key, value in data.items():
            metrics[str(key)] = _to_float(value)
    elif verbose:
        print(f"[recover] Missing final_metrics.json in {run_dir}")  # noqa: T201
    history_path = run_dir / "metrics_history.jsonl"
    if not metrics and history_path.exists():
        last_row: Optional[Dict[str, Any]] = None
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                last_row = json.loads(line)
        if last_row:
            for name in metric_names:
                value = last_row.get(name)
                if value is None:
                    alt = _sanitize_metric_name(name)
                    value = last_row.get(alt)
                metrics[name] = _to_float(value)
    elif not metrics and verbose:
        print(f"[recover] No metrics found for {run_dir}")  # noqa: T201
    return metrics


def _load_best_metrics(
    run_dir: Path,
    metric_specs: Iterable[MetricSpec],
    history_path: Path,
) -> Dict[str, Optional[Dict[str, Any]]]:
    best_path = run_dir / "best_metrics.json"
    if best_path.exists():
        with best_path.open("r", encoding="utf-8") as handle:
            data_raw = json.load(handle)
        best: Dict[str, Optional[Dict[str, Any]]] = {}
        for key, value in data_raw.items():
            if value is None:
                best[str(key)] = None
                continue
            best[str(key)] = {
                "epoch": int(value.get("epoch", 0)),
                "value": _to_float(value.get("value")),
            }
        return best

    best: Dict[str, Optional[Dict[str, Any]]] = {spec.name: None for spec in metric_specs}
    if not history_path.exists():
        return best

    trackers: Dict[str, Dict[str, Any]] = {}
    for spec in metric_specs:
        trackers[spec.name] = {"mode": spec.mode, "value": None, "epoch": None}

    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            epoch = row.get("epoch")
            for spec in metric_specs:
                tracker = trackers[spec.name]
                value = row.get(spec.name)
                if value is None:
                    alt = _sanitize_metric_name(spec.name)
                    value = row.get(alt)
                value_float = _to_float(value)
                if value_float is None:
                    continue
                current = tracker["value"]
                if current is None:
                    tracker["value"] = value_float
                    tracker["epoch"] = epoch
                    continue
                if spec.mode == "min" and value_float < current:
                    tracker["value"] = value_float
                    tracker["epoch"] = epoch
                elif spec.mode == "max" and value_float > current:
                    tracker["value"] = value_float
                    tracker["epoch"] = epoch

    for spec in metric_specs:
        tracker = trackers[spec.name]
        if tracker["value"] is None:
            best[spec.name] = None
        else:
            best[spec.name] = {
                "epoch": None if tracker["epoch"] is None else int(tracker["epoch"]),
                "value": tracker["value"],
            }
    return best


def _build_runs(
    output_root: Path,
    override_key: str,
    values: List[Any],
    final_metric_names: List[str],
    best_specs: List[MetricSpec],
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for idx, raw_value in enumerate(values):
        normalized = _normalize_value(raw_value)
        fallback_label = f"value_{idx + 1:02d}"
        value_label = _format_value_for_path(normalized, fallback=fallback_label)
        run_dir = output_root / f"{override_key.replace('.', '_')}_{value_label}"
        if not run_dir.exists():
            if verbose:
                print(f"[recover] Skip missing run directory {run_dir}")  # noqa: T201
            continue
        final_metrics = _load_final_metrics(run_dir, final_metric_names, verbose=verbose)
        if not final_metrics:
            if verbose:
                print(f"[recover] Skip {run_dir} (no metrics)")  # noqa: T201
            continue

        history_path = run_dir / "metrics_history.jsonl"
        best_metrics = _load_best_metrics(run_dir, best_specs, history_path)

        training_dir_path = run_dir / "training_run_dir.txt"
        training_dir = None
        if training_dir_path.exists():
            training_dir = training_dir_path.read_text(encoding="utf-8").strip() or None

        history_rel = None
        if history_path.exists():
            try:
                history_rel = str(history_path.relative_to(output_root))
            except ValueError:
                history_rel = str(history_path)

        runs.append(
            {
                "value": _json_safe(normalized),
                "value_display": _value_to_display(normalized),
                "value_label": value_label,
                "final_metrics": final_metrics,
                "best_metrics": best_metrics,
                "training_dir": training_dir,
                "history_path": history_rel,
            }
        )
    return runs


def _infer_value_from_resolved(run_dir: Path, override_key: str) -> Any:
    config_path = run_dir / "resolved_training_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    current: Any = data
    for part in override_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _build_runs_from_disk(
    output_root: Path,
    override_key: str,
    final_metric_names: List[str],
    best_specs: List[MetricSpec],
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    prefix = f"{override_key.replace('.', '_')}_"
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if prefix and not run_dir.name.startswith(prefix):
            continue

        value_label = run_dir.name[len(prefix) :] or run_dir.name
        final_metrics = _load_final_metrics(run_dir, final_metric_names, verbose=verbose)
        if not final_metrics:
            if verbose:
                print(f"[recover] Disk-scan skip {run_dir} (no metrics)")  # noqa: T201
            continue

        best_metrics = _load_best_metrics(run_dir, best_specs, run_dir / "metrics_history.jsonl")
        override_value = _infer_value_from_resolved(run_dir, override_key)
        normalized_value = _normalize_value(override_value) if override_value is not None else value_label
        value_display = _value_to_display(normalized_value)
        training_dir_path = run_dir / "training_run_dir.txt"
        training_dir = None
        if training_dir_path.exists():
            training_dir = training_dir_path.read_text(encoding="utf-8").strip() or None
        history_path = run_dir / "metrics_history.jsonl"
        history_rel = None
        if history_path.exists():
            try:
                history_rel = str(history_path.relative_to(output_root))
            except ValueError:
                history_rel = str(history_path)

        runs.append(
            {
                "value": _json_safe(normalized_value),
                "value_display": value_display,
                "value_label": value_label,
                "final_metrics": final_metrics,
                "best_metrics": best_metrics,
                "training_dir": training_dir,
                "history_path": history_rel,
            }
        )
    return runs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate final_metrics.csv and best_* tables for an ablation output directory."
    )
    parser.add_argument(
        "output_root",
        help=(
            "Path to the ablation output directory (contains per-value subfolders). "
            "Passing the generated final_metrics.csv is also supported; the parent directory"
            " will be used automatically."
        ),
    )
    parser.add_argument(
        "--config",
        help="Optional path to the ablation config (defaults to ablation_config_resolved.json inside output root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which runs would be aggregated without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about discovered runs and skips.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.is_file():
        if output_root.suffix.lower() == ".csv":
            output_root = output_root.parent
        else:
            raise FileNotFoundError(
                f"The provided path {output_root} is a file. Specify the ablation output directory"
                " or its final_metrics.csv file."
            )
    if not output_root.exists():
        raise FileNotFoundError(f"Output root {output_root} does not exist.")

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if config_path is None:
        candidates = list(_path_candidates(output_root))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find an ablation config in {output_root}. "
                "Pass --config to specify the source ablation YAML/JSON."
            )
        config_path = candidates[0]

    cfg = _load_config_any(config_path)
    validate_ablation_config(cfg)

    override_key = cfg.variable.override
    values = _as_list(cfg.variable.get("values"))
    final_metric_names = _as_list(cfg.metrics.final)
    best_specs = [MetricSpec(name=entry.name, mode=entry.mode) for entry in _as_list(cfg.metrics.get("best", []))]

    runs = _build_runs(
        output_root,
        override_key,
        values,
        final_metric_names,
        best_specs,
        verbose=args.verbose,
    )
    if not runs:
        if args.verbose:
            print("[recover] Falling back to disk scan for available runs.")  # noqa: T201
        runs = _build_runs_from_disk(
            output_root,
            override_key,
            final_metric_names,
            best_specs,
            verbose=args.verbose,
        )
    if not runs:
        raise RuntimeError("No completed runs with recorded metrics were found; nothing to aggregate.")

    if args.dry_run:
        print(f"Would aggregate {len(runs)} runs in {output_root}")  # noqa: T201
        for run in runs:
            print(f"  - {run['value_label']}: metrics={list(run['final_metrics'].keys())}")  # noqa: T201
        return

    aggregate_and_save_tables(runs, output_root, final_metric_names, best_specs, override_key)


if __name__ == "__main__":
    main(sys.argv[1:])
