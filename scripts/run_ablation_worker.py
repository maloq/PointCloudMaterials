#!/usr/bin/env python3
"""Run a single ablation experiment value (intended for SLURM srun execution)."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ablation import (  # noqa: E402
    _as_list,
    _format_value_for_path,
    _normalize_value,
    compose_training_config,
    load_ablation_config,
    run_single_experiment,
    validate_ablation_config,
)


def _ensure_relative(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path} must be inside {root}") from exc


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a single ablation experiment value.")
    parser.add_argument("--config", required=True, help="Path to the ablation YAML.")
    parser.add_argument("--value-index", type=int, required=True, help="Index into variable.values.")
    parser.add_argument("--override-key", help="Override key (defaults to variable.override in config).")
    parser.add_argument("--output-root", required=True, help="Common ablation output directory.")
    parser.add_argument("--run-output-dir", required=True, help="Directory to store metrics for this value.")
    parser.add_argument("--run-name", required=True, help="Base run name used for experiment naming.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without training.")
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    ablation_cfg = load_ablation_config(config_path)
    validate_ablation_config(ablation_cfg)

    override_key = args.override_key or ablation_cfg.variable.override
    values = _as_list(ablation_cfg.variable.values)
    if args.value_index < 0 or args.value_index >= len(values):
        raise IndexError(f"value-index {args.value_index} out of range (0..{len(values)-1}).")
    value = values[args.value_index]
    value_serializable = _normalize_value(value)
    fallback_label = f"value_{args.value_index + 1:02d}"
    value_label = _format_value_for_path(value_serializable, fallback=fallback_label)

    resolved_output_root = Path(args.output_root).expanduser().resolve()
    resolved_run_dir = Path(args.run_output_dir).expanduser().resolve()
    _ensure_relative(resolved_run_dir, resolved_output_root)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ablation_cfg.experiment
    training_cfg = compose_training_config(
        exp_cfg,
        override_key,
        value,
        run_name=args.run_name,
        value_label=value_label,
    )

    if args.dry_run:
        from omegaconf import OmegaConf  # lazy import

        print(OmegaConf.to_yaml(training_cfg))  # noqa: T201
        return

    run_single_experiment(training_cfg, ablation_cfg.metrics, resolved_run_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
