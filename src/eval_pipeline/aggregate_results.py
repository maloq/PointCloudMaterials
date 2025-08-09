from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _find_eval_result_files(root_dir: str) -> List[str]:
    """Return a list of absolute paths to all eval_results.yaml files under root_dir."""
    matches: List[str] = []
    for current_dir, _subdirs, files in os.walk(root_dir):
        for fname in files:
            if fname == "eval_results.yaml":
                matches.append(os.path.join(current_dir, fname))
    return sorted(matches)


def _flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested mapping using dot-separated keys.

    Non-mapping values are left as-is.
    """
    items: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def _safe_get(container: Mapping[str, Any], dotted_key: str) -> Any:
    """Retrieve value from possibly nested dict via dotted key, returning None if missing."""
    cursor: Any = container
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _row_from_cfg(cfg: DictConfig | Mapping[str, Any], *, config_keys: Sequence[str], include_eval_results: bool, run_dir: str) -> Dict[str, Any]:
    """Build a single table row from one merged eval YAML content."""
    cfg_dict: Dict[str, Any] = (
        OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    )

    row: Dict[str, Any] = {"run_dir": run_dir}

    # Extract requested config keys (top-level or nested via dotted path)
    for key in config_keys:
        row[key] = _safe_get(cfg_dict, key)

    # Add flattened eval_results
    if include_eval_results:
        eval_results = cfg_dict.get("eval_results", {}) or {}
        if isinstance(eval_results, Mapping):
            row.update(_flatten_dict(eval_results, parent_key="eval_results"))
        else:
            # In case eval_results is unexpectedly not a mapping
            row["eval_results"] = eval_results

    return row


def aggregate_eval_results(
    root_dir: str = "output/eval_results",
    *,
    config_keys: Sequence[str] | None = None,
    include_eval_results: bool = True,
    extra_columns: Mapping[str, Any] | None = None,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Crawl all runs under root_dir and build a comparison table.

    - config_keys: dotted keys to extract from the merged config. Defaults to
      ["model_type", "experiment_name"]. You can include nested keys like
      "prediction.model_type" if desired.
    - include_eval_results: whether to append all flattened eval_results.* columns.
    - extra_columns: optional static columns to add to each row (e.g., dataset tag).
    - output_csv: if provided, save the resulting table to this CSV path.
    """
    keys = list(config_keys) if config_keys is not None else ["model_type", "experiment_name"]

    yaml_paths = _find_eval_result_files(root_dir)
    rows: List[Dict[str, Any]] = []
    for yaml_path in yaml_paths:
        try:
            cfg: DictConfig = OmegaConf.load(yaml_path)
        except Exception:
            # Skip malformed files
            continue

        run_dir = os.path.dirname(yaml_path)
        row = _row_from_cfg(
            cfg,
            config_keys=keys,
            include_eval_results=include_eval_results,
            run_dir=run_dir,
        )
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Stable column order: run_dir, requested keys, then eval_results*
    desired_prefix = ["run_dir", *keys]
    other_cols = [c for c in df.columns if c not in desired_prefix]
    df = df[[*desired_prefix, *other_cols]] if not df.empty else df

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df


if __name__ == "__main__":  # Convenience local run
    table = aggregate_eval_results(
        root_dir="output/eval_results",
        config_keys=["model_type", "experiment_name"],
        include_eval_results=True,
        output_csv="output/eval_results_summary.csv",
    )
    print(f"Aggregated {len(table)} runs -> output/eval_results_summary.csv")


