import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig


def _as_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in list(value)]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_data_sources_for_cache(value: Any) -> list[dict[str, Any]]:
    from omegaconf import OmegaConf

    if value is None:
        return []
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        raise ValueError(
            "cfg.data.data_sources must be a list when present, "
            f"got {type(value)!r}."
        )
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(value):
        if hasattr(entry, "keys"):
            entry_dict = dict(entry)
        elif isinstance(entry, dict):
            entry_dict = dict(entry)
        else:
            raise TypeError(
                "cfg.data.data_sources entries must be mapping-like for inference cache spec, "
                f"got entry index {idx} with type {type(entry)!r}."
            )
        # Synthetic format: file / label / n_samples / max_points
        if "file" in entry_dict:
            normalized.append(
                {
                    "file": str(entry_dict["file"]),
                    "label": (
                        None if entry_dict.get("label") is None else int(entry_dict["label"])
                    ),
                    "n_samples": (
                        None
                        if entry_dict.get("n_samples") is None
                        else int(entry_dict["n_samples"])
                    ),
                    "max_points": (
                        None
                        if entry_dict.get("max_points") is None
                        else int(entry_dict["max_points"])
                    ),
                }
            )
        # Real-data format: name / data_path / data_files
        else:
            files = _as_list_of_str(entry_dict.get("data_files"))
            normalized.append(
                {
                    "name": None if entry_dict.get("name") is None else str(entry_dict["name"]),
                    "data_path": str(entry_dict.get("data_path", "")),
                    "data_files": files or [],
                    "radius": None if entry_dict.get("radius") is None else float(entry_dict["radius"]),
                }
            )
    return normalized


def _build_inference_cache_spec(
    *,
    checkpoint_path: str,
    cfg: DictConfig,
    max_batches_latent: int | None,
    max_samples_total: int | None,
    seed_base: int,
    temporal_real_selection: dict[str, Any] | None = None,
    temporal_sequence_inference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "data_kind": str(getattr(cfg.data, "kind", "unknown")),
        "data_path": str(getattr(cfg.data, "data_path", "")),
        "data_files": _as_list_of_str(getattr(cfg.data, "data_files", None)) or [],
        "data_sources": _normalize_data_sources_for_cache(
            getattr(cfg.data, "data_sources", None)
        ),
        "data_radius": _optional_float(getattr(cfg.data, "radius", None)),
        "data_sample_type": str(getattr(cfg.data, "sample_type", "")),
        "data_overlap_fraction": float(getattr(cfg.data, "overlap_fraction", 0.0)),
        "data_n_samples": int(getattr(cfg.data, "n_samples", 0)),
        "data_num_points": int(getattr(cfg.data, "num_points", 0)),
        "data_drop_edge_samples": bool(getattr(cfg.data, "drop_edge_samples", True)),
        "data_edge_drop_layers": getattr(cfg.data, "edge_drop_layers", None),
        "batch_size": int(getattr(cfg, "batch_size", 0)),
        "max_batches_latent": None if max_batches_latent is None else int(max_batches_latent),
        "max_samples_total": None if max_samples_total is None else int(max_samples_total),
        "seed_base": int(seed_base),
        "collect_coords": True,
        "temporal_real_selection": temporal_real_selection,
        "temporal_sequence_inference": temporal_sequence_inference,
    }


def _inference_cache_spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inference_cache_paths(out_dir: Path, cache_filename: str) -> tuple[Path, Path]:
    npz_path = Path(out_dir) / cache_filename
    meta_path = npz_path.with_suffix(npz_path.suffix + ".meta.json")
    return npz_path, meta_path


def _validate_inference_cache_arrays(cache: dict[str, np.ndarray]) -> None:
    required = ("inv_latents", "eq_latents", "phases", "coords", "instance_ids")
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"Inference cache missing arrays: {missing}")
    inv_latents = np.asarray(cache["inv_latents"])
    if inv_latents.ndim < 2:
        raise ValueError(
            "Inference cache 'inv_latents' must have shape [num_samples, latent_dim], "
            f"got shape={tuple(inv_latents.shape)}."
        )
    num_samples = int(inv_latents.shape[0])

    coords = np.asarray(cache["coords"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "Inference cache 'coords' must have shape [num_samples, 3], "
            f"got shape={tuple(coords.shape)}."
        )
    if coords.shape[0] != num_samples:
        raise ValueError(
            "Inference cache sample mismatch between 'inv_latents' and 'coords': "
            f"{num_samples} vs {coords.shape[0]}. "
            f"All cache shapes: {{'inv_latents': {tuple(inv_latents.shape)}, "
            f"'coords': {tuple(coords.shape)}, "
            f"'eq_latents': {tuple(np.asarray(cache['eq_latents']).shape)}, "
            f"'phases': {tuple(np.asarray(cache['phases']).shape)}, "
            f"'instance_ids': {tuple(np.asarray(cache['instance_ids']).shape)}}}."
        )

    for key in ("eq_latents", "phases", "instance_ids"):
        arr = np.asarray(cache[key])
        if arr.size == 0:
            continue
        if arr.shape[0] != num_samples:
            raise ValueError(
                "Inference cache sample mismatch: "
                f"'inv_latents' has {num_samples} rows but '{key}' has shape={tuple(arr.shape)}."
            )


def _load_inference_cache(
    *,
    out_dir: Path,
    cache_filename: str,
    expected_spec: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, str]:
    npz_path, meta_path = _inference_cache_paths(out_dir, cache_filename)
    if not npz_path.exists():
        return None, f"cache file does not exist: {npz_path}"
    if not meta_path.exists():
        return None, f"cache metadata does not exist: {meta_path}"

    with meta_path.open("r") as handle:
        meta = json.load(handle)
    if not isinstance(meta, dict):
        raise ValueError(
            f"Cache metadata at {meta_path} must be a JSON object, got {type(meta)!r}."
        )

    expected_hash = _inference_cache_spec_hash(expected_spec)
    cached_hash = str(meta.get("spec_sha256", ""))
    if cached_hash != expected_hash:
        return None, (
            "cache spec mismatch: "
            f"expected sha256={expected_hash}, found sha256={cached_hash}"
        )

    with np.load(npz_path) as data:
        cache = {key: np.asarray(data[key]) for key in data.files}
    try:
        _validate_inference_cache_arrays(cache)
    except ValueError as exc:
        return None, f"cache validation failed for {npz_path}: {exc}"
    return cache, f"loaded cache from {npz_path}"


def _save_inference_cache(
    *,
    out_dir: Path,
    cache_filename: str,
    cache: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> None:
    _validate_inference_cache_arrays(cache)
    npz_path, meta_path = _inference_cache_paths(out_dir, cache_filename)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        npz_path,
        inv_latents=cache["inv_latents"],
        eq_latents=cache["eq_latents"],
        phases=cache["phases"],
        coords=cache["coords"],
        instance_ids=cache["instance_ids"],
    )
    meta = {
        "spec": spec,
        "spec_sha256": _inference_cache_spec_hash(spec),
        "num_samples": int(cache["inv_latents"].shape[0]) if cache["inv_latents"].ndim >= 1 else 0,
    }
    with meta_path.open("w") as handle:
        json.dump(meta, handle, indent=2)
