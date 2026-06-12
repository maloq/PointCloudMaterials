import hashlib
import json
from pathlib import Path
import threading
from typing import Any
import zipfile

import numpy as np
from omegaconf import DictConfig

from src.data_utils.data_kinds import normalize_data_kind
from .output_layout import write_json


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
        # Static-data format: name / data_path / data_files
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
    inference_batch_size: int,
    max_batches_latent: int | None,
    max_samples_total: int | None,
    seed_base: int,
    temporal_real_selection: dict[str, Any] | None = None,
    temporal_sequence_inference: dict[str, Any] | None = None,
    collector_mode: str = "generic",
) -> dict[str, Any]:
    return {
        "version": 6,
        "collector_contract": {
            "latent_array": "model_forward_output_0_z_inv_contrastive",
            "sample_order": "dataloader_order_preallocated_v3",
            "encoder_group_sampling": "deterministic_fps_for_analysis_v1",
            "temporal_input": "static_anchor_or_full_sequence_v1",
            "coords": "center_positions_for_static_anchor_v1",
            "temporal_anchor_metadata": "anchor_frame_indices_per_sample_v1",
            "cpu_transfer": "blocking_cpu_copy_for_numpy_cache_v2",
        },
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "model_type": str(getattr(cfg, "model_type", "")),
        "data_kind": normalize_data_kind(getattr(cfg.data, "kind", None), default="unknown"),
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
        "checkpoint_batch_size": int(getattr(cfg, "batch_size", 0)),
        "inference_batch_size": int(inference_batch_size),
        "max_batches_latent": None if max_batches_latent is None else int(max_batches_latent),
        "max_samples_total": None if max_samples_total is None else int(max_samples_total),
        "seed_base": int(seed_base),
        "rng_strategy": "seed_once_per_collection_v1",
        "collect_coords": True,
        "collector_mode": str(collector_mode),
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
    _validate_invariant_latent_values(inv_latents)

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
    optional_sample_keys = [
        key
        for key in cache.keys()
        if key not in required and key not in {"spec", "meta"}
    ]
    for key in optional_sample_keys:
        arr = np.asarray(cache[key])
        if arr.size == 0:
            continue
        if arr.ndim == 0:
            raise ValueError(
                "Inference cache dynamic arrays must remain per-sample arrays. "
                f"Key {key!r} was saved as a scalar."
            )
        if arr.shape[0] != num_samples:
            raise ValueError(
                "Inference cache sample mismatch in optional array: "
                f"'inv_latents' has {num_samples} rows but '{key}' has shape={tuple(arr.shape)}."
            )


def _validate_invariant_latent_values(inv_latents: np.ndarray) -> None:
    arr = np.asarray(inv_latents)
    chunk_size = 65536
    zero_rows = 0
    first_zero_row: int | None = None
    first_nonfinite_row: int | None = None
    for start in range(0, int(arr.shape[0]), chunk_size):
        end = min(start + chunk_size, int(arr.shape[0]))
        chunk = np.asarray(arr[start:end], dtype=np.float32)
        finite_rows = np.isfinite(chunk).all(axis=1)
        if not np.all(finite_rows):
            bad = int(np.flatnonzero(~finite_rows)[0])
            first_nonfinite_row = start + bad
            break
        norms = np.linalg.norm(chunk.reshape(chunk.shape[0], -1), axis=1)
        zero_mask = norms <= 1.0e-8
        if np.any(zero_mask):
            zero_rows += int(np.count_nonzero(zero_mask))
            if first_zero_row is None:
                first_zero_row = start + int(np.flatnonzero(zero_mask)[0])

    if first_nonfinite_row is not None:
        raise ValueError(
            "Inference cache 'inv_latents' contains non-finite values. "
            f"first_bad_row={first_nonfinite_row}, shape={tuple(arr.shape)}. "
            "Delete the cache and rerun inference."
        )
    if zero_rows > 0:
        raise ValueError(
            "Inference cache 'inv_latents' contains zero-norm rows. "
            f"zero_rows={zero_rows}, first_zero_row={first_zero_row}, "
            f"shape={tuple(arr.shape)}. This usually means the cache was written "
            "from an incomplete async GPU-to-CPU transfer; delete the cache and rerun inference."
        )


def _normalize_spec_for_compatibility(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec
    normalized = dict(spec)
    normalized.pop("dynamic_motif", None)
    if "data_kind" in normalized:
        normalized["data_kind"] = normalize_data_kind(
            normalized.get("data_kind"),
            default="unknown",
        )
    return normalized


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
        cached_spec = meta.get("spec")
        if _normalize_spec_for_compatibility(cached_spec) != _normalize_spec_for_compatibility(expected_spec):
            return None, (
                "cache spec mismatch: "
                f"expected sha256={expected_hash}, found sha256={cached_hash}"
            )

    try:
        with np.load(npz_path) as data:
            cache = {key: np.asarray(data[key]) for key in data.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        return None, (
            f"cache file is unreadable and will be recomputed: {npz_path}. "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        _validate_inference_cache_arrays(cache)
    except ValueError as exc:
        return None, f"cache validation failed for {npz_path}: {exc}"
    if cached_hash == expected_hash:
        return cache, f"loaded cache from {npz_path}"
    return cache, (
        f"loaded legacy-compatible cache from {npz_path}; "
        "optional dynamic motif arrays may be missing."
    )


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
    tmp_npz_path = npz_path.with_suffix(npz_path.suffix + ".tmp.npz")
    tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    # First-run analysis is dominated by inference and large cache writes; compression
    # saves disk but costs substantial CPU time for latents/coords that are read locally.
    _save_npz_with_progress(
        tmp_npz_path,
        {key: np.asarray(value) for key, value in cache.items()},
    )
    meta = {
        "spec": spec,
        "spec_sha256": _inference_cache_spec_hash(spec),
        "num_samples": int(cache["inv_latents"].shape[0]) if cache["inv_latents"].ndim >= 1 else 0,
        "storage": "npz_uncompressed",
    }
    write_json(tmp_meta_path, meta)
    tmp_npz_path.replace(npz_path)
    tmp_meta_path.replace(meta_path)


def _save_npz_with_progress(path: Path, arrays: dict[str, np.ndarray]) -> None:
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(30.0):
            print(f"[analysis][cache] Still writing {path.name}...", flush=True)

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        np.savez(path, **arrays)
    finally:
        stop_event.set()
        heartbeat.join(timeout=1.0)
