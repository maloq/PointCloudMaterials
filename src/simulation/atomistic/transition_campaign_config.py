from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import PHASE_TO_ID
from .config import GeneratorConfig, REPOSITORY_ROOT, load_config
from .transition_config import TransitionConfig, load_transition_config
from .transition_generator import PreparedInterface, _load_prepared_interface


RUNTIME_POTENTIAL_FIELDS = {
    "enable_cueq",
    "enable_oeq",
    "compile_mode",
    "compile_fullgraph",
    "pad_num_atoms",
    "pad_num_edges",
    "neighbor_skin_A",
}


@dataclass(frozen=True)
class TransitionCampaignExecutionConfig:
    chunk_steps: int
    checkpoint_retention: int


@dataclass(frozen=True)
class TransitionCampaignConfig:
    transition: TransitionConfig
    runtime_generator: GeneratorConfig
    source_evidence: dict[str, object]
    execution: TransitionCampaignExecutionConfig
    config_path: Path

    @property
    def output_root(self) -> Path:
        return self.transition.output.root_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_config": str(self.transition.config_path),
            "transition": self.transition.to_dict(),
            "runtime_generator_config": str(self.runtime_generator.config_path),
            "runtime_generator": self.runtime_generator.to_dict(),
            "source_evidence": self.source_evidence,
            "execution": asdict(self.execution),
            "config_path": str(self.config_path),
        }


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _positive_int(value: Any, *, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"{path}: {context} must be a positive integer, got {value!r}."
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def transition_source_evidence(
    transition: TransitionConfig,
    *,
    prepared: PreparedInterface | None = None,
) -> tuple[PreparedInterface, dict[str, object]]:
    source_root = transition.source_dataset
    relative_paths = (
        Path("manifest.json"),
        Path(transition.source_interface_environment) / "metadata.json",
        Path(transition.source_interface_environment) / "trajectory.npz",
    )
    paths = tuple(source_root / relative for relative in relative_paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Transition campaign cannot bind its prepared source because required "
            f"files are missing: {missing}."
        )
    before = {
        relative.as_posix(): _sha256_file(path)
        for relative, path in zip(relative_paths, paths)
    }
    loaded = _load_prepared_interface(transition) if prepared is None else prepared
    after = {
        relative.as_posix(): _sha256_file(path)
        for relative, path in zip(relative_paths, paths)
    }
    if after != before:
        raise RuntimeError(
            "Transition source files changed while the prepared frame was being loaded; "
            f"before={before}, after={after}. Retry only after the source is stable."
        )
    phase_ids = np.fromiter(
        (PHASE_TO_ID[str(name)] for name in loaded.labels.phase_names),
        dtype=np.int16,
        count=len(loaded.atoms),
    )
    evidence: dict[str, object] = {
        "schema_version": 1,
        "files_sha256": before,
        "selected_frame_step": transition.source_frame_step,
        "atom_count": len(loaded.atoms),
        "positions_A_sha256": _sha256_array(
            np.asarray(loaded.atoms.positions, dtype=np.float64)
        ),
        "cell_vectors_A_sha256": _sha256_array(
            np.asarray(loaded.atoms.cell.array, dtype=np.float64)
        ),
        "slab_bounds_fractional": list(loaded.slab_bounds_fractional),
        "prepared_phase_ids_sha256": _sha256_array(phase_ids),
    }
    return loaded, evidence


def load_content_bound_prepared_interface(
    config: TransitionCampaignConfig,
) -> PreparedInterface:
    prepared, observed = transition_source_evidence(config.transition)
    if observed != config.source_evidence:
        raise RuntimeError(
            "The prepared transition source differs from the content-bound campaign "
            f"identity. expected={config.source_evidence}, observed={observed}."
        )
    return prepared


def _validate_runtime_generator(
    source: GeneratorConfig,
    runtime: GeneratorConfig,
    *,
    path: Path,
) -> None:
    scientific_mismatches: dict[str, object] = {}
    for field in ("dynamics", "system", "validation"):
        source_value = asdict(getattr(source, field))
        runtime_value = asdict(getattr(runtime, field))
        if source_value != runtime_value:
            scientific_mismatches[field] = {
                "source": source_value,
                "runtime": runtime_value,
            }

    source_potential = asdict(source.potential)
    runtime_potential = asdict(runtime.potential)
    potential_mismatches = {
        field: {
            "source": source_potential[field],
            "runtime": runtime_potential[field],
        }
        for field in source_potential
        if field not in RUNTIME_POTENTIAL_FIELDS
        and source_potential[field] != runtime_potential[field]
    }
    if potential_mismatches:
        scientific_mismatches["potential"] = potential_mismatches
    if scientific_mismatches:
        raise RuntimeError(
            f"{path}: runtime_generator_config changes source-bound scientific or model "
            f"fields: {scientific_mismatches}. Only {sorted(RUNTIME_POTENTIAL_FIELDS)} "
            "may differ. Use a byte-identical model/head/dtype and the exact transition "
            "dynamics, system, and validation protocol."
        )


def load_transition_campaign_config(path: str | Path) -> TransitionCampaignConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping.")
    expected_keys = {"transition_config", "runtime_generator_config", "execution"}
    if set(raw) != expected_keys:
        raise KeyError(
            f"{config_path}: root keys must be exactly {sorted(expected_keys)}, got "
            f"{sorted(raw)}."
        )
    execution_raw = raw["execution"]
    if not isinstance(execution_raw, dict):
        raise TypeError(
            f"{config_path}: execution must be a mapping, got "
            f"{type(execution_raw).__name__}."
        )
    expected_execution = {"chunk_steps", "checkpoint_retention"}
    if set(execution_raw) != expected_execution:
        raise KeyError(
            f"{config_path}: execution keys must be exactly "
            f"{sorted(expected_execution)}, got {sorted(execution_raw)}."
        )
    transition = load_transition_config(_repo_path(raw["transition_config"]))
    runtime_generator = load_config(_repo_path(raw["runtime_generator_config"]))
    _validate_runtime_generator(
        transition.generator, runtime_generator, path=config_path
    )
    chunk_steps = _positive_int(
        execution_raw["chunk_steps"], context="execution.chunk_steps", path=config_path
    )
    if chunk_steps % transition.sample_interval:
        raise ValueError(
            f"{config_path}: execution.chunk_steps={chunk_steps} must be divisible by "
            f"transition sample_interval={transition.sample_interval}. Checkpoint "
            "boundaries must coincide with stored trajectory frames."
        )
    if transition.output.overwrite:
        raise ValueError(
            f"{config_path}: the queued campaign requires transition output.overwrite=false. "
            "Committed tasks are resumed and content-validated, never replaced implicitly."
        )
    _, source_evidence = transition_source_evidence(transition)
    return TransitionCampaignConfig(
        transition=transition,
        runtime_generator=runtime_generator,
        source_evidence=source_evidence,
        execution=TransitionCampaignExecutionConfig(
            chunk_steps=chunk_steps,
            checkpoint_retention=_positive_int(
                execution_raw["checkpoint_retention"],
                context="execution.checkpoint_retention",
                path=config_path,
            ),
        ),
        config_path=config_path,
    )
