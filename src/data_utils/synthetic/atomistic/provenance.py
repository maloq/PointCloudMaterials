from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import ase
import numpy as np
import torch
from ase.calculators.calculator import Calculator

if TYPE_CHECKING:
    from .config import GeneratorConfig

from .config import (
    DEFAULT_COMPILE_FULLGRAPH,
    DEFAULT_COMPILE_MODE,
    DEFAULT_ENABLE_OEQ,
    DEFAULT_MD_PROPERTY_MODE,
    DEFAULT_PAD_NUM_ATOMS,
    DEFAULT_PAD_NUM_EDGES,
    REPOSITORY_ROOT,
    mace_kernel_backend,
    potential_calculator_settings,
)


ATOMISTIC_PACKAGE_ROOT = Path(__file__).resolve().parent
PRODUCER_COMPATIBILITY_PATH = (
    REPOSITORY_ROOT
    / "configs/simulation/atomistic/al/producer_compatibility.json"
)
HOMOGENEOUS_LIQUID_SOURCE_PRODUCER_FILES = (
    "artifacts.py",
    "calculator.py",
    "checkpoints.py",
    "config.py",
    "generator.py",
    "homogeneous_liquid_source.py",
    "provenance.py",
    "simulation.py",
    "validation.py",
)


@dataclass(frozen=True)
class CalculatorProvenance:
    source: str
    identity: str
    implementation_class: str
    model_name: str | None
    family: str | None
    model_path: str | None
    model_sha256: str | None
    head: str | None
    available_heads: tuple[str, ...]
    source_url: str | None
    license_identifier: str | None
    usage_mode: str
    validation_report_path: str | None
    validation_report_sha256: str | None
    validation_report_type: str | None
    scientifically_qualified: bool
    qualification_scope: dict[str, object] | None
    settings: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["available_heads"] = list(self.available_heads)
        return result


@dataclass(frozen=True)
class ExecutionProvenance:
    calculator: CalculatorProvenance
    runtime: dict[str, object]
    producer_code: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "calculator": self.calculator.to_dict(),
            "runtime": self.runtime,
            "producer_code": self.producer_code,
        }


def _implementation_class(instance: object) -> str:
    cls = type(instance)
    return f"{cls.__module__}.{cls.__qualname__}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualification_scope_dict(config: GeneratorConfig) -> dict[str, object] | None:
    qualification = config.potential.qualification
    if qualification is None:
        return None
    return {
        "chemical_symbol": qualification.chemical_symbol,
        "pressure_range_GPa": list(qualification.pressure_range_GPa),
        "state_temperature_ranges_K": {
            state: list(temperature_range)
            for state, temperature_range in qualification.state_temperature_ranges_K.items()
        },
        "maximum_timestep_fs": qualification.maximum_timestep_fs,
        "authorized_claims": dict(qualification.authorized_claims),
        "calculator_settings": {
            "implementation_class": qualification.implementation_class,
            "device": qualification.device,
            "default_dtype": qualification.default_dtype,
            "kernel_backend": mace_kernel_backend(
                enable_cueq=qualification.enable_cueq,
                enable_oeq=qualification.enable_oeq,
            ),
            "enable_cueq": qualification.enable_cueq,
            "enable_oeq": qualification.enable_oeq,
            "compile_mode": qualification.compile_mode,
            "compile_fullgraph": qualification.compile_fullgraph,
            "pad_num_atoms": qualification.pad_num_atoms,
            "pad_num_edges": qualification.pad_num_edges,
            "md_property_mode": qualification.md_property_mode,
            "neighbor_skin_A": qualification.neighbor_skin_A,
        },
    }


def configured_mace_provenance(
    config: GeneratorConfig, calculator: object
) -> CalculatorProvenance:
    available_heads = tuple(str(head) for head in calculator.available_heads)
    selected_head = str(calculator.head)
    implementation_class = _implementation_class(calculator)
    qualification = config.potential.qualification
    if (
        qualification is not None
        and qualification.implementation_class != implementation_class
    ):
        raise RuntimeError(
            f"Potential qualification requires calculator implementation="
            f"{qualification.implementation_class!r}, but the loaded calculator is "
            f"{implementation_class!r}."
        )
    validation_report = config.potential.validation_report
    validation_report_sha256 = None
    if validation_report is not None:
        validation_report_sha256 = _sha256_file(validation_report)
        if validation_report_sha256 != config.potential.validation_report_sha256:
            raise RuntimeError(
                f"Potential validation report changed after configuration load: path="
                f"{validation_report}, loaded_sha256="
                f"{config.potential.validation_report_sha256}, observed_sha256="
                f"{validation_report_sha256}. Reload and revalidate the configuration."
            )
    return CalculatorProvenance(
        source="configured_mace_model",
        identity=f"{config.potential.model_name}:{config.potential.sha256}:{selected_head}",
        implementation_class=implementation_class,
        model_name=config.potential.model_name,
        family=config.potential.family,
        model_path=str(config.potential.model_path),
        model_sha256=config.potential.sha256,
        head=selected_head,
        available_heads=available_heads,
        source_url=config.potential.source_url,
        license_identifier=config.potential.license_identifier,
        usage_mode=config.potential.usage_mode,
        validation_report_path=(
            None if validation_report is None else str(validation_report)
        ),
        validation_report_sha256=validation_report_sha256,
        validation_report_type=(
            None if qualification is None else qualification.report_type
        ),
        scientifically_qualified=config.potential.scientifically_qualified,
        qualification_scope=_qualification_scope_dict(config),
        settings=potential_calculator_settings(config.potential),
    )


def injected_calculator_provenance(
    calculator: object,
    *,
    identity: str,
) -> CalculatorProvenance:
    normalized_identity = identity.strip()
    if not normalized_identity:
        raise ValueError(
            "injected_calculator_identity must be a non-empty, scientifically meaningful "
            "identifier for the actual calculator used."
        )
    if not isinstance(calculator, Calculator):
        raise TypeError(
            "Injected atomistic calculators must be ASE Calculator instances so their "
            f"Hamiltonian parameters can be recorded, got {_implementation_class(calculator)}."
        )
    ase_parameters = dict(calculator.parameters)
    try:
        json.dumps(ase_parameters, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Injected calculator {normalized_identity!r} has non-JSON ASE parameters="
            f"{ase_parameters!r}. Provide a calculator whose Hamiltonian parameters can be "
            "persisted exactly."
        ) from exc
    return CalculatorProvenance(
        source="injected_calculator",
        identity=normalized_identity,
        implementation_class=_implementation_class(calculator),
        model_name=None,
        family=None,
        model_path=None,
        model_sha256=None,
        head=None,
        available_heads=(),
        source_url=None,
        license_identifier=None,
        usage_mode="injected_unqualified",
        validation_report_path=None,
        validation_report_sha256=None,
        validation_report_type=None,
        scientifically_qualified=False,
        qualification_scope=None,
        settings={"ase_parameters": ase_parameters},
    )


def _runtime_provenance(calculator: CalculatorProvenance) -> dict[str, object]:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ase": ase.__version__,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if calculator.source == "configured_mace_model":
        result["mace_torch"] = version("mace-torch")
        result["torch_cuda"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        if bool(calculator.settings["enable_cueq"]):
            result["cuequivariance"] = version("cuequivariance")
            result["cuequivariance_torch"] = version("cuequivariance-torch")
            result["cuequivariance_ops_torch_cu12"] = version(
                "cuequivariance-ops-torch-cu12"
            )
        if bool(calculator.settings["enable_oeq"]):
            result["openequivariance"] = version("openequivariance")
        device = str(calculator.settings["device"])
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Cannot record execution provenance for device={device!r}: "
                    "torch.cuda.is_available() is false."
                )
            device_index = torch.device(device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            result["cuda_device_index"] = device_index
            result["cuda_device_name"] = torch.cuda.get_device_name(device_index)
    return result


def _producer_code_provenance(
    relative_paths: tuple[str, ...] | None = None,
) -> dict[str, object]:
    source_paths = (
        sorted(ATOMISTIC_PACKAGE_ROOT.glob("*.py"))
        if relative_paths is None
        else [ATOMISTIC_PACKAGE_ROOT / name for name in relative_paths]
    )
    if not source_paths:
        raise RuntimeError(
            f"No Python producer sources found below {ATOMISTIC_PACKAGE_ROOT}."
        )
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Producer-code scope references missing Python files: {missing}."
        )
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in source_paths:
        relative_path = path.relative_to(ATOMISTIC_PACKAGE_ROOT).as_posix()
        relative_paths.append(relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "algorithm": "sha256",
        "scope": str(ATOMISTIC_PACKAGE_ROOT),
        "files": relative_paths,
        "sha256": digest.hexdigest(),
    }


def homogeneous_liquid_source_producer_code_provenance() -> dict[str, object]:
    return _producer_code_provenance(HOMOGENEOUS_LIQUID_SOURCE_PRODUCER_FILES)


def producer_code_is_compatible(
    observed: object,
    expected: dict[str, object],
) -> bool:
    """Recognize an explicit, digest-bound exact implementation migration."""

    if observed == expected:
        return True
    required_provenance_keys = {"algorithm", "scope", "files", "sha256"}
    if not isinstance(observed, dict) or set(observed) != required_provenance_keys:
        return False
    if (
        observed["algorithm"] != "sha256"
        or expected.get("algorithm") != "sha256"
        or observed["scope"] != expected.get("scope")
        or observed["files"] != expected.get("files")
    ):
        return False
    if not PRODUCER_COMPATIBILITY_PATH.is_file():
        raise FileNotFoundError(
            "Producer code changed and the audited compatibility certificate is missing: "
            f"{PRODUCER_COMPATIBILITY_PATH}. Regenerate the artifact or restore the "
            "certificate; do not bypass provenance validation."
        )
    with PRODUCER_COMPATIBILITY_PATH.open("r", encoding="utf-8") as handle:
        certificate = json.load(handle)
    if not isinstance(certificate, dict) or set(certificate) != {
        "schema_version",
        "migrations",
    }:
        raise RuntimeError(
            f"{PRODUCER_COMPATIBILITY_PATH}: expected exactly schema_version and "
            "migrations."
        )
    if certificate["schema_version"] != 1:
        raise RuntimeError(
            f"{PRODUCER_COMPATIBILITY_PATH}: unsupported schema_version="
            f"{certificate['schema_version']!r}."
        )
    migrations = certificate["migrations"]
    if not isinstance(migrations, list):
        raise TypeError(
            f"{PRODUCER_COMPATIBILITY_PATH}: migrations must be a list."
        )
    expected_migration_keys = {
        "name",
        "files",
        "observed_sha256",
        "active_sha256",
        "changed_files",
        "equivalence_basis",
    }
    for migration in migrations:
        if not isinstance(migration, dict) or set(migration) != expected_migration_keys:
            raise RuntimeError(
                f"{PRODUCER_COMPATIBILITY_PATH}: every migration must contain exactly "
                f"{sorted(expected_migration_keys)}, got {migration!r}."
            )
        if (
            migration["files"] == observed["files"]
            and migration["observed_sha256"] == observed["sha256"]
            and migration["active_sha256"] == expected["sha256"]
        ):
            return True
    return False


def _normalized_schema4_calculator_settings(
    value: object, *, manifest_path: Path
) -> dict[str, object]:
    """Apply the explicit schema-4 defaults used before backend fields existed.

    Actual legacy artifacts still fail the exact serialized-config and producer
    code checks.  This normalization only defines the meaning of omitted fields
    for schema-4 records whose config and producer identity are otherwise current.
    """

    if not isinstance(value, dict):
        raise TypeError(
            f"{manifest_path}: execution_provenance.calculator.settings must be a "
            f"mapping, got {type(value).__name__}."
        )
    allowed = {
        "device",
        "default_dtype",
        "kernel_backend",
        "enable_cueq",
        "enable_oeq",
        "compile_mode",
        "compile_fullgraph",
        "pad_num_atoms",
        "pad_num_edges",
        "md_property_mode",
        "neighbor_skin_A",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise KeyError(
            f"{manifest_path}: unsupported calculator settings in schema-4 provenance: "
            f"{unknown}."
        )
    required_legacy = {"device", "default_dtype", "enable_cueq", "neighbor_skin_A"}
    missing_legacy = sorted(required_legacy - set(value))
    if missing_legacy:
        raise RuntimeError(
            f"{manifest_path}: calculator settings omit required schema-4 fields="
            f"{missing_legacy}."
        )
    normalized = dict(value)
    normalized.setdefault("enable_oeq", DEFAULT_ENABLE_OEQ)
    normalized.setdefault("compile_mode", DEFAULT_COMPILE_MODE)
    normalized.setdefault("compile_fullgraph", DEFAULT_COMPILE_FULLGRAPH)
    normalized.setdefault("pad_num_atoms", DEFAULT_PAD_NUM_ATOMS)
    normalized.setdefault("pad_num_edges", DEFAULT_PAD_NUM_EDGES)
    normalized.setdefault("md_property_mode", DEFAULT_MD_PROPERTY_MODE)
    normalized.setdefault(
        "kernel_backend",
        mace_kernel_backend(
            enable_cueq=bool(normalized["enable_cueq"]),
            enable_oeq=bool(normalized["enable_oeq"]),
        ),
    )
    return normalized


def build_execution_provenance(
    calculator: CalculatorProvenance,
) -> ExecutionProvenance:
    provenance = ExecutionProvenance(
        calculator=calculator,
        runtime=_runtime_provenance(calculator),
        producer_code=_producer_code_provenance(),
    )
    # Assert that the complete record is canonical JSON before it participates in
    # checkpoint identity or is written to a dataset manifest.
    json.dumps(provenance.to_dict(), sort_keys=True, separators=(",", ":"))
    return provenance


def validate_configured_source_manifest(
    manifest: dict[str, object],
    *,
    config: GeneratorConfig,
    manifest_path: Path,
) -> dict[str, object]:
    if manifest.get("schema_version") != 4:
        raise RuntimeError(
            f"{manifest_path}: expected current phase-context schema_version=4, got "
            f"{manifest.get('schema_version')!r}. Legacy artifacts must be regenerated."
        )
    expected_config = config.to_dict()
    observed_config = manifest.get("config")
    if observed_config != expected_config:
        relocated_config = (
            dict(observed_config) if isinstance(observed_config, dict) else None
        )
        if relocated_config is not None:
            relocated_config["config_path"] = expected_config["config_path"]
        if relocated_config != expected_config:
            raise RuntimeError(
                f"{manifest_path}: source generator configuration differs from the "
                "active source_generator_config beyond its repository file location. "
                "The source may use stale preparation dynamics, validation settings, "
                "cell handling, or model metadata; regenerate it with the active "
                "configuration before deriving another dataset."
            )
    execution = manifest.get("execution_provenance")
    if not isinstance(execution, dict):
        raise RuntimeError(
            f"{manifest_path}: source dataset has no execution_provenance mapping. Legacy "
            "artifacts cannot establish calculator, runtime, or producer-code identity and "
            "must be regenerated."
        )
    execution_keys = set(execution)
    if execution_keys != {"calculator", "runtime", "producer_code"}:
        raise RuntimeError(
            f"{manifest_path}: execution_provenance must contain exactly calculator, runtime, "
            f"and producer_code; got keys={sorted(execution_keys)}."
        )
    calculator = execution.get("calculator")
    if not isinstance(calculator, dict):
        raise RuntimeError(
            f"{manifest_path}: execution_provenance.calculator must be a mapping."
        )
    calculator_for_comparison = dict(calculator)
    calculator_for_comparison["settings"] = _normalized_schema4_calculator_settings(
        calculator.get("settings"), manifest_path=manifest_path
    )
    expected_calculator_fields = {
        "source": "configured_mace_model",
        "identity": (
            f"{config.potential.model_name}:{config.potential.sha256}:"
            f"{config.potential.head}"
        ),
        "implementation_class": (
            "src.data_utils.synthetic.atomistic.calculator."
            "VerletSkinMACECalculator"
        ),
        "model_name": config.potential.model_name,
        "family": config.potential.family,
        "model_path": str(config.potential.model_path),
        "model_sha256": config.potential.sha256,
        "head": config.potential.head,
        "source_url": config.potential.source_url,
        "license_identifier": config.potential.license_identifier,
        "usage_mode": config.potential.usage_mode,
        "validation_report_path": (
            None
            if config.potential.validation_report is None
            else str(config.potential.validation_report)
        ),
        "validation_report_sha256": config.potential.validation_report_sha256,
        "validation_report_type": (
            None
            if config.potential.qualification is None
            else config.potential.qualification.report_type
        ),
        "scientifically_qualified": config.potential.scientifically_qualified,
        "qualification_scope": _qualification_scope_dict(config),
        "settings": potential_calculator_settings(config.potential),
    }
    mismatches = {
        name: {"observed": calculator_for_comparison.get(name), "expected": expected}
        for name, expected in expected_calculator_fields.items()
        if calculator_for_comparison.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{manifest_path}: source calculator provenance does not match the configured "
            f"Hamiltonian and usage qualification: mismatches={mismatches}."
        )
    available_heads = calculator.get("available_heads")
    if (
        not isinstance(available_heads, list)
        or not available_heads
        or any(not isinstance(head, str) or not head for head in available_heads)
        or config.potential.head not in available_heads
    ):
        raise RuntimeError(
            f"{manifest_path}: calculator.available_heads={available_heads!r} does not "
            f"contain the selected head={config.potential.head!r}."
        )
    runtime = execution.get("runtime")
    if not isinstance(runtime, dict):
        raise TypeError(f"{manifest_path}: execution_provenance.runtime must be a mapping.")
    required_runtime_fields = {
        "python",
        "numpy",
        "ase",
        "torch",
        "platform",
        "machine",
        "mace_torch",
        "torch_cuda",
        "cudnn",
    }
    if config.potential.enable_cueq:
        required_runtime_fields.update(
            {
                "cuequivariance",
                "cuequivariance_torch",
                "cuequivariance_ops_torch_cu12",
            }
        )
    if config.potential.enable_oeq:
        required_runtime_fields.add("openequivariance")
    if config.potential.device.startswith("cuda"):
        required_runtime_fields.update({"cuda_device_index", "cuda_device_name"})
    missing_runtime_fields = sorted(required_runtime_fields - set(runtime))
    empty_runtime_fields = sorted(
        field
        for field in required_runtime_fields
        if field in runtime and runtime[field] is None
    )
    if missing_runtime_fields or empty_runtime_fields:
        raise RuntimeError(
            f"{manifest_path}: execution runtime provenance is incomplete; missing="
            f"{missing_runtime_fields}, null={empty_runtime_fields}."
        )
    producer_code = execution.get("producer_code")
    expected_producer_code = (
        homogeneous_liquid_source_producer_code_provenance()
        if manifest.get("source_kind") == "immutable_homogeneous_liquid_only"
        else _producer_code_provenance()
    )
    if not producer_code_is_compatible(producer_code, expected_producer_code):
        raise RuntimeError(
            f"{manifest_path}: source producer-code provenance does not match the active "
            f"atomistic package; observed={producer_code!r}, expected="
            f"{expected_producer_code!r}. Regenerate the source before deriving new data."
        )
    if manifest.get("potential_sha256") != config.potential.sha256:
        raise RuntimeError(
            f"{manifest_path}: compatibility potential_sha256="
            f"{manifest.get('potential_sha256')!r} differs from configured sha256="
            f"{config.potential.sha256!r}."
        )
    return calculator
