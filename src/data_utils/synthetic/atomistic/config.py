from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONFIGURED_MACE_IMPLEMENTATION_CLASS = (
    "src.data_utils.synthetic.atomistic.calculator.VerletSkinMACECalculator"
)
QUALIFICATION_REPORT_TYPE = "al_crystallization_mlip_qualification"
REQUIRED_PHASE_STATES = ("solid_bulk", "liquid_bulk", "interface")
ALLOWED_QUALIFIED_STATES = (
    *REQUIRED_PHASE_STATES,
    "strained_solid",
    "nucleus",
    "hcp",
    "bcc",
)
POTENTIAL_CLAIM_NAMES = (
    "phase_context_structure",
    "equilibrium_thermodynamics",
    "kinetics",
)
DEFAULT_ENABLE_OEQ = False
DEFAULT_COMPILE_MODE = None
DEFAULT_COMPILE_FULLGRAPH = False
DEFAULT_PAD_NUM_ATOMS = 0
DEFAULT_PAD_NUM_EDGES = 0
DEFAULT_MD_PROPERTY_MODE = "forces_stress"
SUPPORTED_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}
SUPPORTED_MD_PROPERTY_MODES = {"forces", "forces_stress"}


@dataclass(frozen=True)
class PotentialQualification:
    report_type: str
    model_sha256: str
    head: str | None
    chemical_symbol: str
    pressure_range_GPa: tuple[float, float]
    state_temperature_ranges_K: dict[str, tuple[float, float]]
    maximum_timestep_fs: float
    authorized_claims: dict[str, bool]
    implementation_class: str
    device: str
    default_dtype: str
    enable_cueq: bool
    enable_oeq: bool
    compile_mode: str | None
    compile_fullgraph: bool
    pad_num_atoms: int
    pad_num_edges: int
    md_property_mode: str
    neighbor_skin_A: float


@dataclass(frozen=True)
class PotentialConfig:
    model_name: str
    family: str
    model_path: Path
    sha256: str
    head: str | None
    source_url: str
    license_identifier: str
    usage_mode: str
    validation_report: Path | None
    validation_report_sha256: str | None
    scientifically_qualified: bool
    qualification: PotentialQualification | None
    device: str
    default_dtype: str
    enable_cueq: bool
    enable_oeq: bool
    compile_mode: str | None
    compile_fullgraph: bool
    pad_num_atoms: int
    pad_num_edges: int
    md_property_mode: str
    neighbor_skin_A: float


@dataclass(frozen=True)
class DynamicsConfig:
    pressure_GPa: float
    target_temperature_K: float
    melt_temperature_K: float
    timestep_fs: float
    thermostat_time_fs: float
    barostat_time_fs: float
    solid_equilibration_steps: int
    melt_steps: int
    quench_steps: int
    quench_stages: int
    target_equilibration_steps: int
    interface_evolution_steps: int
    sample_interval: int


@dataclass(frozen=True)
class SystemConfig:
    chemical_symbol: str
    crystal_structure: str
    initial_lattice_constant_A: float
    repetitions: tuple[int, int, int]
    liquid_slab_fraction: float
    interface_half_width_A: float


@dataclass(frozen=True)
class ValidationConfig:
    maximum_force_eV_per_A: float
    maximum_pressure_error_GPa: float
    maximum_temperature_error_K: float
    minimum_pair_distance_A: float
    ptm_rmsd_cutoff: float
    reference_density_cache: Path | None
    maximum_relative_density_error: float | None
    minimum_solid_fcc_fraction: float
    maximum_liquid_crystalline_fraction: float
    minimum_interface_crystalline_fraction: float
    maximum_interface_crystalline_fraction: float


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path
    overwrite: bool
    save_extxyz: bool
    create_visualizations: bool


@dataclass(frozen=True)
class GeneratorConfig:
    dataset_name: str
    random_seeds: tuple[int, ...]
    potential: PotentialConfig
    dynamics: DynamicsConfig
    system: SystemConfig
    validation: ValidationConfig
    output: OutputConfig
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return _serialize(result)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def mace_kernel_backend(*, enable_cueq: bool, enable_oeq: bool) -> str:
    if enable_cueq and enable_oeq:
        return "hybrid_cueq_oeq"
    if enable_cueq:
        return "cueq"
    if enable_oeq:
        return "oeq"
    return "e3nn"


def potential_calculator_settings(potential: PotentialConfig) -> dict[str, object]:
    """Canonical numerical/backend settings bound into provenance and reports."""

    return {
        "device": potential.device,
        "default_dtype": potential.default_dtype,
        "kernel_backend": mace_kernel_backend(
            enable_cueq=potential.enable_cueq,
            enable_oeq=potential.enable_oeq,
        ),
        "enable_cueq": potential.enable_cueq,
        "enable_oeq": potential.enable_oeq,
        "compile_mode": potential.compile_mode,
        "compile_fullgraph": potential.compile_fullgraph,
        "pad_num_atoms": potential.pad_num_atoms,
        "pad_num_edges": potential.pad_num_edges,
        "md_property_mode": potential.md_property_mode,
        "neighbor_skin_A": potential.neighbor_skin_A,
    }


def _mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key!r} must be a mapping, got {type(value).__name__}.")
    return value


def _required(mapping: dict[str, Any], key: str, context: str, path: Path) -> Any:
    if key not in mapping:
        raise KeyError(f"{path}: missing required {context}.{key}.")
    return mapping[key]


def _reject_unknown_keys(
    mapping: dict[str, Any], allowed: set[str], context: str, path: Path
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(
            f"{path}: unsupported keys in {context}: {unknown}. "
            "Remove obsolete/ad hoc controls instead of relying on ignored configuration."
        )


def _positive_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{path}: {context} must be > 0, got {result}.")
    return result


def _nonnegative_int(value: Any, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path}: {context} must be an integer, got {value!r}.")
    result = value
    if result < 0:
        raise ValueError(f"{path}: {context} must be >= 0, got {result}.")
    return result


def _positive_int(value: Any, context: str, path: Path) -> int:
    result = _nonnegative_int(value, context, path)
    if result == 0:
        raise ValueError(f"{path}: {context} must be > 0, got 0.")
    return result


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _finite_range(
    value: Any,
    *,
    context: str,
    path: Path,
    positive: bool,
) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(
            f"{path}: {context} must be a two-item [minimum, maximum] list, "
            f"got {value!r}."
        )
    if any(
        not isinstance(item, (int, float)) or isinstance(item, bool)
        for item in value
    ):
        raise TypeError(
            f"{path}: {context} endpoints must be explicit numbers, got {value!r}."
        )
    lower = float(value[0])
    upper = float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(
            f"{path}: {context} endpoints must be finite, got {value!r}."
        )
    if positive and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(
            f"{path}: {context} endpoints must be > 0, got {value!r}."
        )
    if lower > upper:
        raise ValueError(
            f"{path}: {context} minimum must not exceed its maximum, got {value!r}."
        )
    return lower, upper


def _strict_positive_float(value: Any, *, context: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            f"{path}: {context} must be an explicit number, got {value!r}."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{path}: {context} must be finite and > 0, got {result}."
        )
    return result


def _resolved_evidence_path(value: Any, *, context: str, report_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(
            f"{report_path}: {context} must be a non-empty path string, got {value!r}."
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _verify_evidence_digest(
    path: Path,
    expected_sha256: Any,
    *,
    context: str,
    report_path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{report_path}: {context} does not exist: {path}.")
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 != observed_sha256:
        raise RuntimeError(
            f"{report_path}: {context} SHA-256 mismatch for {path}: report="
            f"{expected_sha256!r}, observed={observed_sha256!r}."
        )


def _validate_qualification_chain(
    report: dict[str, Any],
    *,
    report_path: Path,
    expected_model_name: str,
    expected_model_sha256: str,
    expected_head: str | None,
) -> None:
    report_model_name = _required(report, "model_name", "report", report_path)
    if report_model_name != expected_model_name:
        raise RuntimeError(
            f"{report_path}: qualification report model_name={report_model_name!r}, expected "
            f"{expected_model_name!r}."
        )
    scientifically_qualified = _required(
        report, "scientifically_qualified", "report", report_path
    )
    failures = _required(report, "qualification_failures", "report", report_path)
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise TypeError(
            f"{report_path}: qualification_failures must be a list of non-empty strings, "
            f"got {failures!r}."
        )
    if scientifically_qualified and failures:
        raise RuntimeError(
            f"{report_path}: scientifically_qualified=true conflicts with non-empty "
            f"qualification_failures={failures!r}."
        )
    if not scientifically_qualified and not failures:
        raise RuntimeError(
            f"{report_path}: scientifically_qualified=false requires explicit failure reasons."
        )

    benchmark_path = _resolved_evidence_path(
        _required(report, "benchmark_report", "report", report_path),
        context="benchmark_report",
        report_path=report_path,
    )
    _verify_evidence_digest(
        benchmark_path,
        _required(report, "benchmark_report_sha256", "report", report_path),
        context="benchmark_report",
        report_path=report_path,
    )
    with benchmark_path.open("r", encoding="utf-8") as handle:
        benchmark = json.load(handle)
    if not isinstance(benchmark, dict) or benchmark.get("schema_version") != 1:
        raise RuntimeError(
            f"{benchmark_path}: expected potential benchmark schema_version=1."
        )
    qualification_reports = benchmark.get("qualification_reports")
    if not isinstance(qualification_reports, dict):
        raise TypeError(f"{benchmark_path}: qualification_reports must be a mapping.")
    linked_child_path = _resolved_evidence_path(
        qualification_reports.get(expected_model_name),
        context=f"qualification_reports.{expected_model_name}",
        report_path=benchmark_path,
    )
    if linked_child_path != report_path.resolve():
        raise RuntimeError(
            f"{benchmark_path}: qualification report link for {expected_model_name!r} points "
            f"to {linked_child_path}, not the configured child {report_path.resolve()}."
        )

    models = benchmark.get("models")
    if not isinstance(models, dict) or not isinstance(
        models.get(expected_model_name), dict
    ):
        raise RuntimeError(
            f"{benchmark_path}: no model result exists for {expected_model_name!r}."
        )
    model_result = models[expected_model_name]
    identity = model_result.get("identity")
    if not isinstance(identity, dict):
        raise TypeError(
            f"{benchmark_path}: models.{expected_model_name}.identity must be a mapping."
        )
    if (
        identity.get("model_name") != expected_model_name
        or identity.get("sha256") != expected_model_sha256
        or identity.get("head") != expected_head
    ):
        raise RuntimeError(
            f"{benchmark_path}: parent model identity={identity!r} does not match configured "
            f"model={expected_model_name!r}, sha256={expected_model_sha256!r}, "
            f"head={expected_head!r}."
        )
    if (
        model_result.get("scientifically_qualified") != scientifically_qualified
        or model_result.get("qualification_failures") != failures
    ):
        raise RuntimeError(
            f"{report_path}: child qualification status/failures do not match parent model "
            f"result in {benchmark_path}."
        )
    selection = benchmark.get("selection")
    if not isinstance(selection, dict):
        raise TypeError(f"{benchmark_path}: selection must be a mapping.")
    qualified_models = selection.get("qualified_models")
    if not isinstance(qualified_models, list) or (
        (expected_model_name in qualified_models) != scientifically_qualified
    ):
        raise RuntimeError(
            f"{benchmark_path}: selection.qualified_models={qualified_models!r} is inconsistent "
            f"with model qualification status={scientifically_qualified}."
        )
    if selection.get("automatic_replacement_allowed") is not False or selection.get(
        "manual_comparison_required"
    ) is not True:
        raise RuntimeError(
            f"{benchmark_path}: benchmark selection policy must forbid automatic replacement "
            "and require manual comparison."
        )

    evidence = _mapping(report, "evidence", report_path)
    benchmark_config = benchmark.get("benchmark_config")
    if not isinstance(benchmark_config, dict):
        raise TypeError(f"{benchmark_path}: benchmark_config must be a mapping.")
    child_scope = _mapping(report, "scope", report_path)
    benchmark_scope = benchmark_config.get("qualification_scope")
    if not isinstance(benchmark_scope, dict):
        raise TypeError(
            f"{benchmark_path}: benchmark_config.qualification_scope must be a mapping."
        )
    benchmark_claims = benchmark_scope.get("authorized_claims")
    if not isinstance(benchmark_claims, dict):
        raise TypeError(
            f"{benchmark_path}: benchmark_config.qualification_scope.authorized_claims "
            "must be a mapping."
        )
    if set(benchmark_claims) != set(POTENTIAL_CLAIM_NAMES) or any(
        type(benchmark_claims.get(claim)) is not bool for claim in POTENTIAL_CLAIM_NAMES
    ):
        raise TypeError(
            f"{benchmark_path}: benchmark_config.qualification_scope.authorized_claims "
            f"must contain exactly boolean values for {list(POTENTIAL_CLAIM_NAMES)}, got "
            f"{benchmark_claims!r}."
        )
    effective_authorized_claims = {
        claim: bool(scientifically_qualified and benchmark_claims.get(claim) is True)
        for claim in POTENTIAL_CLAIM_NAMES
    }
    expected_child_scope_fields = {
        "chemical_symbol": benchmark_config.get("chemical_symbol"),
        "pressure_range_GPa": benchmark_scope.get("pressure_range_GPa"),
        "state_temperature_ranges_K": benchmark_scope.get(
            "state_temperature_ranges_K"
        ),
        "maximum_timestep_fs": benchmark_scope.get("maximum_timestep_fs"),
        "authorized_claims": effective_authorized_claims,
    }
    scope_mismatches = {
        key: {
            "qualification_report": child_scope.get(key),
            "benchmark_report": expected_value,
        }
        for key, expected_value in expected_child_scope_fields.items()
        if child_scope.get(key) != expected_value
    }
    if scope_mismatches:
        raise RuntimeError(
            f"{report_path}: qualification scope expands or differs from its bound benchmark "
            f"configuration: mismatches={scope_mismatches}."
        )
    canonical_sha256 = hashlib.sha256(
        json.dumps(benchmark_config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if evidence.get("benchmark_config_canonical_sha256") != canonical_sha256:
        raise RuntimeError(
            f"{report_path}: benchmark config canonical digest does not match parent report."
        )
    benchmark_config_path = _resolved_evidence_path(
        benchmark_config.get("config_path"),
        context="benchmark_config.config_path",
        report_path=benchmark_path,
    )
    _verify_evidence_digest(
        benchmark_config_path,
        evidence.get("benchmark_config_file_sha256"),
        context="benchmark config file",
        report_path=report_path,
    )

    parent_dft_evidence = benchmark.get("dft_reference_evidence")
    parent_melting_evidence = benchmark.get("melting_scans", {}).get(
        expected_model_name
    ) if isinstance(benchmark.get("melting_scans"), dict) else None
    if evidence.get("dft_reference") != parent_dft_evidence or evidence.get(
        "melting_scans"
    ) != parent_melting_evidence:
        raise RuntimeError(
            f"{report_path}: child DFT/melting evidence does not exactly match parent "
            f"benchmark {benchmark_path}."
        )
    if report.get("melting_scan") != parent_melting_evidence:
        raise RuntimeError(
            f"{report_path}: top-level melting_scan does not match parent evidence."
        )
    parent_child_pairs = (
        ("equation_of_state", "equation_of_state"),
        (
            "phase_energy_difference_from_fcc_meV_per_atom",
            "phase_energy_difference_from_fcc_meV_per_atom",
        ),
        ("nve", "nve"),
        ("dft_reference_errors", "dft_reference_errors"),
    )
    for child_key, parent_key in parent_child_pairs:
        if report.get(child_key) != model_result.get(parent_key):
            raise RuntimeError(
                f"{report_path}: child {child_key} does not match parent model result."
            )

    if parent_dft_evidence is not None:
        if not isinstance(parent_dft_evidence, dict):
            raise TypeError(f"{benchmark_path}: dft_reference_evidence must be a mapping.")
        dft_path = _resolved_evidence_path(
            parent_dft_evidence.get("path"),
            context="DFT reference evidence path",
            report_path=benchmark_path,
        )
        _verify_evidence_digest(
            dft_path,
            parent_dft_evidence.get("sha256"),
            context="DFT reference evidence",
            report_path=report_path,
        )
    if parent_melting_evidence is not None:
        if not isinstance(parent_melting_evidence, dict) or not isinstance(
            parent_melting_evidence.get("scans"), list
        ):
            raise TypeError(f"{benchmark_path}: melting evidence/scans are malformed.")
        for scan in parent_melting_evidence["scans"]:
            if not isinstance(scan, dict):
                raise TypeError(f"{benchmark_path}: every melting scan must be a mapping.")
            scan_path = _resolved_evidence_path(
                scan.get("path"),
                context="melting scan path",
                report_path=benchmark_path,
            )
            _verify_evidence_digest(
                scan_path,
                scan.get("sha256"),
                context="melting scan",
                report_path=report_path,
            )
    if scientifically_qualified and (
        parent_dft_evidence is None
        or parent_melting_evidence is None
        or model_result.get("dft_reference_errors") is None
    ):
        raise RuntimeError(
            f"{report_path}: scientifically_qualified=true requires non-null DFT errors and "
            "direct-coexistence melting evidence in the bound parent benchmark."
        )


def _parse_potential_qualification(
    report: dict[str, Any],
    *,
    report_path: Path,
    expected_model_name: str,
    expected_model_sha256: str,
    expected_head: str | None,
    expected_device: str,
    expected_default_dtype: str,
    expected_enable_cueq: bool,
    expected_enable_oeq: bool,
    expected_compile_mode: str | None,
    expected_compile_fullgraph: bool,
    expected_pad_num_atoms: int,
    expected_pad_num_edges: int,
    expected_md_property_mode: str,
    expected_neighbor_skin_A: float,
) -> PotentialQualification:
    _validate_qualification_chain(
        report,
        report_path=report_path,
        expected_model_name=expected_model_name,
        expected_model_sha256=expected_model_sha256,
        expected_head=expected_head,
    )
    report_sha256 = _required(report, "model_sha256", "report", report_path)
    report_head = _required(report, "head", "report", report_path)
    if report_sha256 != expected_model_sha256 or report_head != expected_head:
        raise RuntimeError(
            f"{report_path}: validation report identifies model_sha256="
            f"{report_sha256!r}, head={report_head!r}, but the active potential uses "
            f"model_sha256={expected_model_sha256!r}, head={expected_head!r}."
        )
    scope = _mapping(report, "scope", report_path)
    _reject_unknown_keys(
        scope,
        {
            "chemical_symbol",
            "pressure_range_GPa",
            "state_temperature_ranges_K",
            "maximum_timestep_fs",
            "authorized_claims",
            "calculator_settings",
        },
        "report.scope",
        report_path,
    )
    chemical_symbol = _required(
        scope, "chemical_symbol", "report.scope", report_path
    )
    if not isinstance(chemical_symbol, str) or not chemical_symbol:
        raise TypeError(
            f"{report_path}: report.scope.chemical_symbol must be a non-empty string, "
            f"got {chemical_symbol!r}."
        )
    pressure_range_GPa = _finite_range(
        _required(scope, "pressure_range_GPa", "report.scope", report_path),
        context="report.scope.pressure_range_GPa",
        path=report_path,
        positive=False,
    )
    state_ranges_raw = _mapping(scope, "state_temperature_ranges_K", report_path)
    missing_states = sorted(set(ALLOWED_QUALIFIED_STATES) - set(state_ranges_raw))
    unknown_states = sorted(set(state_ranges_raw) - set(ALLOWED_QUALIFIED_STATES))
    if missing_states or unknown_states:
        raise KeyError(
            f"{report_path}: report.scope.state_temperature_ranges_K has missing required "
            f"states={missing_states} and unknown states={unknown_states}; supported states="
            f"{list(ALLOWED_QUALIFIED_STATES)}."
        )
    state_temperature_ranges_K = {
        state: _finite_range(
            state_ranges_raw[state],
            context=f"report.scope.state_temperature_ranges_K.{state}",
            path=report_path,
            positive=True,
        )
        for state in ALLOWED_QUALIFIED_STATES
        if state in state_ranges_raw
    }
    maximum_timestep_fs = _strict_positive_float(
        _required(scope, "maximum_timestep_fs", "report.scope", report_path),
        context="report.scope.maximum_timestep_fs",
        path=report_path,
    )
    authorized_claims_raw = _mapping(scope, "authorized_claims", report_path)
    missing_claims = sorted(set(POTENTIAL_CLAIM_NAMES) - set(authorized_claims_raw))
    unknown_claims = sorted(set(authorized_claims_raw) - set(POTENTIAL_CLAIM_NAMES))
    if missing_claims or unknown_claims:
        raise KeyError(
            f"{report_path}: report.scope.authorized_claims must contain exactly "
            f"{list(POTENTIAL_CLAIM_NAMES)}; missing={missing_claims}, "
            f"unknown={unknown_claims}."
        )
    if any(type(authorized_claims_raw[claim]) is not bool for claim in POTENTIAL_CLAIM_NAMES):
        raise TypeError(
            f"{report_path}: every report.scope.authorized_claims value must be an exact "
            f"boolean, got {authorized_claims_raw!r}."
        )
    authorized_claims = {
        claim: authorized_claims_raw[claim] for claim in POTENTIAL_CLAIM_NAMES
    }
    calculator_settings = _mapping(scope, "calculator_settings", report_path)
    _reject_unknown_keys(
        calculator_settings,
        {
            "implementation_class",
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
        },
        "report.scope.calculator_settings",
        report_path,
    )
    implementation_class = _required(
        calculator_settings,
        "implementation_class",
        "report.scope.calculator_settings",
        report_path,
    )
    device = _required(
        calculator_settings,
        "device",
        "report.scope.calculator_settings",
        report_path,
    )
    default_dtype = _required(
        calculator_settings,
        "default_dtype",
        "report.scope.calculator_settings",
        report_path,
    )
    enable_cueq = _required(
        calculator_settings,
        "enable_cueq",
        "report.scope.calculator_settings",
        report_path,
    )
    enable_oeq = calculator_settings.get("enable_oeq", DEFAULT_ENABLE_OEQ)
    compile_mode = calculator_settings.get("compile_mode", DEFAULT_COMPILE_MODE)
    compile_fullgraph = calculator_settings.get(
        "compile_fullgraph", DEFAULT_COMPILE_FULLGRAPH
    )
    pad_num_atoms = calculator_settings.get("pad_num_atoms", DEFAULT_PAD_NUM_ATOMS)
    pad_num_edges = calculator_settings.get("pad_num_edges", DEFAULT_PAD_NUM_EDGES)
    md_property_mode = calculator_settings.get(
        "md_property_mode", DEFAULT_MD_PROPERTY_MODE
    )
    if not isinstance(implementation_class, str) or not implementation_class:
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.implementation_class must "
            f"be a non-empty string, got {implementation_class!r}."
        )
    if not isinstance(device, str) or not device:
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.device must be a non-empty "
            f"string, got {device!r}."
        )
    if not isinstance(default_dtype, str) or not default_dtype:
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.default_dtype must be a "
            f"non-empty string, got {default_dtype!r}."
        )
    if not isinstance(enable_cueq, bool):
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.enable_cueq must be a "
            f"boolean, got {enable_cueq!r}."
        )
    if not isinstance(enable_oeq, bool):
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.enable_oeq must be a "
            f"boolean, got {enable_oeq!r}."
        )
    if compile_mode is not None and compile_mode not in SUPPORTED_COMPILE_MODES:
        raise ValueError(
            f"{report_path}: report.scope.calculator_settings.compile_mode must be null "
            f"or one of {sorted(SUPPORTED_COMPILE_MODES)}, got {compile_mode!r}."
        )
    if not isinstance(compile_fullgraph, bool):
        raise TypeError(
            f"{report_path}: report.scope.calculator_settings.compile_fullgraph must be a "
            f"boolean, got {compile_fullgraph!r}."
        )
    pad_num_atoms = _nonnegative_int(
        pad_num_atoms,
        "report.scope.calculator_settings.pad_num_atoms",
        report_path,
    )
    pad_num_edges = _nonnegative_int(
        pad_num_edges,
        "report.scope.calculator_settings.pad_num_edges",
        report_path,
    )
    if not isinstance(md_property_mode, str) or (
        md_property_mode not in SUPPORTED_MD_PROPERTY_MODES
    ):
        raise ValueError(
            f"{report_path}: report.scope.calculator_settings.md_property_mode must be "
            f"one of {sorted(SUPPORTED_MD_PROPERTY_MODES)}, got {md_property_mode!r}."
        )
    kernel_backend = calculator_settings.get(
        "kernel_backend",
        mace_kernel_backend(enable_cueq=enable_cueq, enable_oeq=enable_oeq),
    )
    expected_kernel_backend = mace_kernel_backend(
        enable_cueq=expected_enable_cueq,
        enable_oeq=expected_enable_oeq,
    )
    neighbor_skin_A = _strict_positive_float(
        _required(
            calculator_settings,
            "neighbor_skin_A",
            "report.scope.calculator_settings",
            report_path,
        ),
        context="report.scope.calculator_settings.neighbor_skin_A",
        path=report_path,
    )
    expected_settings = {
        "implementation_class": CONFIGURED_MACE_IMPLEMENTATION_CLASS,
        "device": expected_device,
        "default_dtype": expected_default_dtype,
        "kernel_backend": expected_kernel_backend,
        "enable_cueq": expected_enable_cueq,
        "enable_oeq": expected_enable_oeq,
        "compile_mode": expected_compile_mode,
        "compile_fullgraph": expected_compile_fullgraph,
        "pad_num_atoms": expected_pad_num_atoms,
        "pad_num_edges": expected_pad_num_edges,
        "md_property_mode": expected_md_property_mode,
        "neighbor_skin_A": expected_neighbor_skin_A,
    }
    observed_settings = {
        "implementation_class": implementation_class,
        "device": device,
        "default_dtype": default_dtype,
        "kernel_backend": kernel_backend,
        "enable_cueq": enable_cueq,
        "enable_oeq": enable_oeq,
        "compile_mode": compile_mode,
        "compile_fullgraph": compile_fullgraph,
        "pad_num_atoms": pad_num_atoms,
        "pad_num_edges": pad_num_edges,
        "md_property_mode": md_property_mode,
        "neighbor_skin_A": neighbor_skin_A,
    }
    if observed_settings != expected_settings:
        raise RuntimeError(
            f"{report_path}: validation report calculator settings do not match the "
            f"configured production implementation; observed={observed_settings}, "
            f"expected={expected_settings}."
        )
    return PotentialQualification(
        report_type=QUALIFICATION_REPORT_TYPE,
        model_sha256=str(report_sha256),
        head=report_head,
        chemical_symbol=chemical_symbol,
        pressure_range_GPa=pressure_range_GPa,
        state_temperature_ranges_K=state_temperature_ranges_K,
        maximum_timestep_fs=maximum_timestep_fs,
        authorized_claims=authorized_claims,
        implementation_class=str(implementation_class),
        device=str(device),
        default_dtype=str(default_dtype),
        enable_cueq=bool(enable_cueq),
        enable_oeq=bool(enable_oeq),
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        pad_num_atoms=pad_num_atoms,
        pad_num_edges=pad_num_edges,
        md_property_mode=str(md_property_mode),
        neighbor_skin_A=neighbor_skin_A,
    )


def validate_potential_qualification(
    config: GeneratorConfig,
    *,
    chemical_symbol: str,
    pressure_GPa: float,
    timestep_fs: float,
    state_temperatures_K: dict[str, tuple[float, ...]],
    context: str,
    required_claim: str = "phase_context_structure",
) -> None:
    if required_claim not in POTENTIAL_CLAIM_NAMES:
        raise ValueError(
            f"{context}: required_claim={required_claim!r} is unsupported; expected one of "
            f"{list(POTENTIAL_CLAIM_NAMES)}."
        )
    if config.potential.usage_mode == "exploratory":
        return
    qualification = config.potential.qualification
    report_path = config.potential.validation_report
    if qualification is None or report_path is None:
        raise RuntimeError(
            f"{context}: quantitative potential use requires a parsed qualification report."
        )
    observed_report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if observed_report_sha256 != config.potential.validation_report_sha256:
        raise RuntimeError(
            f"{context}: qualification report changed after configuration load: path="
            f"{report_path}, loaded_sha256={config.potential.validation_report_sha256}, "
            f"observed_sha256={observed_report_sha256}. Reload and revalidate the "
            "configuration."
        )
    if not config.potential.scientifically_qualified:
        raise RuntimeError(
            f"{context}: qualification report {report_path} did not set "
            "scientifically_qualified=true."
        )
    if qualification.report_type != QUALIFICATION_REPORT_TYPE:
        raise RuntimeError(
            f"{context}: qualification report {report_path} has report_type="
            f"{qualification.report_type!r}, expected {QUALIFICATION_REPORT_TYPE!r}."
        )
    if qualification.authorized_claims[required_claim] is not True:
        raise RuntimeError(
            f"{context}: qualification report {report_path} does not authorize required "
            f"claim={required_claim!r}; authorized_claims="
            f"{qualification.authorized_claims}."
        )
    if (
        qualification.model_sha256 != config.potential.sha256
        or qualification.head != config.potential.head
    ):
        raise RuntimeError(
            f"{context}: qualification report {report_path} is bound to model_sha256="
            f"{qualification.model_sha256!r}, head={qualification.head!r}, but the active "
            f"potential uses model_sha256={config.potential.sha256!r}, "
            f"head={config.potential.head!r}."
        )
    qualified_settings = {
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
    }
    requested_settings = {
        "implementation_class": CONFIGURED_MACE_IMPLEMENTATION_CLASS,
        **potential_calculator_settings(config.potential),
    }
    if qualified_settings != requested_settings:
        raise RuntimeError(
            f"{context}: requested calculator settings={requested_settings} differ from "
            f"qualified settings={qualified_settings} in {report_path}."
        )
    if qualification.chemical_symbol != chemical_symbol:
        raise RuntimeError(
            f"{context}: qualification report {report_path} covers chemical_symbol="
            f"{qualification.chemical_symbol!r}, not requested {chemical_symbol!r}."
        )
    pressure_min, pressure_max = qualification.pressure_range_GPa
    if not pressure_min <= pressure_GPa <= pressure_max:
        raise RuntimeError(
            f"{context}: requested pressure={pressure_GPa} GPa is outside qualification "
            f"range [{pressure_min}, {pressure_max}] GPa in {report_path}."
        )
    if timestep_fs > qualification.maximum_timestep_fs:
        raise RuntimeError(
            f"{context}: requested timestep={timestep_fs} fs exceeds qualified maximum="
            f"{qualification.maximum_timestep_fs} fs in {report_path}."
        )
    for state, temperatures_K in state_temperatures_K.items():
        if state not in qualification.state_temperature_ranges_K:
            raise RuntimeError(
                f"{context}: state={state!r} has no temperature qualification in "
                f"{report_path}."
            )
        temperature_min, temperature_max = (
            qualification.state_temperature_ranges_K[state]
        )
        out_of_scope = [
            temperature_K
            for temperature_K in temperatures_K
            if not temperature_min <= temperature_K <= temperature_max
        ]
        if out_of_scope:
            raise RuntimeError(
                f"{context}: state={state!r} requests temperatures_K={out_of_scope} outside "
                f"qualified range [{temperature_min}, {temperature_max}] K in {report_path}."
            )


def _reject_density_controls(value: Any, *, path: Path, location: str = "root") -> None:
    forbidden = {"density_target", "rho_target", "target_density", "avg_nn_dist"}
    if isinstance(value, dict):
        for key, item in value.items():
            child_location = f"{location}.{key}"
            if str(key) in forbidden:
                raise ValueError(
                    f"{path}: {child_location} is not accepted. Number density is an NPT "
                    "simulation result, not a generator input."
                )
            _reject_density_controls(item, path=path, location=child_location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_density_controls(item, path=path, location=f"{location}[{index}]")


def load_config(path: str | Path) -> GeneratorConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping, got {type(raw).__name__}.")
    _reject_density_controls(raw, path=config_path)
    _reject_unknown_keys(
        raw,
        {
            "kind",
            "data_path",
            "num_points",
            "model_points",
            "radius",
            "sample_type",
            "overlap_fraction",
            "n_samples",
            "dataset_max_samples",
            "train_ratio",
            "drop_edge_samples",
            "pre_normalize",
            "normalize",
            "auto_cutoff",
            "synthetic",
            "dataset_name",
            "random_seeds",
            "potential",
            "dynamics",
            "system",
            "validation",
            "output",
        },
        "root",
        config_path,
    )

    potential_raw = _mapping(raw, "potential", config_path)
    dynamics_raw = _mapping(raw, "dynamics", config_path)
    system_raw = _mapping(raw, "system", config_path)
    validation_raw = _mapping(raw, "validation", config_path)
    output_raw = _mapping(raw, "output", config_path)
    _reject_unknown_keys(
        potential_raw,
        {
            "model_name",
            "family",
            "model_path",
            "sha256",
            "head",
            "source_url",
            "license_identifier",
            "usage_mode",
            "validation_report",
            "device",
            "default_dtype",
            "enable_cueq",
            "enable_oeq",
            "compile_mode",
            "compile_fullgraph",
            "pad_num_atoms",
            "pad_num_edges",
            "md_property_mode",
            "neighbor_skin_A",
        },
        "potential",
        config_path,
    )
    _reject_unknown_keys(
        dynamics_raw,
        {
            "pressure_GPa",
            "target_temperature_K",
            "melt_temperature_K",
            "timestep_fs",
            "thermostat_time_fs",
            "barostat_time_fs",
            "solid_equilibration_steps",
            "melt_steps",
            "quench_steps",
            "quench_stages",
            "target_equilibration_steps",
            "interface_evolution_steps",
            "sample_interval",
        },
        "dynamics",
        config_path,
    )
    _reject_unknown_keys(
        system_raw,
        {
            "chemical_symbol",
            "crystal_structure",
            "initial_lattice_constant_A",
            "repetitions",
            "liquid_slab_fraction",
            "interface_half_width_A",
        },
        "system",
        config_path,
    )
    _reject_unknown_keys(
        validation_raw,
        {
            "maximum_force_eV_per_A",
            "maximum_pressure_error_GPa",
            "maximum_temperature_error_K",
            "minimum_pair_distance_A",
            "ptm_rmsd_cutoff",
            "reference_density_cache",
            "maximum_relative_density_error",
            "minimum_solid_fcc_fraction",
            "maximum_liquid_crystalline_fraction",
            "minimum_interface_crystalline_fraction",
            "maximum_interface_crystalline_fraction",
        },
        "validation",
        config_path,
    )
    _reject_unknown_keys(
        output_raw,
        {"root_dir", "overwrite", "save_extxyz", "create_visualizations"},
        "output",
        config_path,
    )

    model_path = _resolve_repo_path(
        _required(potential_raw, "model_path", "potential", config_path)
    )
    datasets_root = (REPOSITORY_ROOT / "datasets").resolve()
    if not model_path.is_relative_to(datasets_root):
        raise ValueError(
            f"{config_path}: potential.model_path must be repository-owned data below "
            f"{datasets_root}, got {model_path}."
        )
    repetitions_raw = _required(system_raw, "repetitions", "system", config_path)
    if not isinstance(repetitions_raw, list) or len(repetitions_raw) != 3:
        raise ValueError(
            f"{config_path}: system.repetitions must be a three-item list, got {repetitions_raw!r}."
        )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in repetitions_raw):
        raise TypeError(
            f"{config_path}: system.repetitions entries must be integers, got "
            f"{repetitions_raw!r}."
        )
    repetitions = tuple(repetitions_raw)
    if min(repetitions) < 2:
        raise ValueError(
            f"{config_path}: every system.repetitions entry must be >= 2, got {repetitions}."
        )

    random_seeds_raw = _required(raw, "random_seeds", "root", config_path)
    if not isinstance(random_seeds_raw, list) or not random_seeds_raw:
        raise TypeError(
            f"{config_path}: random_seeds must be a non-empty list of integers, "
            f"got {random_seeds_raw!r}."
        )
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in random_seeds_raw):
        raise TypeError(
            f"{config_path}: every random_seeds entry must be an integer, "
            f"got {random_seeds_raw!r}."
        )
    if len(set(random_seeds_raw)) != len(random_seeds_raw):
        raise ValueError(
            f"{config_path}: random_seeds must be unique, got {random_seeds_raw!r}."
        )

    target_temperature = _positive_float(
        _required(dynamics_raw, "target_temperature_K", "dynamics", config_path),
        "dynamics.target_temperature_K",
        config_path,
    )
    melt_temperature = _positive_float(
        _required(dynamics_raw, "melt_temperature_K", "dynamics", config_path),
        "dynamics.melt_temperature_K",
        config_path,
    )
    if melt_temperature <= target_temperature:
        raise ValueError(
            f"{config_path}: dynamics.melt_temperature_K={melt_temperature} must exceed "
            f"target_temperature_K={target_temperature}."
        )

    slab_fraction = float(_required(system_raw, "liquid_slab_fraction", "system", config_path))
    if not 0.2 <= slab_fraction <= 0.8:
        raise ValueError(
            f"{config_path}: system.liquid_slab_fraction must be within [0.2, 0.8], "
            f"got {slab_fraction}."
        )

    density_cache_value = validation_raw.get("reference_density_cache")
    density_error_value = validation_raw.get("maximum_relative_density_error")
    if density_cache_value is not None or density_error_value is not None:
        raise ValueError(
            f"{config_path}: legacy reference_density_cache validation is disabled. Its "
            "first/last-frame density inference has no explicit temperature, pressure, phase, "
            "units, protocol, or potential provenance and cannot qualify the configured MLIP. "
            "Set both reference_density_cache and maximum_relative_density_error to null."
        )
    density_cache = None
    density_error = None
    ptm_rmsd_cutoff = _positive_float(
        _required(validation_raw, "ptm_rmsd_cutoff", "validation", config_path),
        "validation.ptm_rmsd_cutoff",
        config_path,
    )
    if ptm_rmsd_cutoff > 1.0:
        raise ValueError(
            f"{config_path}: validation.ptm_rmsd_cutoff is a dimensionless normalized "
            f"RMSD and must be <= 1, got {ptm_rmsd_cutoff}."
        )
    minimum_solid_fcc_fraction = float(
        _required(
            validation_raw, "minimum_solid_fcc_fraction", "validation", config_path
        )
    )
    maximum_liquid_crystalline_fraction = float(
        _required(
            validation_raw,
            "maximum_liquid_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    minimum_interface_crystalline_fraction = float(
        _required(
            validation_raw,
            "minimum_interface_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    maximum_interface_crystalline_fraction = float(
        _required(
            validation_raw,
            "maximum_interface_crystalline_fraction",
            "validation",
            config_path,
        )
    )
    for context, value in (
        ("validation.minimum_solid_fcc_fraction", minimum_solid_fcc_fraction),
        (
            "validation.maximum_liquid_crystalline_fraction",
            maximum_liquid_crystalline_fraction,
        ),
        (
            "validation.minimum_interface_crystalline_fraction",
            minimum_interface_crystalline_fraction,
        ),
        (
            "validation.maximum_interface_crystalline_fraction",
            maximum_interface_crystalline_fraction,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{config_path}: {context} must be in [0, 1], got {value}.")
    if minimum_interface_crystalline_fraction >= maximum_interface_crystalline_fraction:
        raise ValueError(
            f"{config_path}: validation.minimum_interface_crystalline_fraction must be less "
            "than validation.maximum_interface_crystalline_fraction."
        )

    device = str(_required(potential_raw, "device", "potential", config_path)).strip()
    if not device:
        raise ValueError(f"{config_path}: potential.device must be a non-empty string.")
    default_dtype = str(potential_raw.get("default_dtype", "float32"))
    if default_dtype not in {"float32", "float64"}:
        raise ValueError(
            f"{config_path}: potential.default_dtype must be 'float32' or 'float64', "
            f"got {default_dtype!r}."
        )
    enable_cueq = _required(potential_raw, "enable_cueq", "potential", config_path)
    if not isinstance(enable_cueq, bool):
        raise TypeError(
            f"{config_path}: potential.enable_cueq must be a boolean, "
            f"got {enable_cueq!r}."
        )
    enable_oeq = potential_raw.get("enable_oeq", DEFAULT_ENABLE_OEQ)
    if not isinstance(enable_oeq, bool):
        raise TypeError(
            f"{config_path}: potential.enable_oeq must be a boolean, "
            f"got {enable_oeq!r}."
        )
    if (enable_cueq or enable_oeq) and not device.startswith("cuda"):
        raise ValueError(
            f"{config_path}: accelerated MACE kernels require potential.device='cuda' or "
            f"an explicit CUDA index, got device={device!r}, enable_cueq={enable_cueq}, "
            f"enable_oeq={enable_oeq}."
        )
    compile_mode_raw = potential_raw.get("compile_mode", DEFAULT_COMPILE_MODE)
    if compile_mode_raw is not None and not isinstance(compile_mode_raw, str):
        raise TypeError(
            f"{config_path}: potential.compile_mode must be a string or null, "
            f"got {compile_mode_raw!r}."
        )
    compile_mode = compile_mode_raw
    if compile_mode is not None and compile_mode not in SUPPORTED_COMPILE_MODES:
        raise ValueError(
            f"{config_path}: potential.compile_mode must be null or one of "
            f"{sorted(SUPPORTED_COMPILE_MODES)}, got {compile_mode!r}."
        )
    compile_fullgraph = potential_raw.get(
        "compile_fullgraph", DEFAULT_COMPILE_FULLGRAPH
    )
    if not isinstance(compile_fullgraph, bool):
        raise TypeError(
            f"{config_path}: potential.compile_fullgraph must be a boolean, "
            f"got {compile_fullgraph!r}."
        )
    pad_num_atoms = _nonnegative_int(
        potential_raw.get("pad_num_atoms", DEFAULT_PAD_NUM_ATOMS),
        "potential.pad_num_atoms",
        config_path,
    )
    pad_num_edges = _nonnegative_int(
        potential_raw.get("pad_num_edges", DEFAULT_PAD_NUM_EDGES),
        "potential.pad_num_edges",
        config_path,
    )
    if compile_mode is None:
        if compile_fullgraph or pad_num_atoms != 0 or pad_num_edges != 0:
            raise ValueError(
                f"{config_path}: compile_fullgraph and fixed padding require a non-null "
                f"potential.compile_mode; got compile_fullgraph={compile_fullgraph}, "
                f"pad_num_atoms={pad_num_atoms}, pad_num_edges={pad_num_edges}."
            )
    else:
        if pad_num_atoms == 0 or pad_num_edges == 0:
            raise ValueError(
                f"{config_path}: compiled MACE requires explicit positive fixed-shape "
                f"potential.pad_num_atoms and potential.pad_num_edges; got "
                f"{pad_num_atoms} and {pad_num_edges}."
            )
        if compile_fullgraph and (enable_cueq or enable_oeq):
            raise ValueError(
                f"{config_path}: mace-torch 0.3.16 accelerated kernels require "
                "potential.compile_fullgraph=false."
            )
    md_property_mode = potential_raw.get(
        "md_property_mode", DEFAULT_MD_PROPERTY_MODE
    )
    if not isinstance(md_property_mode, str) or (
        md_property_mode not in SUPPORTED_MD_PROPERTY_MODES
    ):
        raise ValueError(
            f"{config_path}: potential.md_property_mode must be one of "
            f"{sorted(SUPPORTED_MD_PROPERTY_MODES)}, got {md_property_mode!r}."
        )
    neighbor_skin_A = _positive_float(
        _required(potential_raw, "neighbor_skin_A", "potential", config_path),
        "potential.neighbor_skin_A",
        config_path,
    )
    potential_sha256 = str(
        _required(potential_raw, "sha256", "potential", config_path)
    ).lower()
    if len(potential_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in potential_sha256
    ):
        raise ValueError(
            f"{config_path}: potential.sha256 must be 64 lowercase hexadecimal characters, "
            f"got {potential_sha256!r}."
        )
    model_name = str(
        _required(potential_raw, "model_name", "potential", config_path)
    ).strip()
    family = str(_required(potential_raw, "family", "potential", config_path)).strip()
    source_url = str(
        _required(potential_raw, "source_url", "potential", config_path)
    ).strip()
    license_identifier = str(
        _required(potential_raw, "license_identifier", "potential", config_path)
    ).strip()
    for context, value in (
        ("potential.model_name", model_name),
        ("potential.family", family),
        ("potential.source_url", source_url),
        ("potential.license_identifier", license_identifier),
    ):
        if not value:
            raise ValueError(f"{config_path}: {context} must be a non-empty string.")
    potential_head_raw = _required(potential_raw, "head", "potential", config_path)
    if potential_head_raw is not None and not isinstance(potential_head_raw, str):
        raise TypeError(
            f"{config_path}: potential.head must be a string or null, "
            f"got {potential_head_raw!r}."
        )
    potential_head = (
        potential_head_raw.strip() if isinstance(potential_head_raw, str) else None
    )
    if isinstance(potential_head_raw, str) and not potential_head:
        raise ValueError(
            f"{config_path}: potential.head must be a non-empty string or null."
        )
    usage_mode = str(
        _required(potential_raw, "usage_mode", "potential", config_path)
    ).strip()
    if usage_mode not in {"exploratory", "quantitative"}:
        raise ValueError(
            f"{config_path}: potential.usage_mode must be 'exploratory' or "
            f"'quantitative', got {usage_mode!r}."
        )
    validation_report_raw = _required(
        potential_raw, "validation_report", "potential", config_path
    )
    validation_report = (
        None
        if validation_report_raw is None
        else _resolve_repo_path(validation_report_raw)
    )
    scientifically_qualified = False
    qualification = None
    validation_report_sha256 = None
    if validation_report is not None:
        if not validation_report.is_file():
            raise FileNotFoundError(
                f"{config_path}: potential.validation_report is not a file: "
                f"{validation_report}."
            )
        report_bytes = validation_report.read_bytes()
        validation_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        report = json.loads(report_bytes)
        if not isinstance(report, dict):
            raise TypeError(
                f"{validation_report}: validation report root must be a mapping, "
                f"got {type(report).__name__}."
            )
        if report.get("schema_version") != 1:
            raise RuntimeError(
                f"{validation_report}: expected validation report schema_version=1, "
                f"got {report.get('schema_version')!r}."
            )
        if report.get("report_type") != QUALIFICATION_REPORT_TYPE:
            raise RuntimeError(
                f"{validation_report}: expected report_type="
                f"{QUALIFICATION_REPORT_TYPE!r}, got {report.get('report_type')!r}."
            )
        scientifically_qualified_raw = _required(
            report,
            "scientifically_qualified",
            "report",
            validation_report,
        )
        if not isinstance(scientifically_qualified_raw, bool):
            raise TypeError(
                f"{validation_report}: report.scientifically_qualified must be a boolean, "
                f"got {scientifically_qualified_raw!r}."
            )
        scientifically_qualified = scientifically_qualified_raw
        qualification = _parse_potential_qualification(
            report,
            report_path=validation_report,
            expected_model_name=model_name,
            expected_model_sha256=potential_sha256,
            expected_head=potential_head,
            expected_device=device,
            expected_default_dtype=default_dtype,
            expected_enable_cueq=enable_cueq,
            expected_enable_oeq=enable_oeq,
            expected_compile_mode=compile_mode,
            expected_compile_fullgraph=compile_fullgraph,
            expected_pad_num_atoms=pad_num_atoms,
            expected_pad_num_edges=pad_num_edges,
            expected_md_property_mode=md_property_mode,
            expected_neighbor_skin_A=neighbor_skin_A,
        )
    if usage_mode == "quantitative" and not scientifically_qualified:
        raise RuntimeError(
            f"{config_path}: potential.usage_mode='quantitative' requires a validation_report "
            "for the exact model SHA/head with scientifically_qualified=true. The configured "
            "potential has not met that requirement."
        )

    config = GeneratorConfig(
        dataset_name=str(_required(raw, "dataset_name", "root", config_path)),
        random_seeds=tuple(random_seeds_raw),
        potential=PotentialConfig(
            model_name=model_name,
            family=family,
            model_path=model_path,
            sha256=potential_sha256,
            head=potential_head,
            source_url=source_url,
            license_identifier=license_identifier,
            usage_mode=usage_mode,
            validation_report=validation_report,
            validation_report_sha256=validation_report_sha256,
            scientifically_qualified=scientifically_qualified,
            qualification=qualification,
            device=device,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
            enable_oeq=enable_oeq,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            pad_num_atoms=pad_num_atoms,
            pad_num_edges=pad_num_edges,
            md_property_mode=md_property_mode,
            neighbor_skin_A=neighbor_skin_A,
        ),
        dynamics=DynamicsConfig(
            pressure_GPa=float(_required(dynamics_raw, "pressure_GPa", "dynamics", config_path)),
            target_temperature_K=target_temperature,
            melt_temperature_K=melt_temperature,
            timestep_fs=_positive_float(
                _required(dynamics_raw, "timestep_fs", "dynamics", config_path),
                "dynamics.timestep_fs",
                config_path,
            ),
            thermostat_time_fs=_positive_float(
                _required(dynamics_raw, "thermostat_time_fs", "dynamics", config_path),
                "dynamics.thermostat_time_fs",
                config_path,
            ),
            barostat_time_fs=_positive_float(
                _required(dynamics_raw, "barostat_time_fs", "dynamics", config_path),
                "dynamics.barostat_time_fs",
                config_path,
            ),
            solid_equilibration_steps=_nonnegative_int(
                _required(dynamics_raw, "solid_equilibration_steps", "dynamics", config_path),
                "dynamics.solid_equilibration_steps",
                config_path,
            ),
            melt_steps=_nonnegative_int(
                _required(dynamics_raw, "melt_steps", "dynamics", config_path),
                "dynamics.melt_steps",
                config_path,
            ),
            quench_steps=_nonnegative_int(
                _required(dynamics_raw, "quench_steps", "dynamics", config_path),
                "dynamics.quench_steps",
                config_path,
            ),
            quench_stages=_positive_int(
                _required(dynamics_raw, "quench_stages", "dynamics", config_path),
                "dynamics.quench_stages",
                config_path,
            ),
            target_equilibration_steps=_nonnegative_int(
                _required(dynamics_raw, "target_equilibration_steps", "dynamics", config_path),
                "dynamics.target_equilibration_steps",
                config_path,
            ),
            interface_evolution_steps=_nonnegative_int(
                _required(
                    dynamics_raw, "interface_evolution_steps", "dynamics", config_path
                ),
                "dynamics.interface_evolution_steps",
                config_path,
            ),
            sample_interval=_positive_int(
                _required(dynamics_raw, "sample_interval", "dynamics", config_path),
                "dynamics.sample_interval",
                config_path,
            ),
        ),
        system=SystemConfig(
            chemical_symbol=str(_required(system_raw, "chemical_symbol", "system", config_path)),
            crystal_structure=str(
                _required(system_raw, "crystal_structure", "system", config_path)
            ).lower(),
            initial_lattice_constant_A=_positive_float(
                _required(system_raw, "initial_lattice_constant_A", "system", config_path),
                "system.initial_lattice_constant_A",
                config_path,
            ),
            repetitions=repetitions,
            liquid_slab_fraction=slab_fraction,
            interface_half_width_A=_positive_float(
                _required(system_raw, "interface_half_width_A", "system", config_path),
                "system.interface_half_width_A",
                config_path,
            ),
        ),
        validation=ValidationConfig(
            maximum_force_eV_per_A=_positive_float(
                _required(validation_raw, "maximum_force_eV_per_A", "validation", config_path),
                "validation.maximum_force_eV_per_A",
                config_path,
            ),
            maximum_pressure_error_GPa=_positive_float(
                _required(validation_raw, "maximum_pressure_error_GPa", "validation", config_path),
                "validation.maximum_pressure_error_GPa",
                config_path,
            ),
            maximum_temperature_error_K=_positive_float(
                _required(
                    validation_raw,
                    "maximum_temperature_error_K",
                    "validation",
                    config_path,
                ),
                "validation.maximum_temperature_error_K",
                config_path,
            ),
            minimum_pair_distance_A=_positive_float(
                _required(validation_raw, "minimum_pair_distance_A", "validation", config_path),
                "validation.minimum_pair_distance_A",
                config_path,
            ),
            ptm_rmsd_cutoff=ptm_rmsd_cutoff,
            reference_density_cache=density_cache,
            maximum_relative_density_error=density_error,
            minimum_solid_fcc_fraction=minimum_solid_fcc_fraction,
            maximum_liquid_crystalline_fraction=maximum_liquid_crystalline_fraction,
            minimum_interface_crystalline_fraction=minimum_interface_crystalline_fraction,
            maximum_interface_crystalline_fraction=maximum_interface_crystalline_fraction,
        ),
        output=OutputConfig(
            root_dir=_resolve_repo_path(
                _required(output_raw, "root_dir", "output", config_path)
            ),
            overwrite=bool(output_raw.get("overwrite", False)),
            save_extxyz=bool(output_raw.get("save_extxyz", True)),
            create_visualizations=bool(output_raw.get("create_visualizations", True)),
        ),
        config_path=config_path,
    )
    validate_potential_qualification(
        config,
        chemical_symbol=config.system.chemical_symbol,
        pressure_GPa=config.dynamics.pressure_GPa,
        timestep_fs=config.dynamics.timestep_fs,
        state_temperatures_K={
            "solid_bulk": (config.dynamics.target_temperature_K,),
            "liquid_bulk": (
                config.dynamics.target_temperature_K,
                config.dynamics.melt_temperature_K,
            ),
            "interface": (
                config.dynamics.target_temperature_K,
                config.dynamics.melt_temperature_K,
            ),
        },
        context=f"base phase generator configuration {config_path}",
        required_claim="phase_context_structure",
    )
    return config
