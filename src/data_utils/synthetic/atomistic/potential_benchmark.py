from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from ase import Atoms, units
from ase.build import bulk
from ase.data import atomic_numbers
from ase.eos import EquationOfState
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .config import (
    POTENTIAL_CLAIM_NAMES,
    REPOSITORY_ROOT,
    load_config,
    potential_calculator_settings,
)
from .generator import select_calculator


@dataclass(frozen=True)
class DFTReferenceConfig:
    extxyz: Path
    code: str
    level_of_theory: str
    pseudopotential: str
    plane_wave_cutoff_eV: float
    kpoint_spacing_per_A: float
    source_url: str
    minimum_frames_per_state: int


@dataclass(frozen=True)
class QualificationScopeConfig:
    pressure_range_GPa: tuple[float, float]
    state_temperature_ranges_K: dict[str, tuple[float, float]]
    maximum_timestep_fs: float
    authorized_claims: dict[str, bool]


@dataclass(frozen=True)
class MeltingReferenceConfig:
    temperature_K: float
    source_url: str
    maximum_absolute_error_K: float
    minimum_protocol_count: int
    minimum_replicas_per_temperature: int
    minimum_production_duration_ps: float
    maximum_protocol_spread_K: float


@dataclass(frozen=True)
class PotentialBenchmarkConfig:
    model_configs: tuple[Path, ...]
    chemical_symbol: str
    reference_fcc_lattice_constant_A: float
    eos_volume_scales: tuple[float, ...]
    supercell_repetitions: tuple[int, int, int]
    nve_temperature_K: float
    nve_timesteps_fs: tuple[float, ...]
    nve_duration_ps: float
    nve_sample_interval_fs: float
    nve_random_seed: int
    application_structures: dict[str, Path]
    dft_reference: DFTReferenceConfig | None
    required_dft_states: tuple[str, ...]
    qualification_scope: QualificationScopeConfig
    melting_reference: MeltingReferenceConfig
    melting_scan_summaries: dict[str, tuple[Path, ...]]
    maximum_nve_drift_meV_per_atom_ps: float
    maximum_nve_excursion_meV_per_atom: float
    maximum_nve_detrended_rms_meV_per_atom: float
    maximum_nve_drift_difference_meV_per_atom_ps: float
    maximum_cache_energy_difference_meV_per_atom: float
    maximum_cache_force_rmse_eV_per_A: float
    maximum_cache_stress_difference_GPa: float
    maximum_dft_energy_rmse_meV_per_atom: float
    maximum_dft_force_rmse_eV_per_A: float
    maximum_dft_stress_rmse_GPa: float
    output_json: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return _serialize(result)


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(path)


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {key!r} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(
    mapping: dict[str, Any], allowed: set[str], context: str, path: Path
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise KeyError(f"{path}: unsupported keys in {context}: {unknown}.")


def _positive_float(value: Any, context: str, path: Path) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path}: {context} must be finite and > 0, got {result}.")
    return result


def _positive_int(value: Any, context: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: {context} must be a positive integer, got {value!r}.")
    return value


def _float_list(value: Any, context: str, path: Path) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{path}: {context} must be a non-empty list, got {value!r}.")
    result = tuple(float(item) for item in value)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: {context} contains non-finite values: {result}.")
    return result


def load_potential_benchmark_config(path: str | Path) -> PotentialBenchmarkConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{config_path}: root must be a mapping, got {type(raw).__name__}.")
    _reject_unknown(
        raw,
        {
            "model_configs",
            "chemical_symbol",
            "reference_fcc_lattice_constant_A",
            "equation_of_state",
            "nve",
            "application_structures",
            "dft_reference",
            "required_dft_states",
            "qualification_scope",
            "melting_reference",
            "melting_scan_summaries",
            "thresholds",
            "output_json",
        },
        "root",
        config_path,
    )
    eos_raw = _mapping(raw, "equation_of_state", config_path)
    nve_raw = _mapping(raw, "nve", config_path)
    scope_raw = _mapping(raw, "qualification_scope", config_path)
    melting_reference_raw = _mapping(raw, "melting_reference", config_path)
    thresholds_raw = _mapping(raw, "thresholds", config_path)
    _reject_unknown(
        eos_raw, {"volume_scales", "supercell_repetitions"}, "equation_of_state", config_path
    )
    _reject_unknown(
        nve_raw,
        {"temperature_K", "timesteps_fs", "duration_ps", "sample_interval_fs", "random_seed"},
        "nve",
        config_path,
    )
    _reject_unknown(
        scope_raw,
        {
            "pressure_range_GPa",
            "state_temperature_ranges_K",
            "maximum_timestep_fs",
            "authorized_claims",
        },
        "qualification_scope",
        config_path,
    )
    _reject_unknown(
        melting_reference_raw,
        {
            "temperature_K",
            "source_url",
            "maximum_absolute_error_K",
            "minimum_protocol_count",
            "minimum_replicas_per_temperature",
            "minimum_production_duration_ps",
            "maximum_protocol_spread_K",
        },
        "melting_reference",
        config_path,
    )
    _reject_unknown(
        thresholds_raw,
        {
            "maximum_nve_drift_meV_per_atom_ps",
            "maximum_nve_excursion_meV_per_atom",
            "maximum_nve_detrended_rms_meV_per_atom",
            "maximum_nve_drift_difference_meV_per_atom_ps",
            "maximum_cache_energy_difference_meV_per_atom",
            "maximum_cache_force_rmse_eV_per_A",
            "maximum_cache_stress_difference_GPa",
            "maximum_dft_energy_rmse_meV_per_atom",
            "maximum_dft_force_rmse_eV_per_A",
            "maximum_dft_stress_rmse_GPa",
        },
        "thresholds",
        config_path,
    )

    model_configs_raw = raw.get("model_configs")
    if not isinstance(model_configs_raw, list) or len(model_configs_raw) < 2:
        raise ValueError(
            f"{config_path}: model_configs must list at least two generator configs for a "
            f"comparison, got {model_configs_raw!r}."
        )
    model_configs = tuple(_repo_path(item) for item in model_configs_raw)
    for model_config in model_configs:
        if not model_config.is_file():
            raise FileNotFoundError(
                f"{config_path}: model generator config does not exist: {model_config}."
            )

    scales = _float_list(eos_raw.get("volume_scales"), "equation_of_state.volume_scales", config_path)
    if len(scales) < 5 or any(scale <= 0.0 for scale in scales):
        raise ValueError(
            f"{config_path}: equation_of_state.volume_scales must contain at least five "
            f"positive values bracketing 1.0, got {scales}."
        )
    if not min(scales) < 1.0 < max(scales):
        raise ValueError(
            f"{config_path}: equation_of_state.volume_scales must bracket 1.0, got {scales}."
        )
    repetitions_raw = eos_raw.get("supercell_repetitions")
    if (
        not isinstance(repetitions_raw, list)
        or len(repetitions_raw) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in repetitions_raw)
        or min(repetitions_raw) < 2
    ):
        raise ValueError(
            f"{config_path}: equation_of_state.supercell_repetitions must be three integers "
            f">= 2, got {repetitions_raw!r}."
        )

    timesteps = _float_list(nve_raw.get("timesteps_fs"), "nve.timesteps_fs", config_path)
    if len(timesteps) < 2 or any(timestep <= 0.0 for timestep in timesteps):
        raise ValueError(
            f"{config_path}: nve.timesteps_fs must contain at least two positive values for "
            f"a convergence check, got {timesteps}."
        )
    sample_interval_fs = _positive_float(
        nve_raw.get("sample_interval_fs"), "nve.sample_interval_fs", config_path
    )
    for timestep in timesteps:
        ratio = sample_interval_fs / timestep
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"{config_path}: nve.sample_interval_fs={sample_interval_fs} must be an "
                f"integer multiple of every timestep; {timestep} fs gives ratio {ratio}."
            )

    application_raw = raw.get("application_structures", {})
    if not isinstance(application_raw, dict):
        raise TypeError(
            f"{config_path}: application_structures must map labels to extxyz paths, got "
            f"{type(application_raw).__name__}."
        )
    application_structures = {
        str(label): _repo_path(structure_path)
        for label, structure_path in application_raw.items()
    }
    for label, structure_path in application_structures.items():
        if not structure_path.is_file():
            raise FileNotFoundError(
                f"{config_path}: application structure {label!r} does not exist: "
                f"{structure_path}."
            )

    dft_raw = raw.get("dft_reference")
    dft_reference: DFTReferenceConfig | None
    if dft_raw is None:
        dft_reference = None
    else:
        if not isinstance(dft_raw, dict):
            raise TypeError(
                f"{config_path}: dft_reference must be null or a provenance mapping, got "
                f"{type(dft_raw).__name__}."
            )
        _reject_unknown(
            dft_raw,
            {
                "extxyz",
                "code",
                "level_of_theory",
                "pseudopotential",
                "plane_wave_cutoff_eV",
                "kpoint_spacing_per_A",
                "source_url",
                "minimum_frames_per_state",
            },
            "dft_reference",
            config_path,
        )
        extxyz_path = _repo_path(dft_raw.get("extxyz"))
        if not extxyz_path.is_file():
            raise FileNotFoundError(
                f"{config_path}: dft_reference.extxyz does not exist: {extxyz_path}."
            )
        string_values = {
            key: str(dft_raw.get(key, "")).strip()
            for key in ("code", "level_of_theory", "pseudopotential", "source_url")
        }
        empty_fields = sorted(key for key, value in string_values.items() if not value)
        if empty_fields:
            raise ValueError(
                f"{config_path}: dft_reference fields must be non-empty: {empty_fields}."
            )
        minimum_frames_per_state = _positive_int(
            dft_raw.get("minimum_frames_per_state"),
            "dft_reference.minimum_frames_per_state",
            config_path,
        )
        if minimum_frames_per_state < 5:
            raise ValueError(
                f"{config_path}: dft_reference.minimum_frames_per_state must be >= 5 "
                f"independent configurations for every state, got "
                f"{minimum_frames_per_state}."
            )
        dft_reference = DFTReferenceConfig(
            extxyz=extxyz_path,
            code=string_values["code"],
            level_of_theory=string_values["level_of_theory"],
            pseudopotential=string_values["pseudopotential"],
            plane_wave_cutoff_eV=_positive_float(
                dft_raw.get("plane_wave_cutoff_eV"),
                "dft_reference.plane_wave_cutoff_eV",
                config_path,
            ),
            kpoint_spacing_per_A=_positive_float(
                dft_raw.get("kpoint_spacing_per_A"),
                "dft_reference.kpoint_spacing_per_A",
                config_path,
            ),
            source_url=string_values["source_url"],
            minimum_frames_per_state=minimum_frames_per_state,
        )
    states_raw = raw.get("required_dft_states")
    if not isinstance(states_raw, list) or not states_raw or any(
        not isinstance(item, str) or not item for item in states_raw
    ):
        raise TypeError(
            f"{config_path}: required_dft_states must be a non-empty list of names, got "
            f"{states_raw!r}."
        )
    if len(set(states_raw)) != len(states_raw):
        raise ValueError(f"{config_path}: required_dft_states contains duplicates: {states_raw!r}.")

    pressure_range = _float_list(
        scope_raw.get("pressure_range_GPa"),
        "qualification_scope.pressure_range_GPa",
        config_path,
    )
    if len(pressure_range) != 2 or pressure_range[0] > pressure_range[1]:
        raise ValueError(
            f"{config_path}: qualification_scope.pressure_range_GPa must be [lower, upper], "
            f"got {pressure_range}."
        )
    state_ranges_raw = scope_raw.get("state_temperature_ranges_K")
    if not isinstance(state_ranges_raw, dict):
        raise TypeError(
            f"{config_path}: qualification_scope.state_temperature_ranges_K must map every "
            f"required DFT state to [lower, upper], got {type(state_ranges_raw).__name__}."
        )
    if set(state_ranges_raw) != set(states_raw):
        raise ValueError(
            f"{config_path}: qualification scope states must exactly match "
            f"required_dft_states; scope={sorted(state_ranges_raw)}, "
            f"required={sorted(states_raw)}."
        )
    state_temperature_ranges: dict[str, tuple[float, float]] = {}
    for state, state_range_raw in state_ranges_raw.items():
        state_range = _float_list(
            state_range_raw,
            f"qualification_scope.state_temperature_ranges_K.{state}",
            config_path,
        )
        if len(state_range) != 2 or state_range[0] <= 0.0 or state_range[0] > state_range[1]:
            raise ValueError(
                f"{config_path}: qualification temperature range for {state!r} must be "
                f"positive [lower, upper], got {state_range}."
            )
        state_temperature_ranges[str(state)] = (state_range[0], state_range[1])

    authorized_claims_raw = scope_raw.get("authorized_claims")
    if not isinstance(authorized_claims_raw, dict):
        raise TypeError(
            f"{config_path}: qualification_scope.authorized_claims must be a mapping "
            f"with exactly {list(POTENTIAL_CLAIM_NAMES)}, got "
            f"{type(authorized_claims_raw).__name__}."
        )
    missing_claims = sorted(set(POTENTIAL_CLAIM_NAMES) - set(authorized_claims_raw))
    unknown_claims = sorted(set(authorized_claims_raw) - set(POTENTIAL_CLAIM_NAMES))
    if missing_claims or unknown_claims:
        raise KeyError(
            f"{config_path}: qualification_scope.authorized_claims must contain exactly "
            f"{list(POTENTIAL_CLAIM_NAMES)}; missing={missing_claims}, "
            f"unknown={unknown_claims}."
        )
    if any(type(authorized_claims_raw[claim]) is not bool for claim in POTENTIAL_CLAIM_NAMES):
        raise TypeError(
            f"{config_path}: every qualification_scope.authorized_claims value must be an "
            f"exact boolean, got {authorized_claims_raw!r}."
        )
    authorized_claims = {
        claim: authorized_claims_raw[claim] for claim in POTENTIAL_CLAIM_NAMES
    }
    if authorized_claims["kinetics"]:
        raise ValueError(
            f"{config_path}: this benchmark cannot authorize kinetics because it does not "
            "validate liquid transport coefficients or kinetic-time rescaling; set "
            "qualification_scope.authorized_claims.kinetics=false."
        )
    maximum_timestep_fs = _positive_float(
        scope_raw.get("maximum_timestep_fs"),
        "qualification_scope.maximum_timestep_fs",
        config_path,
    )
    if maximum_timestep_fs not in timesteps:
        raise ValueError(
            f"{config_path}: qualification_scope.maximum_timestep_fs="
            f"{maximum_timestep_fs} must be explicitly included in nve.timesteps_fs="
            f"{list(timesteps)}."
        )

    melting_raw = raw.get("melting_scan_summaries")
    if not isinstance(melting_raw, dict):
        raise TypeError(
            f"{config_path}: melting_scan_summaries must map model_name to a list of "
            f"velocity_summary.json paths, got {type(melting_raw).__name__}."
        )
    melting_summaries: dict[str, tuple[Path, ...]] = {}
    for model_name, summary_paths_raw in melting_raw.items():
        if not isinstance(summary_paths_raw, list) or not summary_paths_raw:
            raise TypeError(
                f"{config_path}: melting_scan_summaries.{model_name} must be a non-empty "
                f"list of paths, got {summary_paths_raw!r}."
            )
        summary_paths = tuple(_repo_path(item) for item in summary_paths_raw)
        for summary_path in summary_paths:
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"{config_path}: melting scan for model {model_name!r} does not exist: "
                    f"{summary_path}."
                )
        melting_summaries[str(model_name)] = summary_paths
    output_json = _repo_path(raw.get("output_json"))

    random_seed = nve_raw.get("random_seed")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError(
            f"{config_path}: nve.random_seed must be an integer, got {random_seed!r}."
        )
    melting_source_url = str(melting_reference_raw.get("source_url", "")).strip()
    if not melting_source_url:
        raise ValueError(
            f"{config_path}: melting_reference.source_url must identify the physical "
            f"reference used to judge the model melting point."
        )
    return PotentialBenchmarkConfig(
        model_configs=model_configs,
        chemical_symbol=str(raw.get("chemical_symbol")),
        reference_fcc_lattice_constant_A=_positive_float(
            raw.get("reference_fcc_lattice_constant_A"),
            "reference_fcc_lattice_constant_A",
            config_path,
        ),
        eos_volume_scales=scales,
        supercell_repetitions=tuple(repetitions_raw),
        nve_temperature_K=_positive_float(
            nve_raw.get("temperature_K"), "nve.temperature_K", config_path
        ),
        nve_timesteps_fs=timesteps,
        nve_duration_ps=_positive_float(
            nve_raw.get("duration_ps"), "nve.duration_ps", config_path
        ),
        nve_sample_interval_fs=sample_interval_fs,
        nve_random_seed=random_seed,
        application_structures=application_structures,
        dft_reference=dft_reference,
        required_dft_states=tuple(states_raw),
        qualification_scope=QualificationScopeConfig(
            pressure_range_GPa=(pressure_range[0], pressure_range[1]),
            state_temperature_ranges_K=state_temperature_ranges,
            maximum_timestep_fs=maximum_timestep_fs,
            authorized_claims=authorized_claims,
        ),
        melting_reference=MeltingReferenceConfig(
            temperature_K=_positive_float(
                melting_reference_raw.get("temperature_K"),
                "melting_reference.temperature_K",
                config_path,
            ),
            source_url=melting_source_url,
            maximum_absolute_error_K=_positive_float(
                melting_reference_raw.get("maximum_absolute_error_K"),
                "melting_reference.maximum_absolute_error_K",
                config_path,
            ),
            minimum_protocol_count=_positive_int(
                melting_reference_raw.get("minimum_protocol_count"),
                "melting_reference.minimum_protocol_count",
                config_path,
            ),
            minimum_replicas_per_temperature=_positive_int(
                melting_reference_raw.get("minimum_replicas_per_temperature"),
                "melting_reference.minimum_replicas_per_temperature",
                config_path,
            ),
            minimum_production_duration_ps=_positive_float(
                melting_reference_raw.get("minimum_production_duration_ps"),
                "melting_reference.minimum_production_duration_ps",
                config_path,
            ),
            maximum_protocol_spread_K=_positive_float(
                melting_reference_raw.get("maximum_protocol_spread_K"),
                "melting_reference.maximum_protocol_spread_K",
                config_path,
            ),
        ),
        melting_scan_summaries=melting_summaries,
        maximum_nve_drift_meV_per_atom_ps=_positive_float(
            thresholds_raw.get("maximum_nve_drift_meV_per_atom_ps"),
            "thresholds.maximum_nve_drift_meV_per_atom_ps",
            config_path,
        ),
        maximum_nve_excursion_meV_per_atom=_positive_float(
            thresholds_raw.get("maximum_nve_excursion_meV_per_atom"),
            "thresholds.maximum_nve_excursion_meV_per_atom",
            config_path,
        ),
        maximum_nve_detrended_rms_meV_per_atom=_positive_float(
            thresholds_raw.get("maximum_nve_detrended_rms_meV_per_atom"),
            "thresholds.maximum_nve_detrended_rms_meV_per_atom",
            config_path,
        ),
        maximum_nve_drift_difference_meV_per_atom_ps=_positive_float(
            thresholds_raw.get("maximum_nve_drift_difference_meV_per_atom_ps"),
            "thresholds.maximum_nve_drift_difference_meV_per_atom_ps",
            config_path,
        ),
        maximum_cache_energy_difference_meV_per_atom=_positive_float(
            thresholds_raw.get("maximum_cache_energy_difference_meV_per_atom"),
            "thresholds.maximum_cache_energy_difference_meV_per_atom",
            config_path,
        ),
        maximum_cache_force_rmse_eV_per_A=_positive_float(
            thresholds_raw.get("maximum_cache_force_rmse_eV_per_A"),
            "thresholds.maximum_cache_force_rmse_eV_per_A",
            config_path,
        ),
        maximum_cache_stress_difference_GPa=_positive_float(
            thresholds_raw.get("maximum_cache_stress_difference_GPa"),
            "thresholds.maximum_cache_stress_difference_GPa",
            config_path,
        ),
        maximum_dft_energy_rmse_meV_per_atom=_positive_float(
            thresholds_raw.get("maximum_dft_energy_rmse_meV_per_atom"),
            "thresholds.maximum_dft_energy_rmse_meV_per_atom",
            config_path,
        ),
        maximum_dft_force_rmse_eV_per_A=_positive_float(
            thresholds_raw.get("maximum_dft_force_rmse_eV_per_A"),
            "thresholds.maximum_dft_force_rmse_eV_per_A",
            config_path,
        ),
        maximum_dft_stress_rmse_GPa=_positive_float(
            thresholds_raw.get("maximum_dft_stress_rmse_GPa"),
            "thresholds.maximum_dft_stress_rmse_GPa",
            config_path,
        ),
        output_json=output_json,
        config_path=config_path,
    )


def _phase_cell(
    symbol: str,
    phase: str,
    reference_fcc_a_A: float,
    *,
    hcp_c_over_a: float | None = None,
) -> Atoms:
    fcc_atomic_volume_A3 = reference_fcc_a_A**3 / 4.0
    if phase == "fcc":
        return bulk(symbol, "fcc", a=reference_fcc_a_A, cubic=True)
    if phase == "bcc":
        bcc_a_A = (2.0 * fcc_atomic_volume_A3) ** (1.0 / 3.0)
        return bulk(symbol, "bcc", a=bcc_a_A, cubic=True)
    if phase == "hcp":
        ideal_c_over_a = (
            np.sqrt(8.0 / 3.0) if hcp_c_over_a is None else hcp_c_over_a
        )
        hcp_a_A = (
            4.0 * fcc_atomic_volume_A3 / (np.sqrt(3.0) * ideal_c_over_a)
        ) ** (1.0 / 3.0)
        return bulk(symbol, "hcp", a=hcp_a_A, c=ideal_c_over_a * hcp_a_A)
    raise ValueError(f"Unsupported phase {phase!r}; expected fcc, hcp, or bcc.")


def equation_of_state_metrics(
    calculator: object,
    *,
    symbol: str,
    phase: str,
    reference_fcc_a_A: float,
    volume_scales: tuple[float, ...],
    repetitions: tuple[int, int, int],
    _hcp_c_over_a: float | None = None,
) -> dict[str, Any]:
    base = _phase_cell(
        symbol,
        phase,
        reference_fcc_a_A,
        hcp_c_over_a=_hcp_c_over_a,
    ).repeat(repetitions)
    volumes_per_atom_A3: list[float] = []
    energies_per_atom_eV: list[float] = []
    maximum_forces_eV_per_A: list[float] = []
    for volume_scale in volume_scales:
        atoms = base.copy()
        atoms.set_cell(
            np.asarray(atoms.cell) * volume_scale ** (1.0 / 3.0),
            scale_atoms=True,
        )
        atoms.calc = calculator
        energy_eV = float(atoms.get_potential_energy())
        forces_eV_per_A = np.asarray(atoms.get_forces(), dtype=np.float64)
        volumes_per_atom_A3.append(float(atoms.get_volume() / len(atoms)))
        energies_per_atom_eV.append(energy_eV / len(atoms))
        maximum_forces_eV_per_A.append(
            float(np.linalg.norm(forces_eV_per_A, axis=1).max())
        )
    sampled_values = np.asarray(
        [volumes_per_atom_A3, energies_per_atom_eV, maximum_forces_eV_per_A],
        dtype=np.float64,
    )
    if not np.isfinite(sampled_values).all():
        raise FloatingPointError(f"{phase}: EOS scan produced non-finite values.")
    minimum_index = int(np.argmin(energies_per_atom_eV))
    if minimum_index in {0, len(volume_scales) - 1}:
        raise RuntimeError(
            f"{phase}: minimum sampled energy occurs at the edge of the EOS volume scan "
            f"(scale={volume_scales[minimum_index]}). Widen the configured volume bracket."
        )
    eos = EquationOfState(volumes_per_atom_A3, energies_per_atom_eV, eos="birchmurnaghan")
    equilibrium_volume_A3, equilibrium_energy_eV, bulk_modulus_eV_per_A3 = eos.fit()
    if (
        not np.isfinite(equilibrium_volume_A3)
        or not np.isfinite(equilibrium_energy_eV)
        or not np.isfinite(bulk_modulus_eV_per_A3)
        or bulk_modulus_eV_per_A3 <= 0.0
    ):
        raise RuntimeError(
            f"{phase}: invalid EOS fit volume={equilibrium_volume_A3}, "
            f"energy={equilibrium_energy_eV}, bulk_modulus={bulk_modulus_eV_per_A3}."
        )
    if not min(volumes_per_atom_A3) < equilibrium_volume_A3 < max(volumes_per_atom_A3):
        raise RuntimeError(
            f"{phase}: fitted equilibrium volume {equilibrium_volume_A3:.8f} A^3/atom lies "
            f"outside sampled range [{min(volumes_per_atom_A3):.8f}, "
            f"{max(volumes_per_atom_A3):.8f}]."
        )
    result: dict[str, Any] = {
        "sampled_volume_per_atom_A3": volumes_per_atom_A3,
        "sampled_energy_per_atom_eV": energies_per_atom_eV,
        "sampled_maximum_force_eV_per_A": maximum_forces_eV_per_A,
        "equilibrium_volume_per_atom_A3": float(equilibrium_volume_A3),
        "equilibrium_energy_per_atom_eV": float(equilibrium_energy_eV),
        "bulk_modulus_GPa": float(bulk_modulus_eV_per_A3 / units.GPa),
    }
    if phase == "fcc":
        result["equilibrium_lattice_constant_A"] = float(
            (4.0 * equilibrium_volume_A3) ** (1.0 / 3.0)
        )
    if phase == "hcp" and _hcp_c_over_a is None:
        ratios = np.linspace(1.54, 1.72, 10, dtype=np.float64)
        ratio_energies: list[float] = []
        for ratio in ratios:
            hcp_a_A = (
                4.0
                * equilibrium_volume_A3
                / (np.sqrt(3.0) * float(ratio))
            ) ** (1.0 / 3.0)
            atoms = bulk(
                symbol,
                "hcp",
                a=hcp_a_A,
                c=float(ratio) * hcp_a_A,
            ).repeat(repetitions)
            atoms.calc = calculator
            ratio_energies.append(float(atoms.get_potential_energy() / len(atoms)))
        best_index = int(np.argmin(ratio_energies))
        if best_index in {0, len(ratios) - 1}:
            raise RuntimeError(
                "hcp: minimum c/a energy lies at the edge of scan "
                f"[{ratios[0]}, {ratios[-1]}]; energies={ratio_energies}."
            )
        best_ratio = float(ratios[best_index])
        relaxed = equation_of_state_metrics(
            calculator,
            symbol=symbol,
            phase=phase,
            reference_fcc_a_A=reference_fcc_a_A,
            volume_scales=volume_scales,
            repetitions=repetitions,
            _hcp_c_over_a=best_ratio,
        )
        relaxed["c_over_a_scan"] = {
            "ratios": ratios.tolist(),
            "energy_per_atom_eV": ratio_energies,
            "selected_ratio": best_ratio,
        }
        return relaxed
    if phase == "hcp":
        result["equilibrium_c_over_a"] = float(_hcp_c_over_a)
    return result


def nve_energy_conservation(
    calculator: object,
    *,
    symbol: str,
    lattice_constant_A: float,
    repetitions: tuple[int, int, int],
    temperature_K: float,
    timestep_fs: float,
    duration_ps: float,
    sample_interval_fs: float,
    random_seed: int,
) -> dict[str, Any]:
    atoms = bulk(symbol, "fcc", a=lattice_constant_A, cubic=True).repeat(repetitions)
    return nve_energy_conservation_from_atoms(
        atoms,
        calculator,
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        duration_ps=duration_ps,
        sample_interval_fs=sample_interval_fs,
        random_seed=random_seed,
    )


def nve_energy_conservation_from_atoms(
    initial_atoms: Atoms,
    calculator: object,
    *,
    temperature_K: float,
    timestep_fs: float,
    duration_ps: float,
    sample_interval_fs: float,
    random_seed: int,
) -> dict[str, Any]:
    atoms = initial_atoms.copy()
    atoms.calc = calculator
    rng = np.random.default_rng(random_seed)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, force_temp=True, rng=rng
    )
    Stationary(atoms, preserve_temperature=True)
    total_steps = int(round(duration_ps * 1000.0 / timestep_fs))
    if not np.isclose(total_steps * timestep_fs, duration_ps * 1000.0, atol=1.0e-10):
        raise ValueError(
            f"NVE duration {duration_ps} ps is not an integer number of {timestep_fs} fs steps."
        )
    sample_steps = int(round(sample_interval_fs / timestep_fs))
    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    times_ps = [0.0]
    energies_eV_per_atom = [float(atoms.get_total_energy() / len(atoms))]
    temperatures_K = [float(atoms.get_temperature())]
    while dynamics.nsteps < total_steps:
        run_steps = min(sample_steps, total_steps - dynamics.nsteps)
        dynamics.run(run_steps)
        times_ps.append(float(dynamics.nsteps * timestep_fs / 1000.0))
        energies_eV_per_atom.append(float(atoms.get_total_energy() / len(atoms)))
        temperatures_K.append(float(atoms.get_temperature()))
    time_array = np.asarray(times_ps, dtype=np.float64)
    energy_array = np.asarray(energies_eV_per_atom, dtype=np.float64)
    temperature_array = np.asarray(temperatures_K, dtype=np.float64)
    if not np.isfinite(energy_array).all() or not np.isfinite(temperature_array).all():
        raise FloatingPointError(
            f"NVE run produced non-finite energy or temperature at timestep={timestep_fs} fs."
        )
    slope_eV_per_atom_ps, intercept_eV_per_atom = np.polyfit(time_array, energy_array, 1)
    fitted = slope_eV_per_atom_ps * time_array + intercept_eV_per_atom
    result = {
        "timestep_fs": timestep_fs,
        "duration_ps": duration_ps,
        "sample_interval_fs": sample_interval_fs,
        "drift_meV_per_atom_ps": float(1000.0 * slope_eV_per_atom_ps),
        "maximum_excursion_meV_per_atom": float(
            1000.0 * np.max(np.abs(energy_array - energy_array[0]))
        ),
        "detrended_rms_meV_per_atom": float(
            1000.0 * np.sqrt(np.mean(np.square(energy_array - fitted)))
        ),
        "initial_temperature_K": temperatures_K[0],
        "mean_temperature_K": float(np.mean(temperature_array)),
        "final_temperature_K": temperatures_K[-1],
        "sample_count": len(times_ps),
    }
    numeric_values = [
        value
        for key, value in result.items()
        if key not in {"sample_count"}
    ]
    if not np.isfinite(numeric_values).all():
        raise FloatingPointError(
            f"NVE diagnostics are non-finite at timestep={timestep_fs} fs: {result}."
        )
    return result


def _evaluate_structure(
    atoms: Atoms, calculator: object
) -> tuple[dict[str, Any], np.ndarray]:
    evaluated = atoms.copy()
    evaluated.calc = calculator
    forces = np.asarray(evaluated.get_forces(), dtype=np.float64)
    stress = np.asarray(evaluated.get_stress(voigt=True), dtype=np.float64)
    energy = float(evaluated.get_potential_energy())
    if (
        not np.isfinite(energy)
        or not np.isfinite(forces).all()
        or not np.isfinite(stress).all()
    ):
        raise FloatingPointError("Application-structure MLIP evaluation is non-finite.")
    summary = {
        "atom_count": len(evaluated),
        "energy_per_atom_eV": float(energy / len(evaluated)),
        "force_rms_eV_per_A": float(np.sqrt(np.mean(np.square(forces)))),
        "maximum_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
        "stress_GPa": (stress / units.GPa).tolist(),
        "pressure_GPa": float(-np.mean(stress[:3]) / units.GPa),
    }
    return summary, forces


def verlet_cache_parity(
    calculator: object,
    *,
    chemical_symbol: str,
    lattice_constant_A: float,
    repetitions: tuple[int, int, int],
    random_seed: int,
) -> dict[str, float | int]:
    required_attributes = (
        "_cached_batch",
        "_reference_cell_A",
        "_reference_scaled_positions",
        "graph_rebuild_count",
        "graph_reuse_count",
        "reset",
    )
    missing = [name for name in required_attributes if not hasattr(calculator, name)]
    if missing:
        raise TypeError(
            "Production cache parity requires VerletSkinMACECalculator attributes; "
            f"missing={missing}, calculator={type(calculator).__module__}."
            f"{type(calculator).__qualname__}."
        )
    base = bulk(
        chemical_symbol, "fcc", a=lattice_constant_A, cubic=True
    ).repeat(repetitions)
    _evaluate_structure(base, calculator)
    perturbed = base.copy()
    rng = np.random.default_rng(random_seed)
    perturbed.positions += rng.normal(0.0, 0.015, size=perturbed.positions.shape)
    reuse_before = int(calculator.graph_reuse_count)
    cached_summary, cached_forces = _evaluate_structure(perturbed, calculator)
    reuse_after = int(calculator.graph_reuse_count)
    if reuse_after <= reuse_before:
        raise RuntimeError(
            "Cache parity perturbation did not exercise graph reuse; reduce the configured "
            "perturbation or inspect the Verlet validity bound."
        )
    calculator._cached_batch = None
    calculator._reference_cell_A = None
    calculator._reference_scaled_positions = None
    calculator.reset()
    fresh_summary, fresh_forces = _evaluate_structure(perturbed, calculator)
    result: dict[str, float | int] = {
        "cached_graph_reuse_increment": reuse_after - reuse_before,
        "energy_difference_meV_per_atom": float(
            1000.0
            * abs(
                cached_summary["energy_per_atom_eV"]
                - fresh_summary["energy_per_atom_eV"]
            )
        ),
        "force_rmse_eV_per_A": float(
            np.sqrt(np.mean(np.square(cached_forces - fresh_forces)))
        ),
        "maximum_stress_difference_GPa": float(
            np.max(
                np.abs(
                    np.asarray(cached_summary["stress_GPa"])
                    - np.asarray(fresh_summary["stress_GPa"])
                )
            )
        ),
    }
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError(f"Verlet cache parity produced non-finite metrics: {result}.")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _geometry_sha256(atoms: Atoms) -> str:
    """Hash periodic geometry independent of atom order, wrapping, and metadata labels."""

    digest = hashlib.sha256()
    numbers = np.asarray(atoms.numbers, dtype=np.dtype("<i8"))
    wrapped_fractional = np.asarray(
        atoms.get_scaled_positions(wrap=True), dtype=np.float64
    )
    if not np.isfinite(wrapped_fractional).all():
        raise FloatingPointError("Cannot hash a geometry with non-finite positions.")
    fractional_resolution = 10**12
    quantized_fractional = (
        np.rint(wrapped_fractional * fractional_resolution).astype(np.int64)
        % fractional_resolution
    )
    order = np.lexsort(
        (
            quantized_fractional[:, 2],
            quantized_fractional[:, 1],
            quantized_fractional[:, 0],
            numbers,
        )
    )
    arrays = (
        numbers[order],
        np.asarray(atoms.pbc, dtype=np.uint8),
        np.asarray(atoms.cell.array, dtype=np.dtype("<f8")),
        np.asarray(quantized_fractional[order], dtype=np.dtype("<i8")),
    )
    for name, array in zip(
        ("numbers", "pbc", "cell", "wrapped_fractional_1e-12"), arrays
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(contiguous.shape, dtype=np.dtype("<i8")).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_reference_frames(
    reference: DFTReferenceConfig,
    *,
    chemical_symbol: str,
    required_states: tuple[str, ...],
    scope: QualificationScopeConfig,
) -> tuple[list[Atoms], dict[str, Any]]:
    path = reference.extxyz
    if reference.minimum_frames_per_state < 5:
        raise ValueError(
            f"{path}: dft_reference.minimum_frames_per_state must be >= 5, got "
            f"{reference.minimum_frames_per_state}."
        )
    frames = read(path, index=":")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"DFT reference file contains no frames: {path}.")
    expected_atomic_number = atomic_numbers[chemical_symbol]
    state_temperatures: dict[str, list[float]] = {
        state: [] for state in required_states
    }
    state_pressures: dict[str, list[float]] = {
        state: [] for state in required_states
    }
    configuration_ids: set[str] = set()
    geometry_hash_to_configuration_id: dict[str, str] = {}
    for frame_index, atoms in enumerate(frames):
        state = atoms.info.get("state")
        if state not in state_temperatures:
            raise ValueError(
                f"{path}: DFT reference frame {frame_index} has state={state!r}; every "
                f"frame must use one of the required states {required_states}."
            )
        configuration_id = atoms.info.get("configuration_id")
        if not isinstance(configuration_id, str) or not configuration_id:
            raise ValueError(
                f"{path}: DFT reference frame {frame_index} must define a non-empty unique "
                f"Atoms.info['configuration_id']."
            )
        if configuration_id in configuration_ids:
            raise ValueError(
                f"{path}: duplicate DFT configuration_id={configuration_id!r}; repeated "
                f"frames do not count as independent validation configurations."
            )
        configuration_ids.add(configuration_id)
        geometry_sha256 = _geometry_sha256(atoms)
        duplicate_configuration_id = geometry_hash_to_configuration_id.get(
            geometry_sha256
        )
        if duplicate_configuration_id is not None:
            raise ValueError(
                f"{path}: DFT configuration_id={configuration_id!r} duplicates the exact "
                f"periodic geometry of configuration_id={duplicate_configuration_id!r} "
                f"(geometry_sha256={geometry_sha256}); changing configuration_id does not "
                "make a repeated frame independent."
            )
        geometry_hash_to_configuration_id[geometry_sha256] = configuration_id
        temperature_K = atoms.info.get("temperature_K")
        target_pressure_GPa = atoms.info.get("target_pressure_GPa")
        if (
            not isinstance(temperature_K, (int, float))
            or isinstance(temperature_K, bool)
            or not np.isfinite(temperature_K)
        ):
            raise ValueError(
                f"{path}: frame {configuration_id!r} must define finite numeric "
                f"Atoms.info['temperature_K'], got {temperature_K!r}."
            )
        if (
            not isinstance(target_pressure_GPa, (int, float))
            or isinstance(target_pressure_GPa, bool)
            or not np.isfinite(target_pressure_GPa)
            or not scope.pressure_range_GPa[0]
            <= float(target_pressure_GPa)
            <= scope.pressure_range_GPa[1]
        ):
            raise ValueError(
                f"{path}: frame {configuration_id!r} target_pressure_GPa="
                f"{target_pressure_GPa!r} is not finite and within qualification scope "
                f"{scope.pressure_range_GPa}."
            )
        state_temperatures[str(state)].append(float(temperature_K))
        state_pressures[str(state)].append(float(target_pressure_GPa))
        expected_info = {
            "reference_code": reference.code,
            "reference_level_of_theory": reference.level_of_theory,
            "reference_pseudopotential": reference.pseudopotential,
        }
        for info_key, expected_value in expected_info.items():
            if atoms.info.get(info_key) != expected_value:
                raise ValueError(
                    f"{path}: frame {configuration_id!r} {info_key}="
                    f"{atoms.info.get(info_key)!r}, expected {expected_value!r} from the "
                    f"benchmark's DFT provenance."
                )
        for info_key, expected_value in (
            ("reference_plane_wave_cutoff_eV", reference.plane_wave_cutoff_eV),
            ("reference_kpoint_spacing_per_A", reference.kpoint_spacing_per_A),
        ):
            observed_value = atoms.info.get(info_key)
            if not isinstance(observed_value, (int, float)) or not np.isclose(
                float(observed_value), expected_value, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"{path}: frame {configuration_id!r} {info_key}={observed_value!r}, "
                    f"expected {expected_value}."
                )
        numbers = np.asarray(atoms.numbers)
        if len(atoms) < 2 or not np.all(numbers == expected_atomic_number):
            raise ValueError(
                f"{path}: frame {configuration_id!r} must contain at least two and only "
                f"{chemical_symbol} atoms; atomic_numbers={sorted(set(numbers.tolist()))}."
            )
        if not bool(np.all(atoms.pbc)):
            raise ValueError(
                f"{path}: frame {configuration_id!r} must be periodic in all axes for bulk, "
                f"interface, stress, and NPT validation; pbc={atoms.pbc.tolist()}."
            )
        positions = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        if not np.isfinite(positions).all() or not np.isfinite(cell).all():
            raise FloatingPointError(
                f"{path}: frame {configuration_id!r} has non-finite positions or cell."
            )
        if float(np.linalg.det(cell)) <= 0.0:
            raise ValueError(
                f"{path}: frame {configuration_id!r} has non-positive cell determinant "
                f"{float(np.linalg.det(cell))}."
            )
        calculator = atoms.calc
        results = {} if calculator is None else calculator.results
        missing = sorted({"energy", "forces", "stress"} - set(results))
        if missing:
            raise ValueError(
                f"{path}: DFT reference frame {frame_index} state={state!r} lacks calculator "
                f"results {missing}; energy, forces, and stress are required for NPT MD use."
            )
        reference_energy = float(atoms.get_potential_energy())
        reference_forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        reference_stress = np.asarray(atoms.get_stress(voigt=True), dtype=np.float64)
        if (
            not np.isfinite(reference_energy)
            or reference_forces.shape != (len(atoms), 3)
            or reference_stress.shape != (6,)
            or not np.isfinite(reference_forces).all()
            or not np.isfinite(reference_stress).all()
        ):
            raise FloatingPointError(
                f"{path}: frame {configuration_id!r} has invalid/non-finite reference "
                f"energy, forces, or stress."
            )
    for state, temperatures in state_temperatures.items():
        if len(temperatures) < reference.minimum_frames_per_state:
            raise ValueError(
                f"{path}: state={state!r} has {len(temperatures)} independent frames, below "
                f"dft_reference.minimum_frames_per_state="
                f"{reference.minimum_frames_per_state}."
            )
        required_range = scope.state_temperature_ranges_K[state]
        if min(temperatures) > required_range[0] or max(temperatures) < required_range[1]:
            raise ValueError(
                f"{path}: state={state!r} temperature coverage "
                f"[{min(temperatures)}, {max(temperatures)}] K does not span qualification "
                f"range {required_range}."
            )
        pressures = state_pressures[state]
        if (
            min(pressures) > scope.pressure_range_GPa[0]
            or max(pressures) < scope.pressure_range_GPa[1]
        ):
            raise ValueError(
                f"{path}: state={state!r} target-pressure coverage "
                f"[{min(pressures)}, {max(pressures)}] GPa does not span qualification "
                f"range {scope.pressure_range_GPa}."
            )
    evidence = {
        "path": str(path),
        "sha256": _sha256_file(path),
        "frame_count": len(frames),
        "configuration_ids_unique": True,
        "geometry_hashes_unique": True,
        "state_frame_counts": {
            state: len(temperatures) for state, temperatures in state_temperatures.items()
        },
        "state_temperature_ranges_K": {
            state: [min(temperatures), max(temperatures)]
            for state, temperatures in state_temperatures.items()
        },
        "state_target_pressure_ranges_GPa": {
            state: [min(pressures), max(pressures)]
            for state, pressures in state_pressures.items()
        },
        "code": reference.code,
        "level_of_theory": reference.level_of_theory,
        "pseudopotential": reference.pseudopotential,
        "plane_wave_cutoff_eV": reference.plane_wave_cutoff_eV,
        "kpoint_spacing_per_A": reference.kpoint_spacing_per_A,
        "source_url": reference.source_url,
    }
    return frames, evidence


def dft_error_metrics(frames: list[Atoms], calculator: object) -> dict[str, Any]:
    energy_errors_eV_per_atom: list[float] = []
    force_errors_eV_per_A: list[np.ndarray] = []
    stress_errors_GPa: list[np.ndarray] = []
    frame_states: list[str] = []
    for atoms in frames:
        reference_energy_eV = float(atoms.get_potential_energy())
        reference_forces = np.asarray(atoms.get_forces(), dtype=np.float64).copy()
        reference_stress = np.asarray(atoms.get_stress(voigt=True), dtype=np.float64).copy()
        state = str(atoms.info["state"])

        predicted = atoms.copy()
        predicted.calc = calculator
        predicted_energy_eV = float(predicted.get_potential_energy())
        predicted_forces = np.asarray(predicted.get_forces(), dtype=np.float64)
        predicted_stress = np.asarray(predicted.get_stress(voigt=True), dtype=np.float64)
        if (
            not np.isfinite(predicted_energy_eV)
            or not np.isfinite(predicted_forces).all()
            or not np.isfinite(predicted_stress).all()
        ):
            raise FloatingPointError(
                f"MLIP prediction is non-finite for DFT configuration "
                f"{atoms.info.get('configuration_id')!r}, state={state!r}."
            )
        energy_errors_eV_per_atom.append(
            float((predicted_energy_eV - reference_energy_eV) / len(predicted))
        )
        force_errors_eV_per_A.append(predicted_forces - reference_forces)
        stress_errors_GPa.append((predicted_stress - reference_stress) / units.GPa)
        frame_states.append(state)
    energy_errors = np.asarray(energy_errors_eV_per_atom, dtype=np.float64)
    energy_offset = float(np.mean(energy_errors))
    centered_energy_errors = energy_errors - energy_offset

    def summarize(indices: list[int]) -> dict[str, float | int]:
        selected_energy = centered_energy_errors[indices]
        selected_forces = np.concatenate(
            [force_errors_eV_per_A[index] for index in indices], axis=0
        )
        selected_stress = np.stack([stress_errors_GPa[index] for index in indices])
        values = np.concatenate(
            [selected_energy.ravel(), selected_forces.ravel(), selected_stress.ravel()]
        )
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite residual reached DFT error aggregation.")
        return {
            "frame_count": len(indices),
            "energy_rmse_meV_per_atom_after_global_constant_offset": float(
                1000.0 * np.sqrt(np.mean(np.square(selected_energy)))
            ),
            "force_rmse_eV_per_A": float(
                np.sqrt(np.mean(np.square(selected_forces)))
            ),
            "force_mae_eV_per_A": float(np.mean(np.abs(selected_forces))),
            "stress_rmse_GPa": float(
                np.sqrt(np.mean(np.square(selected_stress)))
            ),
            "stress_mae_GPa": float(np.mean(np.abs(selected_stress))),
        }

    overall = summarize(list(range(len(frames))))
    by_state = {
        state: summarize(
            [index for index, frame_state in enumerate(frame_states) if frame_state == state]
        )
        for state in sorted(set(frame_states))
    }
    return {
        "frame_count": len(frames),
        "states": sorted(set(frame_states)),
        "energy_offset_eV_per_atom": energy_offset,
        **overall,
        "by_state": by_state,
    }


def _melting_evidence(
    paths: tuple[Path, ...] | None,
    *,
    model_sha256: str,
    head: str | None,
    implementation_class: str,
    calculator_settings: dict[str, object],
    chemical_symbol: str,
    scope: QualificationScopeConfig,
    reference: MeltingReferenceConfig,
) -> tuple[dict[str, Any] | None, list[str]]:
    if paths is None:
        return None, ["no direct-coexistence temperature scan summaries were supplied"]
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise ImportError(
            "Melting-evidence verification requires scipy in the pointnet environment."
        ) from exc
    from .transition_generator import _resolve_zero_velocity

    def verify_file_digest(
        value: Any,
        expected_digest: Any,
        *,
        context: str,
        summary_path: Path,
    ) -> Path:
        file_path = Path(str(value)).expanduser()
        if not file_path.is_absolute():
            file_path = REPOSITORY_ROOT / file_path
        file_path = file_path.resolve()
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{summary_path}: {context} does not exist: {file_path}."
            )
        observed_digest = _sha256_file(file_path)
        if expected_digest != observed_digest:
            raise RuntimeError(
                f"{summary_path}: {context} SHA-256 mismatch for {file_path}: report="
                f"{expected_digest!r}, observed={observed_digest!r}."
            )
        return file_path

    def assert_close(
        observed: Any,
        expected: float,
        *,
        context: str,
        summary_path: Path,
    ) -> None:
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not np.isclose(
            float(observed), expected, rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError(
                f"{summary_path}: self-reported {context}={observed!r} does not equal "
                f"the value {expected:.16g} recomputed from per-replica velocities."
            )

    def verify_run_artifacts(
        replica_run: dict[str, Any],
        *,
        branch: Any,
        branch_index: int,
        replica_index: int,
        transition_config: Any,
        prepared_interface: Any,
        summary_path: Path,
    ) -> dict[str, Any]:
        from .simulation import ThermodynamicTrace
        from .transition_analysis import (
            _linear_fit,
            _thermodynamic_stationarity,
            analyze_transition,
        )

        run_name = f"{branch.name}/replica_{replica_index:03d}"
        if replica_run.get("run_name") != run_name:
            raise RuntimeError(
                f"{summary_path}: replica {replica_index} run_name="
                f"{replica_run.get('run_name')!r}, expected {run_name!r} from the parsed "
                "transition config."
            )
        artifacts = replica_run.get("artifacts")
        required_artifacts = {
            "trajectory.npz",
            "equilibration_trajectory.npz",
            "transition_progress.npz",
            "metadata.json",
        }
        if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
            raise ValueError(
                f"{summary_path}: {run_name} must bind exactly the per-run artifacts "
                f"{sorted(required_artifacts)}, got "
                f"{sorted(artifacts) if isinstance(artifacts, dict) else artifacts!r}."
            )
        artifact_paths: dict[str, Path] = {}
        for filename in sorted(required_artifacts):
            record = artifacts[filename]
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise TypeError(
                    f"{summary_path}: {run_name} artifact {filename} must contain exactly "
                    "path and sha256."
                )
            artifact_path = verify_file_digest(
                record["path"],
                record["sha256"],
                context=f"{run_name}/{filename}",
                summary_path=summary_path,
            )
            expected_path = (
                transition_config.output.root_dir / run_name / filename
            ).resolve()
            if artifact_path != expected_path:
                raise RuntimeError(
                    f"{summary_path}: artifact {filename} for {run_name} resolves to "
                    f"{artifact_path}, expected {expected_path} from the parsed config."
                )
            artifact_paths[filename] = artifact_path

        with artifact_paths["metadata.json"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        configured_seed = transition_config.random_seeds[replica_index]
        expected_simulation_seed = int(
            np.random.SeedSequence([configured_seed, branch_index]).generate_state(1)[0]
        )
        if metadata.get("schema_version") != 2:
            raise ValueError(
                f"{artifact_paths['metadata.json']}: expected schema_version=2, got "
                f"{metadata.get('schema_version')!r}."
            )
        if metadata.get("branch") != asdict(branch):
            raise RuntimeError(
                f"{artifact_paths['metadata.json']}: branch metadata differs from the "
                "parsed transition config."
            )
        expected_replica = {
            "index": replica_index,
            "configured_replica_seed": configured_seed,
            "simulation_seed": expected_simulation_seed,
        }
        if metadata.get("replica") != expected_replica:
            raise RuntimeError(
                f"{artifact_paths['metadata.json']}: replica provenance="
                f"{metadata.get('replica')!r}, expected {expected_replica!r}."
            )
        expected_source = {
            "dataset": str(transition_config.source_dataset),
            "environment": transition_config.source_interface_environment,
            "frame_step": transition_config.source_frame_step,
            "slab_bounds_fractional": list(
                prepared_interface.slab_bounds_fractional
            ),
        }
        if metadata.get("source") != expected_source:
            raise RuntimeError(
                f"{artifact_paths['metadata.json']}: source provenance differs from the "
                "hash-bound source selected by the transition config."
            )
        physics = metadata.get("physics")
        if not isinstance(physics, dict):
            raise TypeError(f"{artifact_paths['metadata.json']}: physics must be a mapping.")
        metadata_calculator = physics.get("calculator")
        if not isinstance(metadata_calculator, dict):
            raise TypeError(
                f"{artifact_paths['metadata.json']}: physics.calculator must be a mapping."
            )
        if (
            metadata_calculator.get("implementation_class") != implementation_class
            or metadata_calculator.get("model_sha256") != model_sha256
            or metadata_calculator.get("head") != head
            or metadata_calculator.get("settings") != calculator_settings
            or physics.get("pressure_GPa")
            != transition_config.generator.dynamics.pressure_GPa
            or physics.get("timestep_fs")
            != transition_config.generator.dynamics.timestep_fs
            or physics.get("ptm_normalized_rmsd_cutoff")
            != transition_config.analysis.ptm_rmsd_cutoff
        ):
            raise RuntimeError(
                f"{artifact_paths['metadata.json']}: calculator, ensemble, or PTM settings "
                "do not match the benchmarked parsed protocol."
            )

        def load_thermodynamic_arrays(
            artifact_path: Path,
            *,
            final_step: int,
        ) -> dict[str, np.ndarray]:
            required = {
                "step",
                "positions_A",
                "cell_vectors_A",
                "temperature_K",
                "pressure_GPa",
                "volume_A3",
                "potential_energy_eV_per_atom",
            }
            with np.load(artifact_path, allow_pickle=False) as archive:
                missing = sorted(required - set(archive.files))
                if missing:
                    raise ValueError(
                        f"{artifact_path}: missing thermodynamic arrays {missing}."
                    )
                arrays = {
                    name: np.asarray(archive[name]).copy()
                    for name in required
                }
            step = np.asarray(arrays["step"])
            expected_steps = np.arange(
                0,
                final_step + transition_config.sample_interval,
                transition_config.sample_interval,
                dtype=step.dtype,
            )
            if not np.array_equal(step, expected_steps):
                raise RuntimeError(
                    f"{artifact_path}: stored steps={step.tolist()} do not equal the parsed "
                    f"protocol grid={expected_steps.tolist()}."
                )
            frame_count = len(step)
            positions_A = np.asarray(arrays["positions_A"])
            if positions_A.shape != (
                frame_count,
                len(prepared_interface.atoms),
                3,
            ) or not np.isfinite(positions_A).all():
                raise ValueError(
                    f"{artifact_path}: positions_A must be finite with shape "
                    f"({frame_count}, {len(prepared_interface.atoms)}, 3), got "
                    f"{positions_A.shape}."
                )
            for name in (
                "temperature_K",
                "pressure_GPa",
                "volume_A3",
                "potential_energy_eV_per_atom",
            ):
                if arrays[name].shape != (frame_count,) or not np.isfinite(
                    arrays[name]
                ).all():
                    raise FloatingPointError(
                        f"{artifact_path}: {name} must be finite with shape "
                        f"({frame_count},), got {arrays[name].shape}."
                    )
            cell = np.asarray(arrays["cell_vectors_A"], dtype=np.float64)
            if cell.shape != (frame_count, 3, 3) or not np.isfinite(cell).all():
                raise FloatingPointError(
                    f"{artifact_path}: cell_vectors_A must be finite with shape "
                    f"({frame_count}, 3, 3), got {cell.shape}."
                )
            determinants = np.linalg.det(cell)
            if np.any(determinants <= 0.0) or not np.allclose(
                determinants,
                arrays["volume_A3"],
                rtol=1.0e-10,
                atol=1.0e-8,
            ):
                raise RuntimeError(
                    f"{artifact_path}: volume_A3 is inconsistent with positive cell "
                    "determinants."
                )
            return arrays

        production = load_thermodynamic_arrays(
            artifact_paths["trajectory.npz"], final_step=branch.production_steps
        )
        equilibration = load_thermodynamic_arrays(
            artifact_paths["equilibration_trajectory.npz"],
            final_step=branch.equilibration_steps,
        )
        boundary_mismatches: list[str] = []
        for array_name in (
            "positions_A",
            "cell_vectors_A",
            "temperature_K",
            "pressure_GPa",
            "volume_A3",
            "potential_energy_eV_per_atom",
        ):
            if not np.array_equal(
                equilibration[array_name][-1], production[array_name][0]
            ):
                boundary_mismatches.append(array_name)
        if boundary_mismatches:
            raise RuntimeError(
                f"{summary_path}: {run_name} equilibration and production artifacts are not "
                "one continuous integrator trace at their shared boundary; differing arrays="
                f"{boundary_mismatches}."
            )
        frame_count = len(production["step"])
        with np.load(
            artifact_paths["transition_progress.npz"], allow_pickle=False
        ) as progress_archive:
            required_progress = {
                "step",
                "time_ps",
                "crystalline_fraction",
                "profile_contrast",
                "signed_interface_advance_A",
                "mean_interface_advance_A",
                "fitted_interface_velocity_m_per_s",
                "individual_interface_velocities_m_per_s",
                "individual_interface_fit_r_squared",
                "velocity_fit_r_squared",
                "velocity_fit_ols_standard_error_m_per_s",
                "velocity_fit_residual_rms_A",
            }
            missing_progress = sorted(required_progress - set(progress_archive.files))
            if missing_progress:
                raise ValueError(
                    f"{artifact_paths['transition_progress.npz']}: missing arrays "
                    f"{missing_progress}."
                )
            progress_arrays = {
                name: np.asarray(progress_archive[name]).copy()
                for name in required_progress
            }
        if not np.array_equal(progress_arrays["step"], production["step"]):
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: step array differs from the "
                "hash-bound production trajectory."
            )
        expected_time_ps = (
            production["step"].astype(np.float64)
            * transition_config.generator.dynamics.timestep_fs
            / 1000.0
        )
        if not np.array_equal(progress_arrays["time_ps"], expected_time_ps):
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: time_ps is inconsistent with "
                "the parsed timestep and production steps."
            )
        crystalline_fraction = np.asarray(
            progress_arrays["crystalline_fraction"], dtype=np.float64
        )
        profile_contrast = np.asarray(
            progress_arrays["profile_contrast"], dtype=np.float64
        )
        signed_advance = np.asarray(
            progress_arrays["signed_interface_advance_A"], dtype=np.float64
        )
        mean_advance = np.asarray(
            progress_arrays["mean_interface_advance_A"], dtype=np.float64
        )
        if (
            crystalline_fraction.shape != (frame_count,)
            or profile_contrast.shape != (frame_count,)
            or signed_advance.shape != (frame_count, 2)
            or mean_advance.shape != (frame_count,)
            or not np.isfinite(
                np.concatenate(
                    (
                        crystalline_fraction,
                        profile_contrast,
                        signed_advance.reshape(-1),
                        mean_advance,
                    )
                )
            ).all()
        ):
            raise FloatingPointError(
                f"{artifact_paths['transition_progress.npz']}: front/profile arrays have "
                "invalid shapes or non-finite values."
            )
        if not np.allclose(
            mean_advance, np.mean(signed_advance, axis=1), rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: mean_interface_advance_A "
                "does not equal the mean of the two spatial fronts."
            )
        steady_mask = (
            (production["step"] >= branch.steady_state_start_step)
            & (production["step"] <= branch.steady_state_end_step)
        )
        if np.count_nonzero(steady_mask) < 3:
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: fewer than three stored "
                "frames occur in the parsed steady fit interval."
            )
        minimum_contrast = float(np.min(profile_contrast[steady_mask]))
        if minimum_contrast < transition_config.analysis.minimum_profile_contrast:
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: minimum fitted profile "
                f"contrast={minimum_contrast:.6g} is below parsed gate="
                f"{transition_config.analysis.minimum_profile_contrast:.6g}."
            )
        mean_fit = _linear_fit(expected_time_ps[steady_mask], mean_advance[steady_mask])
        front_fits = tuple(
            _linear_fit(
                expected_time_ps[steady_mask], signed_advance[steady_mask, front_index]
            )
            for front_index in range(2)
        )
        velocity = mean_fit[0] * 100.0
        individual_velocities = np.asarray(
            [fit[0] * 100.0 for fit in front_fits], dtype=np.float64
        )
        individual_r_squared = np.asarray(
            [fit[1] for fit in front_fits], dtype=np.float64
        )
        if (
            mean_fit[1] < transition_config.analysis.minimum_velocity_fit_r_squared
            or np.any(
                individual_r_squared
                < transition_config.analysis.minimum_velocity_fit_r_squared
            )
            or (
                transition_config.analysis.minimum_velocity_fit_r_squared > 0.0
                and individual_velocities[0] * individual_velocities[1] <= 0.0
            )
        ):
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: recomputed spatial-front "
                "fits violate the parsed quality gates."
            )
        crystalline_change = float(
            crystalline_fraction[-1] - crystalline_fraction[0]
        )
        if branch.expected_direction == "growth":
            signed_change = crystalline_change
            signed_velocity = velocity
        elif branch.expected_direction == "melting":
            signed_change = -crystalline_change
            signed_velocity = -velocity
        else:
            signed_change = np.inf
            signed_velocity = np.inf
        if (
            signed_change < branch.minimum_crystalline_fraction_change
            or (
                branch.expected_direction != "unconstrained"
                and branch.minimum_crystalline_fraction_change > 0.0
                and signed_velocity <= 0.0
            )
        ):
            raise RuntimeError(
                f"{artifact_paths['transition_progress.npz']}: recomputed front direction "
                f"does not satisfy branch={branch.name!r} expectations."
            )

        production_trace = ThermodynamicTrace(
            step=production["step"],
            temperature_K=production["temperature_K"],
            pressure_GPa=production["pressure_GPa"],
            volume_A3=production["volume_A3"],
            potential_energy_eV_per_atom=production[
                "potential_energy_eV_per_atom"
            ],
            positions_A=production["positions_A"],
            cell_vectors_A=production["cell_vectors_A"],
        )
        equilibration_trace = ThermodynamicTrace(
            step=equilibration["step"],
            temperature_K=equilibration["temperature_K"],
            pressure_GPa=equilibration["pressure_GPa"],
            volume_A3=equilibration["volume_A3"],
            potential_energy_eV_per_atom=equilibration[
                "potential_energy_eV_per_atom"
            ],
            positions_A=equilibration["positions_A"],
            cell_vectors_A=equilibration["cell_vectors_A"],
        )
        stationarity = _thermodynamic_stationarity(
            equilibration_trace,
            production_trace,
            steady_mask=steady_mask,
            target_temperature_K=branch.temperature_K,
            target_pressure_GPa=transition_config.generator.dynamics.pressure_GPa,
            maximum_temperature_error_K=(
                transition_config.generator.validation.maximum_temperature_error_K
            ),
            maximum_pressure_error_GPa=(
                transition_config.generator.validation.maximum_pressure_error_GPa
            ),
            branch_name=run_name,
        )
        reanalyzed = analyze_transition(
            production_trace,
            equilibration_trace=equilibration_trace,
            chemical_symbol=transition_config.generator.system.chemical_symbol,
            timestep_fs=transition_config.generator.dynamics.timestep_fs,
            slab_bounds_fractional=prepared_interface.slab_bounds_fractional,
            profile_bins=transition_config.analysis.profile_bins,
            profile_smoothing_bins=(
                transition_config.analysis.profile_smoothing_bins
            ),
            ptm_rmsd_cutoff=transition_config.analysis.ptm_rmsd_cutoff,
            minimum_profile_contrast=(
                transition_config.analysis.minimum_profile_contrast
            ),
            minimum_velocity_fit_r_squared=(
                transition_config.analysis.minimum_velocity_fit_r_squared
            ),
            target_pressure_GPa=transition_config.generator.dynamics.pressure_GPa,
            maximum_temperature_error_K=(
                transition_config.generator.validation.maximum_temperature_error_K
            ),
            maximum_pressure_error_GPa=(
                transition_config.generator.validation.maximum_pressure_error_GPa
            ),
            branch=branch,
            progress=lambda _message: None,
        )
        for artifact_array_name, reanalyzed_value in (
            ("crystalline_fraction", reanalyzed.crystalline_fraction),
            ("profile_contrast", reanalyzed.profile_contrast),
            ("signed_interface_advance_A", reanalyzed.signed_interface_advance_A),
            ("mean_interface_advance_A", reanalyzed.mean_interface_advance_A),
        ):
            observed_value = np.asarray(progress_arrays[artifact_array_name])
            if observed_value.shape != reanalyzed_value.shape or not np.allclose(
                observed_value,
                reanalyzed_value,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"{artifact_paths['transition_progress.npz']}: "
                    f"{artifact_array_name} differs from PTM reanalysis of the hashed "
                    "coordinate trajectory."
                )
        if asdict(stationarity) != asdict(reanalyzed.stationarity):
            raise RuntimeError(
                f"{summary_path}: {run_name} standalone stationarity recomputation differs "
                "from full PTM transition reanalysis."
            )
        metadata_transition = metadata.get("transition")
        if not isinstance(metadata_transition, dict):
            raise TypeError(
                f"{artifact_paths['metadata.json']}: transition must be a mapping."
            )
        if metadata_transition.get("thermodynamic_stationarity") != asdict(
            reanalyzed.stationarity
        ):
            raise RuntimeError(
                f"{artifact_paths['metadata.json']}: thermodynamic stationarity summary "
                "does not equal recomputation from hashed trajectories."
            )
        recomputed = {
            "velocity_m_per_s": reanalyzed.fitted_interface_velocity_m_per_s,
            "fit_r_squared": reanalyzed.velocity_fit_r_squared,
            "fit_ols_standard_error_m_per_s": (
                reanalyzed.velocity_fit_ols_standard_error_m_per_s
            ),
            "fit_residual_rms_A": reanalyzed.velocity_fit_residual_rms_A,
            "minimum_fit_profile_contrast": float(
                np.min(reanalyzed.profile_contrast[steady_mask])
            ),
            "individual_front_velocities_m_per_s": (
                reanalyzed.individual_interface_velocities_m_per_s
            ),
            "individual_front_fit_r_squared": (
                reanalyzed.individual_interface_fit_r_squared
            ),
            "production_initial_cell_vectors_A": np.asarray(
                production["cell_vectors_A"][0], dtype=np.float64
            ),
        }
        scalar_progress_keys = {
            "fitted_interface_velocity_m_per_s": "velocity_m_per_s",
            "velocity_fit_r_squared": "fit_r_squared",
            "velocity_fit_ols_standard_error_m_per_s": (
                "fit_ols_standard_error_m_per_s"
            ),
            "velocity_fit_residual_rms_A": "fit_residual_rms_A",
        }
        for stored_key, recomputed_key in scalar_progress_keys.items():
            assert_close(
                np.asarray(progress_arrays[stored_key]).item(),
                float(recomputed[recomputed_key]),
                context=f"{run_name} transition_progress.{stored_key}",
                summary_path=summary_path,
            )
        for stored_key, recomputed_key in (
            (
                "individual_interface_velocities_m_per_s",
                "individual_front_velocities_m_per_s",
            ),
            (
                "individual_interface_fit_r_squared",
                "individual_front_fit_r_squared",
            ),
        ):
            observed = np.asarray(progress_arrays[stored_key], dtype=np.float64)
            expected = np.asarray(recomputed[recomputed_key], dtype=np.float64)
            if observed.shape != (2,) or not np.allclose(
                observed, expected, rtol=1.0e-12, atol=1.0e-12
            ):
                raise RuntimeError(
                    f"{artifact_paths['transition_progress.npz']}: {stored_key}="
                    f"{observed.tolist()} differs from raw-array recomputation="
                    f"{expected.tolist()}."
                )
        return recomputed

    failures: list[str] = []
    scans: list[dict[str, Any]] = []
    protocol_signatures: set[tuple[Any, ...]] = set()
    interpolated_temperatures: list[float] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("schema_version") != 2:
            raise ValueError(
                f"{path}: expected transition velocity summary schema_version=2 with "
                "hash-bound per-run artifacts, got "
                f"{result.get('schema_version')!r}."
            )
        calculator = result.get("calculator")
        protocol = result.get("protocol")
        if not isinstance(calculator, dict) or not isinstance(protocol, dict):
            raise TypeError(f"{path}: calculator and protocol must be mappings.")
        if calculator.get("model_sha256") != model_sha256 or calculator.get("head") != head:
            raise RuntimeError(
                f"{path}: melting scan identifies model_sha256="
                f"{calculator.get('model_sha256')!r}, head={calculator.get('head')!r}, but "
                f"the benchmarked potential uses model_sha256={model_sha256!r}, head={head!r}."
            )
        if (
            calculator.get("implementation_class") != implementation_class
            or calculator.get("settings") != calculator_settings
        ):
            raise RuntimeError(
                f"{path}: melting scan calculator implementation/settings do not match the "
                f"qualified production calculator; observed implementation="
                f"{calculator.get('implementation_class')!r}, settings="
                f"{calculator.get('settings')!r}, expected implementation="
                f"{implementation_class!r}, settings={calculator_settings}."
            )
        if protocol.get("chemical_symbol") != chemical_symbol:
            raise RuntimeError(
                f"{path}: melting protocol chemical_symbol="
                f"{protocol.get('chemical_symbol')!r}, expected {chemical_symbol!r}."
            )
        pressure_GPa = protocol.get("pressure_GPa")
        timestep_fs = protocol.get("timestep_fs")
        if not isinstance(pressure_GPa, (int, float)) or not (
            scope.pressure_range_GPa[0]
            <= float(pressure_GPa)
            <= scope.pressure_range_GPa[1]
        ):
            raise RuntimeError(
                f"{path}: melting protocol pressure_GPa={pressure_GPa!r} lies outside "
                f"qualification scope {scope.pressure_range_GPa}."
            )
        if (
            not isinstance(timestep_fs, (int, float))
            or not np.isfinite(timestep_fs)
            or float(timestep_fs) <= 0.0
            or float(timestep_fs) > scope.maximum_timestep_fs
        ):
            raise RuntimeError(
                f"{path}: melting protocol timestep_fs={timestep_fs!r} exceeds scoped "
                f"maximum {scope.maximum_timestep_fs}."
            )
        transition_config_path = verify_file_digest(
            protocol.get("config_file"),
            protocol.get("config_file_sha256"),
            context="transition protocol config_file",
            summary_path=path,
        )
        from .transition_config import load_transition_config
        from .transition_generator import _load_prepared_interface

        transition_config = load_transition_config(transition_config_path)
        canonical_config_sha256 = hashlib.sha256(
            json.dumps(
                transition_config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if protocol.get("config_sha256") != canonical_config_sha256:
            raise RuntimeError(
                f"{path}: protocol.config_sha256={protocol.get('config_sha256')!r} does "
                "not match the canonical parsed transition config "
                f"{canonical_config_sha256!r}."
            )
        transition_potential = transition_config.generator.potential
        if (
            transition_potential.sha256 != model_sha256
            or transition_potential.head != head
            or potential_calculator_settings(transition_potential)
            != calculator_settings
        ):
            raise RuntimeError(
                f"{path}: parsed transition config uses a different model or calculator "
                "setting than the benchmarked production potential."
            )
        prepared_interface = _load_prepared_interface(transition_config)
        serialized_analysis = asdict(transition_config.analysis)
        expected_protocol = {
            "config_sha256": canonical_config_sha256,
            "config_file": str(transition_config.config_path),
            "config_file_sha256": _sha256_file(transition_config.config_path),
            "chemical_symbol": transition_config.generator.system.chemical_symbol,
            "pressure_GPa": transition_config.generator.dynamics.pressure_GPa,
            "timestep_fs": transition_config.generator.dynamics.timestep_fs,
            "atom_count": len(prepared_interface.atoms),
            "conventional_cell_repetitions": list(
                transition_config.generator.system.repetitions
            ),
            "initial_lattice_constant_A": (
                transition_config.generator.system.initial_lattice_constant_A
            ),
            "prepared_source_cell_vectors_A": np.asarray(
                prepared_interface.atoms.cell, dtype=np.float64
            ).tolist(),
            "interface_normal_fractional_cell_axis": [0, 0, 1],
            "interface_normal_crystal_direction": "[001]",
            "source": {
                "generator_config": str(transition_config.generator.config_path),
                "generator_config_sha256": _sha256_file(
                    transition_config.generator.config_path
                ),
                "dataset": str(transition_config.source_dataset),
                "dataset_manifest_sha256": _sha256_file(
                    transition_config.source_dataset / "manifest.json"
                ),
                "environment": transition_config.source_interface_environment,
                "frame_step": transition_config.source_frame_step,
            },
            "analysis": serialized_analysis,
            "sample_interval_steps": transition_config.sample_interval,
            "replica_seeds": list(transition_config.random_seeds),
            "temperature_runs": [
                {
                    "name": branch.name,
                    "temperature_K": branch.temperature_K,
                    "equilibration_steps": branch.equilibration_steps,
                    "production_steps": branch.production_steps,
                    "steady_state_fit_step_interval": [
                        branch.steady_state_start_step,
                        branch.steady_state_end_step,
                    ],
                }
                for branch in transition_config.temperature_runs
            ],
        }
        if protocol != expected_protocol:
            differing_keys = sorted(
                key
                for key in set(protocol) | set(expected_protocol)
                if protocol.get(key) != expected_protocol.get(key)
            )
            raise RuntimeError(
                f"{path}: transition summary protocol differs from the parsed, hash-bound "
                f"configuration in keys={differing_keys}."
            )
        source = protocol.get("source")
        if not isinstance(source, dict):
            raise TypeError(f"{path}: protocol.source must be a mapping.")
        verify_file_digest(
            source.get("generator_config"),
            source.get("generator_config_sha256"),
            context="source generator_config",
            summary_path=path,
        )
        source_dataset = Path(str(source.get("dataset"))).expanduser()
        if not source_dataset.is_absolute():
            source_dataset = REPOSITORY_ROOT / source_dataset
        verify_file_digest(
            source_dataset.resolve() / "manifest.json",
            source.get("dataset_manifest_sha256"),
            context="source dataset manifest",
            summary_path=path,
        )
        atom_count = protocol.get("atom_count")
        repetitions = protocol.get("conventional_cell_repetitions")
        normal = protocol.get("interface_normal_crystal_direction")
        cell = np.asarray(protocol.get("prepared_source_cell_vectors_A"), dtype=np.float64)
        if (
            not isinstance(atom_count, int)
            or atom_count <= 0
            or not isinstance(repetitions, list)
            or len(repetitions) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in repetitions
            )
            or not isinstance(normal, str)
            or cell.shape != (3, 3)
            or not np.isfinite(cell).all()
            or float(np.linalg.det(cell)) <= 0.0
        ):
            raise ValueError(f"{path}: melting protocol system identity is incomplete/invalid.")
        expected_atom_count = 4 * int(np.prod(repetitions))
        if atom_count != expected_atom_count:
            raise ValueError(
                f"{path}: atom_count={atom_count} is inconsistent with the declared "
                f"conventional FCC repetitions={repetitions}; expected {expected_atom_count}."
            )
        signature = (
            tuple(int(value) for value in repetitions),
            normal,
        )
        protocol_signatures.add(signature)
        analysis = protocol.get("analysis")
        if not isinstance(analysis, dict):
            raise TypeError(f"{path}: protocol.analysis must be a mapping.")
        minimum_fit_r_squared = float(analysis.get("minimum_velocity_fit_r_squared"))
        minimum_profile_contrast = float(analysis.get("minimum_profile_contrast"))
        ptm_rmsd_cutoff = float(analysis.get("ptm_rmsd_cutoff"))
        if (
            not np.isfinite(
                [minimum_fit_r_squared, minimum_profile_contrast, ptm_rmsd_cutoff]
            ).all()
            or not 0.0 <= minimum_fit_r_squared <= 1.0
            or minimum_profile_contrast <= 0.0
            or not 0.0 < ptm_rmsd_cutoff <= 1.0
        ):
            raise ValueError(f"{path}: invalid transition protocol analysis={analysis!r}.")
        replica_seeds = protocol.get("replica_seeds")
        if (
            not isinstance(replica_seeds, list)
            or len(replica_seeds) < 2
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in replica_seeds
            )
            or len(set(replica_seeds)) != len(replica_seeds)
        ):
            raise ValueError(f"{path}: protocol.replica_seeds is invalid: {replica_seeds!r}.")
        temperature_runs = protocol.get("temperature_runs")
        temperature_summaries = result.get("temperatures")
        if not isinstance(temperature_runs, list) or not isinstance(
            temperature_summaries, list
        ):
            raise TypeError(f"{path}: protocol temperature_runs and temperatures must be lists.")
        if len(temperature_runs) != len(temperature_summaries):
            raise ValueError(
                f"{path}: protocol has {len(temperature_runs)} temperature runs but summary "
                f"has {len(temperature_summaries)}."
            )
        protocol_temperatures = np.asarray(
            [run.get("temperature_K") for run in temperature_runs], dtype=np.float64
        )
        if (
            not np.isfinite(protocol_temperatures).all()
            or np.any(np.diff(protocol_temperatures) <= 0.0)
        ):
            raise ValueError(
                f"{path}: protocol temperature grid must be finite and strictly increasing, "
                f"got {protocol_temperatures.tolist()}."
            )
        for branch_index, (run, summary, configured_branch) in enumerate(
            zip(
                temperature_runs,
                temperature_summaries,
                transition_config.temperature_runs,
            )
        ):
            if not isinstance(run, dict) or not isinstance(summary, dict):
                raise TypeError(f"{path}: every temperature run/summary must be a mapping.")
            if run.get("name") != summary.get("name") or run.get(
                "temperature_K"
            ) != summary.get("temperature_K"):
                raise RuntimeError(
                    f"{path}: protocol run name/temperature {run.get('name')!r}/"
                    f"{run.get('temperature_K')!r} does not match summary "
                    f"{summary.get('name')!r}/{summary.get('temperature_K')!r}."
                )
            temperature_K = float(run["temperature_K"])
            out_of_scope_states = [
                state
                for state in ("solid_bulk", "liquid_bulk", "interface")
                if not scope.state_temperature_ranges_K[state][0]
                <= temperature_K
                <= scope.state_temperature_ranges_K[state][1]
            ]
            if out_of_scope_states:
                raise RuntimeError(
                    f"{path}: coexistence temperature {temperature_K} K is outside "
                    f"qualification scope for states={out_of_scope_states}."
                )
            replica_count = summary.get("replica_count")
            replica_runs = summary.get("runs")
            if not isinstance(replica_runs, list):
                raise TypeError(
                    f"{path}: temperature {temperature_K} K summary.runs must be a list."
                )
            if replica_count != len(replica_runs) or replica_count != len(replica_seeds):
                raise RuntimeError(
                    f"{path}: temperature {temperature_K} K replica_count={replica_count!r}, "
                    f"len(runs)={len(replica_runs)}, and len(protocol.replica_seeds)="
                    f"{len(replica_seeds)} must agree."
                )
            production_steps = run.get("production_steps")
            if not isinstance(replica_count, int) or (
                replica_count < reference.minimum_replicas_per_temperature
            ):
                failures.append(
                    f"{path}: temperature {summary.get('temperature_K')} K has "
                    f"{replica_count!r} replicas, below "
                    f"{reference.minimum_replicas_per_temperature}"
                )
            if not isinstance(production_steps, int) or production_steps <= 0:
                raise ValueError(f"{path}: invalid production_steps={production_steps!r}.")
            duration_ps = production_steps * float(timestep_fs) / 1000.0
            if duration_ps < reference.minimum_production_duration_ps:
                failures.append(
                    f"{path}: production duration {duration_ps:.6g} ps at "
                    f"{summary.get('temperature_K')} K is below "
                    f"{reference.minimum_production_duration_ps:.6g} ps"
                )
            fit_interval = run.get("steady_state_fit_step_interval")
            if (
                not isinstance(fit_interval, list)
                or len(fit_interval) != 2
                or any(not isinstance(value, int) for value in fit_interval)
            ):
                raise ValueError(f"{path}: invalid steady-state fit interval {fit_interval!r}.")
            indexed_runs: dict[int, dict[str, Any]] = {}
            simulation_seeds: set[int] = set()
            for replica_run in replica_runs:
                if not isinstance(replica_run, dict):
                    raise TypeError(f"{path}: every per-replica run must be a mapping.")
                replica_index = replica_run.get("replica_index")
                if (
                    not isinstance(replica_index, int)
                    or isinstance(replica_index, bool)
                    or replica_index in indexed_runs
                    or not 0 <= replica_index < len(replica_seeds)
                ):
                    raise ValueError(
                        f"{path}: invalid/duplicate replica_index={replica_index!r} at "
                        f"temperature {temperature_K} K."
                    )
                if replica_run.get("configured_replica_seed") != replica_seeds[replica_index]:
                    raise RuntimeError(
                        f"{path}: replica {replica_index} configured seed does not match "
                        "protocol.replica_seeds."
                    )
                simulation_seed = replica_run.get("simulation_seed")
                if (
                    not isinstance(simulation_seed, int)
                    or isinstance(simulation_seed, bool)
                    or simulation_seed in simulation_seeds
                ):
                    raise ValueError(
                        f"{path}: invalid/duplicate simulation_seed={simulation_seed!r} at "
                        f"temperature {temperature_K} K."
                    )
                simulation_seeds.add(simulation_seed)
                recomputed_run = verify_run_artifacts(
                    replica_run,
                    branch=configured_branch,
                    branch_index=branch_index,
                    replica_index=replica_index,
                    transition_config=transition_config,
                    prepared_interface=prepared_interface,
                    summary_path=path,
                )
                for reported_key in (
                    "velocity_m_per_s",
                    "fit_r_squared",
                    "fit_ols_standard_error_m_per_s",
                    "fit_residual_rms_A",
                    "minimum_fit_profile_contrast",
                ):
                    assert_close(
                        replica_run.get(reported_key),
                        float(recomputed_run[reported_key]),
                        context=f"replica {replica_index} {reported_key}",
                        summary_path=path,
                    )
                for reported_key in (
                    "individual_front_velocities_m_per_s",
                    "individual_front_fit_r_squared",
                ):
                    observed = np.asarray(
                        replica_run.get(reported_key), dtype=np.float64
                    )
                    expected = np.asarray(
                        recomputed_run[reported_key], dtype=np.float64
                    )
                    if observed.shape != (2,) or not np.allclose(
                        observed, expected, rtol=1.0e-12, atol=1.0e-12
                    ):
                        raise RuntimeError(
                            f"{path}: replica {replica_index} self-reported "
                            f"{reported_key}={observed.tolist()} differs from hashed-artifact "
                            f"recomputation={expected.tolist()}."
                        )
                numeric_run = np.asarray(
                    [
                        recomputed_run["velocity_m_per_s"],
                        recomputed_run["fit_r_squared"],
                        recomputed_run["fit_ols_standard_error_m_per_s"],
                        recomputed_run["fit_residual_rms_A"],
                        recomputed_run["minimum_fit_profile_contrast"],
                        *recomputed_run["individual_front_velocities_m_per_s"],
                        *recomputed_run["individual_front_fit_r_squared"],
                    ],
                    dtype=np.float64,
                )
                if numeric_run.shape != (9,) or not np.isfinite(numeric_run).all():
                    raise FloatingPointError(
                        f"{path}: replica {replica_index} at {temperature_K} K has "
                        "non-finite/incomplete fit diagnostics."
                    )
                if (
                    numeric_run[1] < minimum_fit_r_squared
                    or np.any(numeric_run[7:9] < minimum_fit_r_squared)
                    or numeric_run[4] < minimum_profile_contrast
                    or numeric_run[2] < 0.0
                    or numeric_run[3] < 0.0
                ):
                    raise RuntimeError(
                        f"{path}: replica {replica_index} at {temperature_K} K violates its "
                        f"reported analysis quality gates; diagnostics={numeric_run.tolist()}, "
                        f"analysis={analysis}."
                    )
                if replica_run.get("fit_step_interval") != fit_interval:
                    raise RuntimeError(
                        f"{path}: replica {replica_index} fit interval="
                        f"{replica_run.get('fit_step_interval')!r} differs from protocol "
                        f"interval={fit_interval}."
                    )
                replica_cell = np.asarray(
                    replica_run.get("production_initial_cell_vectors_A"),
                    dtype=np.float64,
                )
                if (
                    replica_cell.shape != (3, 3)
                    or not np.isfinite(replica_cell).all()
                    or float(np.linalg.det(replica_cell)) <= 0.0
                ):
                    raise ValueError(
                        f"{path}: invalid production initial cell for replica {replica_index}."
                    )
                if not np.allclose(
                    replica_cell,
                    recomputed_run["production_initial_cell_vectors_A"],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        f"{path}: replica {replica_index} self-reported production initial "
                        "cell differs from the hashed production trajectory."
                    )
                verified_run = dict(replica_run)
                verified_run["velocity_m_per_s"] = float(
                    recomputed_run["velocity_m_per_s"]
                )
                indexed_runs[replica_index] = verified_run
            velocities = np.asarray(
                [indexed_runs[index]["velocity_m_per_s"] for index in range(replica_count)],
                dtype=np.float64,
            )
            recomputed_mean = float(np.mean(velocities))
            recomputed_sd = float(np.std(velocities, ddof=1))
            recomputed_se = recomputed_sd / np.sqrt(replica_count)
            half_width = float(
                student_t.ppf(0.975, df=replica_count - 1) * recomputed_se
            )
            recomputed_interval = [
                recomputed_mean - half_width,
                recomputed_mean + half_width,
            ]
            assert_close(
                summary.get("mean_velocity_m_per_s"),
                recomputed_mean,
                context="mean_velocity_m_per_s",
                summary_path=path,
            )
            assert_close(
                summary.get("sample_standard_deviation_m_per_s"),
                recomputed_sd,
                context="sample_standard_deviation_m_per_s",
                summary_path=path,
            )
            assert_close(
                summary.get("standard_error_m_per_s"),
                recomputed_se,
                context="standard_error_m_per_s",
                summary_path=path,
            )
            observed_interval = summary.get("confidence_interval_95_m_per_s")
            if (
                not isinstance(observed_interval, list)
                or len(observed_interval) != 2
            ):
                raise ValueError(
                    f"{path}: invalid confidence_interval_95_m_per_s="
                    f"{observed_interval!r}."
                )
            for endpoint_index, expected_endpoint in enumerate(recomputed_interval):
                assert_close(
                    observed_interval[endpoint_index],
                    expected_endpoint,
                    context=f"confidence_interval_95_m_per_s[{endpoint_index}]",
                    summary_path=path,
                )
        bracket = result.get("zero_velocity_bracket_K")
        zero_velocity = result.get("zero_velocity")
        if not isinstance(zero_velocity, dict):
            raise TypeError(f"{path}: zero_velocity must be a mapping.")
        recomputed_zero_velocity = _resolve_zero_velocity(temperature_summaries)
        if zero_velocity != recomputed_zero_velocity or bracket != recomputed_zero_velocity.get(
            "bracket_temperature_K"
        ):
            raise RuntimeError(
                f"{path}: self-reported zero-velocity result does not match recomputation "
                f"from per-replica velocities; reported zero_velocity={zero_velocity!r}, "
                f"reported bracket={bracket!r}, recomputed={recomputed_zero_velocity!r}."
            )
        if bracket is None:
            reason = zero_velocity.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    f"{path}: unresolved melting scan must provide zero_velocity.reason."
                )
            failures.append(f"{path}: zero-velocity bracket unresolved: {reason}")
        else:
            interpolated = zero_velocity.get("interpolated_temperature_K")
            if (
                not isinstance(bracket, list)
                or len(bracket) != 2
                or not all(isinstance(value, (int, float)) for value in bracket)
                or not np.isfinite(bracket).all()
                or not float(bracket[0]) < float(bracket[1])
                or zero_velocity.get("status") != "resolved_for_this_finite_protocol"
                or not isinstance(interpolated, (int, float))
                or not np.isfinite(interpolated)
                or not float(bracket[0]) <= float(interpolated) <= float(bracket[1])
            ):
                raise ValueError(
                    f"{path}: malformed resolved zero-velocity result: bracket={bracket!r}, "
                    f"interpolated={interpolated!r}, status="
                    f"{zero_velocity.get('status')!r}."
                )
            interpolated_temperature = float(interpolated)
            interpolated_temperatures.append(interpolated_temperature)
            melting_error = abs(interpolated_temperature - reference.temperature_K)
            if melting_error > reference.maximum_absolute_error_K:
                failures.append(
                    f"{path}: predicted melting temperature {interpolated_temperature:.3f} K "
                    f"differs from reference {reference.temperature_K:.3f} K by "
                    f"{melting_error:.3f} K, above {reference.maximum_absolute_error_K:.3f} K"
                )
        scans.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "protocol": protocol,
                "zero_velocity": zero_velocity,
            }
        )
    if len(paths) < reference.minimum_protocol_count:
        failures.append(
            f"only {len(paths)} coexistence protocols supplied; at least "
            f"{reference.minimum_protocol_count} independent finite-size/orientation protocols "
            f"are required"
        )
    if len(protocol_signatures) < reference.minimum_protocol_count:
        failures.append(
            "coexistence summaries do not provide enough distinct cell-size/orientation "
            "protocols for finite-protocol validation"
        )
    if len(interpolated_temperatures) >= 2:
        spread = max(interpolated_temperatures) - min(interpolated_temperatures)
        if spread > reference.maximum_protocol_spread_K:
            failures.append(
                f"coexistence protocol melting-temperature spread {spread:.3f} K exceeds "
                f"{reference.maximum_protocol_spread_K:.3f} K"
            )
    evidence = {
        "reference": asdict(reference),
        "scans": scans,
        "distinct_protocol_count": len(protocol_signatures),
        "interpolated_temperatures_K": interpolated_temperatures,
    }
    return evidence, failures


def run_potential_benchmark(
    config: PotentialBenchmarkConfig,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    generator_configs = [load_config(path) for path in config.model_configs]
    model_names = [item.potential.model_name for item in generator_configs]
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            f"Benchmark model_name values must be unique, got {model_names}. "
            f"Use explicit names for distinct checkpoints/heads."
        )
    if any(item.system.chemical_symbol != config.chemical_symbol for item in generator_configs):
        raise ValueError(
            f"Every generator config must target {config.chemical_symbol}; got "
            f"{[item.system.chemical_symbol for item in generator_configs]}."
        )

    application_frames = {
        label: read(path, index=-1)
        for label, path in config.application_structures.items()
    }
    for label, atoms in application_frames.items():
        if not np.all(atoms.numbers == atomic_numbers[config.chemical_symbol]) or not np.all(
            atoms.pbc
        ):
            raise ValueError(
                f"Application structure {label!r} must contain only "
                f"{config.chemical_symbol} and be fully periodic."
            )
    if config.dft_reference is None:
        reference_frames = None
        dft_evidence = None
    else:
        reference_frames, dft_evidence = _read_reference_frames(
            config.dft_reference,
            chemical_symbol=config.chemical_symbol,
            required_states=config.required_dft_states,
            scope=config.qualification_scope,
        )
    results: dict[str, Any] = {}
    melting_results: dict[str, Any] = {}
    application_force_predictions: dict[str, dict[str, np.ndarray]] = {}
    for generator_config in generator_configs:
        potential = generator_config.potential
        model_name = potential.model_name
        melting_result, melting_failures = _melting_evidence(
            config.melting_scan_summaries.get(model_name),
            model_sha256=potential.sha256,
            head=potential.head,
            implementation_class=(
                "src.data_utils.synthetic.atomistic.calculator."
                "VerletSkinMACECalculator"
            ),
            calculator_settings=potential_calculator_settings(potential),
            chemical_symbol=config.chemical_symbol,
            scope=config.qualification_scope,
            reference=config.melting_reference,
        )
        melting_results[model_name] = melting_result
        progress(
            f"Benchmarking {model_name}: family={potential.family}, "
            f"head={potential.head}, sha256={potential.sha256}"
        )
        calculator, execution_provenance = select_calculator(
            generator_config,
            calculator=None,
            injected_calculator_identity=None,
        )
        calculator_provenance = execution_provenance.calculator
        if (
            calculator_provenance.model_sha256 != potential.sha256
            or calculator_provenance.head != potential.head
            or calculator_provenance.model_name != potential.model_name
        ):
            raise RuntimeError(
                f"Benchmark calculator identity {calculator_provenance.identity!r} does not "
                f"match configured model={potential.model_name!r}, sha256={potential.sha256!r}, "
                f"head={potential.head!r}. Qualification cannot be attributed to a different "
                "calculator."
            )
        eos_results = {
            phase: equation_of_state_metrics(
                calculator,
                symbol=config.chemical_symbol,
                phase=phase,
                reference_fcc_a_A=config.reference_fcc_lattice_constant_A,
                volume_scales=config.eos_volume_scales,
                repetitions=config.supercell_repetitions,
            )
            for phase in ("fcc", "hcp", "bcc")
        }
        fcc_energy = eos_results["fcc"]["equilibrium_energy_per_atom_eV"]
        phase_energy_differences = {
            phase: float(
                1000.0
                * (metrics["equilibrium_energy_per_atom_eV"] - fcc_energy)
            )
            for phase, metrics in eos_results.items()
        }
        ground_state = min(
            phase_energy_differences, key=phase_energy_differences.__getitem__
        )
        fcc_lattice_constant_A = eos_results["fcc"]["equilibrium_lattice_constant_A"]
        cache_parity = verlet_cache_parity(
            calculator,
            chemical_symbol=config.chemical_symbol,
            lattice_constant_A=fcc_lattice_constant_A,
            repetitions=config.supercell_repetitions,
            random_seed=config.nve_random_seed,
        )
        if reference_frames is None:
            nve_initial_states = {
                "fcc_solid_diagnostic": (
                    bulk(
                        config.chemical_symbol,
                        "fcc",
                        a=fcc_lattice_constant_A,
                        cubic=True,
                    ).repeat(config.supercell_repetitions),
                    config.nve_temperature_K,
                )
            }
        else:
            nve_initial_states = {}
            for state_index, state in enumerate(
                ("solid_bulk", "liquid_bulk", "interface")
            ):
                state_frames = [
                    atoms for atoms in reference_frames if atoms.info["state"] == state
                ]
                target_temperature = config.qualification_scope.state_temperature_ranges_K[
                    state
                ][1]
                selected_frame = min(
                    state_frames,
                    key=lambda atoms: abs(
                        float(atoms.info["temperature_K"]) - target_temperature
                    ),
                )
                nve_initial_states[state] = (selected_frame, target_temperature)
        nve_results = {
            state: [
                nve_energy_conservation_from_atoms(
                    initial_atoms,
                    calculator,
                    temperature_K=temperature_K,
                    timestep_fs=timestep_fs,
                    duration_ps=config.nve_duration_ps,
                    sample_interval_fs=config.nve_sample_interval_fs,
                    random_seed=config.nve_random_seed + state_index,
                )
                for timestep_fs in config.nve_timesteps_fs
            ]
            for state_index, (state, (initial_atoms, temperature_K)) in enumerate(
                nve_initial_states.items()
            )
        }
        application_evaluations = {
            label: _evaluate_structure(atoms, calculator)
            for label, atoms in application_frames.items()
        }
        application_results = {
            label: evaluation[0]
            for label, evaluation in application_evaluations.items()
        }
        application_force_predictions[model_name] = {
            label: evaluation[1]
            for label, evaluation in application_evaluations.items()
        }
        dft_results = (
            None if reference_frames is None else dft_error_metrics(reference_frames, calculator)
        )

        failure_reasons: list[str] = []
        if ground_state != "fcc":
            failure_reasons.append(
                f"static phase ordering predicts {ground_state} rather than FCC as the Al ground state"
            )
        for metric, threshold in (
            (
                "energy_difference_meV_per_atom",
                config.maximum_cache_energy_difference_meV_per_atom,
            ),
            ("force_rmse_eV_per_A", config.maximum_cache_force_rmse_eV_per_A),
            (
                "maximum_stress_difference_GPa",
                config.maximum_cache_stress_difference_GPa,
            ),
        ):
            observed = cache_parity[metric]
            if observed > threshold:
                failure_reasons.append(
                    f"Verlet cache parity {metric}={observed:.6g} exceeds {threshold:.6g}"
                )
        for state, state_nve_results in nve_results.items():
            maximum_absolute_drift = max(
                abs(item["drift_meV_per_atom_ps"]) for item in state_nve_results
            )
            maximum_excursion = max(
                item["maximum_excursion_meV_per_atom"] for item in state_nve_results
            )
            maximum_detrended_rms = max(
                item["detrended_rms_meV_per_atom"] for item in state_nve_results
            )
            drift_difference = max(
                item["drift_meV_per_atom_ps"] for item in state_nve_results
            ) - min(item["drift_meV_per_atom_ps"] for item in state_nve_results)
            for metric_name, observed, allowed in (
                (
                    "absolute drift (meV/atom/ps)",
                    maximum_absolute_drift,
                    config.maximum_nve_drift_meV_per_atom_ps,
                ),
                (
                    "maximum excursion (meV/atom)",
                    maximum_excursion,
                    config.maximum_nve_excursion_meV_per_atom,
                ),
                (
                    "detrended RMS (meV/atom)",
                    maximum_detrended_rms,
                    config.maximum_nve_detrended_rms_meV_per_atom,
                ),
                (
                    "timestep drift difference (meV/atom/ps)",
                    drift_difference,
                    config.maximum_nve_drift_difference_meV_per_atom_ps,
                ),
            ):
                if not np.isfinite(observed) or observed > allowed:
                    failure_reasons.append(
                        f"NVE state={state} {metric_name}={observed:.6g} exceeds "
                        f"{allowed:.6g}"
                    )
        if dft_results is None:
            failure_reasons.append(
                "no provenance-complete Al DFT reference covering solid, liquid, "
                "interface, strained, nucleus, HCP, and BCC states was supplied"
            )
        else:
            for state, state_metrics in dft_results["by_state"].items():
                for metric, threshold in (
                    (
                        "energy_rmse_meV_per_atom_after_global_constant_offset",
                        config.maximum_dft_energy_rmse_meV_per_atom,
                    ),
                    ("force_rmse_eV_per_A", config.maximum_dft_force_rmse_eV_per_A),
                    ("stress_rmse_GPa", config.maximum_dft_stress_rmse_GPa),
                ):
                    observed = state_metrics[metric]
                    if not np.isfinite(observed) or observed > threshold:
                        failure_reasons.append(
                            f"DFT state={state} {metric}={observed:.6g} exceeds "
                            f"threshold {threshold:.6g}"
                        )
        failure_reasons.extend(melting_failures)

        results[model_name] = {
            "identity": {
                "model_name": calculator_provenance.model_name,
                "family": calculator_provenance.family,
                "head": calculator_provenance.head,
                "sha256": calculator_provenance.model_sha256,
                "source_url": calculator_provenance.source_url,
                "license_identifier": calculator_provenance.license_identifier,
                "model_path": calculator_provenance.model_path,
                "usage_mode": calculator_provenance.usage_mode,
                "implementation_class": calculator_provenance.implementation_class,
                "calculator_settings": calculator_provenance.settings,
            },
            "execution_provenance": execution_provenance.to_dict(),
            "equation_of_state": eos_results,
            "phase_energy_difference_from_fcc_meV_per_atom": phase_energy_differences,
            "predicted_ground_state": ground_state,
            "verlet_cache_parity": cache_parity,
            "nve": nve_results,
            "application_structures": application_results,
            "dft_reference_errors": dft_results,
            "scientifically_qualified": not failure_reasons,
            "qualification_failures": failure_reasons,
        }

    application_disagreement: dict[str, Any] = {}
    baseline_name = model_names[0]
    baseline_fcc_energy = results[baseline_name]["equation_of_state"]["fcc"][
        "equilibrium_energy_per_atom_eV"
    ]
    for candidate_name in model_names[1:]:
        candidate_fcc_energy = results[candidate_name]["equation_of_state"]["fcc"][
            "equilibrium_energy_per_atom_eV"
        ]
        pair_key = f"{candidate_name}_minus_{baseline_name}"
        pair_result: dict[str, Any] = {}
        for label in application_frames:
            baseline_forces = application_force_predictions[baseline_name][label]
            candidate_forces = application_force_predictions[candidate_name][label]
            baseline_excess = (
                results[baseline_name]["application_structures"][label][
                    "energy_per_atom_eV"
                ]
                - baseline_fcc_energy
            )
            candidate_excess = (
                results[candidate_name]["application_structures"][label][
                    "energy_per_atom_eV"
                ]
                - candidate_fcc_energy
            )
            pair_result[label] = {
                "relative_excess_energy_difference_meV_per_atom": float(
                    1000.0 * (candidate_excess - baseline_excess)
                ),
                "force_rmse_eV_per_A": float(
                    np.sqrt(np.mean(np.square(candidate_forces - baseline_forces)))
                ),
            }
        application_disagreement[pair_key] = pair_result

    qualified_candidates = [
        name
        for name in model_names[1:]
        if results[name]["scientifically_qualified"]
    ]
    report = {
        "schema_version": 1,
        "report_type": "al_crystallization_mlip_benchmark",
        "benchmark_config": config.to_dict(),
        "models": results,
        "application_model_disagreement": application_disagreement,
        "dft_reference_evidence": dft_evidence,
        "melting_scans": melting_results,
        "selection": {
            "qualified_models": [
                name for name, result in results.items() if result["scientifically_qualified"]
            ],
            "qualified_replacement_candidates": qualified_candidates,
            "automatic_replacement_allowed": False,
            "manual_comparison_required": True,
            "policy": (
                "Qualification establishes eligibility, not superiority. A newer foundation "
                "model never replaces the baseline automatically; compare per-state DFT "
                "errors, thermodynamics, stability, speed, and uncertainty for the intended "
                "claim."
            ),
        },
    }
    qualification_reports: dict[str, str] = {}
    qualification_payloads: list[tuple[Path, dict[str, Any]]] = []
    for model_name, model_result in results.items():
        identity = model_result["identity"]
        effective_authorized_claims = {
            claim: bool(
                model_result["scientifically_qualified"]
                and config.qualification_scope.authorized_claims[claim]
            )
            for claim in POTENTIAL_CLAIM_NAMES
        }
        qualification_path = config.output_json.with_name(
            f"{config.output_json.stem}.{model_name}.qualification.json"
        )
        qualification_report = {
            "schema_version": 1,
            "report_type": "al_crystallization_mlip_qualification",
            "model_name": model_name,
            "model_sha256": identity["sha256"],
            "head": identity["head"],
            "scientifically_qualified": model_result["scientifically_qualified"],
            "qualification_failures": model_result["qualification_failures"],
            "benchmark_report": str(config.output_json),
            "scope": {
                "chemical_symbol": config.chemical_symbol,
                "pressure_range_GPa": list(
                    config.qualification_scope.pressure_range_GPa
                ),
                "state_temperature_ranges_K": {
                    state: list(temperature_range)
                    for state, temperature_range in (
                        config.qualification_scope.state_temperature_ranges_K.items()
                    )
                },
                "maximum_timestep_fs": (
                    config.qualification_scope.maximum_timestep_fs
                ),
                "authorized_claims": effective_authorized_claims,
                "calculator_settings": {
                    "implementation_class": identity["implementation_class"],
                    **identity["calculator_settings"],
                },
            },
            "evidence": {
                "benchmark_config_canonical_sha256": hashlib.sha256(
                    json.dumps(
                        config.to_dict(), sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "benchmark_config_file_sha256": _sha256_file(config.config_path),
                "dft_reference": dft_evidence,
                "melting_scans": melting_results[model_name],
            },
            "equation_of_state": model_result["equation_of_state"],
            "phase_energy_difference_from_fcc_meV_per_atom": model_result[
                "phase_energy_difference_from_fcc_meV_per_atom"
            ],
            "nve": model_result["nve"],
            "dft_reference_errors": model_result["dft_reference_errors"],
            "melting_scan": melting_results[model_name],
        }
        qualification_reports[model_name] = str(qualification_path)
        qualification_payloads.append((qualification_path, qualification_report))
    report["qualification_reports"] = qualification_reports
    _write_json_atomic(config.output_json, report)
    benchmark_report_sha256 = _sha256_file(config.output_json)
    for qualification_path, qualification_report in qualification_payloads:
        qualification_report["benchmark_report_sha256"] = benchmark_report_sha256
        _write_json_atomic(qualification_path, qualification_report)
    progress(f"Wrote potential benchmark report to {config.output_json}")
    return report
