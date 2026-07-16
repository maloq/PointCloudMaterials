from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from .artifacts import PHASE_NAMES
from .simulation import ThermodynamicTrace
from .transition_config import TransitionBranchConfig


STRUCTURE_NAMES = ("other", "fcc", "hcp", "bcc", "ico")
CRYSTALLINE_STRUCTURE_TYPES = np.array([1, 2, 3], dtype=np.int32)
STRUCTURE_COLORS = (
    "#8d99ae",
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#e9c46a",
)


@dataclass(frozen=True)
class ThermodynamicStationarity:
    equilibration_tail_mean_temperature_K: float
    equilibration_tail_mean_pressure_GPa: float
    production_start_mean_temperature_K: float
    production_start_mean_pressure_GPa: float
    steady_state_mean_temperature_K: float
    steady_state_mean_pressure_GPa: float
    steady_state_temperature_block_drift_K: float
    steady_state_pressure_block_drift_GPa: float


@dataclass(frozen=True)
class TransitionAnalysis:
    step: np.ndarray
    time_ps: np.ndarray
    structure_fractions: np.ndarray
    crystalline_fraction: np.ndarray
    prepared_liquid_slab_crystalline_fraction: np.ndarray
    prepared_solid_region_crystalline_fraction: np.ndarray
    crystalline_profile: np.ndarray
    smoothed_crystalline_profile: np.ndarray
    profile_bin_centers_fractional: np.ndarray
    profile_threshold: float
    liquid_profile_baseline: float
    solid_profile_baseline: float
    profile_contrast: np.ndarray
    interface_positions_fractional: np.ndarray
    signed_interface_advance_A: np.ndarray
    mean_interface_advance_A: np.ndarray
    net_crystalline_fraction_change: float
    net_mean_interface_advance_A: float
    fitted_interface_velocity_m_per_s: float
    individual_interface_velocities_m_per_s: np.ndarray
    individual_interface_fit_r_squared: np.ndarray
    velocity_fit_r_squared: float
    velocity_fit_ols_standard_error_m_per_s: float
    velocity_fit_residual_rms_A: float
    velocity_fit_start_step: int
    velocity_fit_end_step: int
    stationarity: ThermodynamicStationarity


@dataclass(frozen=True)
class PhaseRdfAnalysis:
    step: np.ndarray
    time_ps: np.ndarray
    distance_A: np.ndarray
    phase_names: tuple[str, ...]
    g_r: np.ndarray


def phase_rdf_metadata(
    analysis: PhaseRdfAnalysis,
    *,
    cutoff_A: float,
    bins: int,
) -> dict[str, object]:
    return {
        "definition": (
            "Total-Al radial distribution around central atoms grouped by their initial "
            "prepared phase provenance; neighboring atoms may have any phase label."
        ),
        "normalization": "whole-cell instantaneous number density",
        "backend": "OVITO compiled partial radial distribution function",
        "phase_names": list(analysis.phase_names),
        "cutoff_A": cutoff_A,
        "bins": bins,
        "frames": len(analysis.step),
    }


def write_phase_rdf_archive(path: Path, analysis: PhaseRdfAnalysis) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            distance_A=analysis.distance_A,
            phase_names=np.asarray(analysis.phase_names),
            g_r=analysis.g_r,
        )


def _ptm_structure_types(atoms: Atoms, rmsd_cutoff: float) -> np.ndarray:
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
        from ovito.pipeline import Pipeline, StaticSource
    except ImportError as exc:
        raise ImportError(
            "Transition analysis requires OVITO Polyhedral Template Matching. Install the "
            "repository requirements in the pointnet environment."
        ) from exc
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))
    modifier = PolyhedralTemplateMatchingModifier()
    modifier.rmsd_cutoff = rmsd_cutoff
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()
    return np.asarray(data.particles["Structure Type"], dtype=np.int32)


def analyze_phase_rdf(
    trace: ThermodynamicTrace,
    *,
    chemical_symbol: str,
    prepared_phase_ids: np.ndarray,
    timestep_fs: float,
    cutoff_A: float,
    bins: int,
    branch_name: str,
    progress: Callable[[str], None] = print,
) -> PhaseRdfAnalysis:
    """Compute a center-conditioned Al RDF for each prepared phase provenance.

    Neighbors may belong to any phase. Selecting only the central atoms avoids treating the
    spatially finite interface and slab regions as independent periodic bulk systems.
    """

    frame_count, atom_count, _ = trace.positions_A.shape
    try:
        from ovito.data import ParticleType
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import CoordinationAnalysisModifier
    except ImportError as exc:
        raise ImportError(
            "Per-phase RDF analysis requires OVITO's compiled coordination analysis. "
            "Install the repository requirements in the pointnet environment."
        ) from exc

    phase_fractions = (
        np.bincount(prepared_phase_ids, minlength=len(PHASE_NAMES)) / atom_count
    )
    g_r = np.empty((frame_count, len(PHASE_NAMES), bins), dtype=np.float64)
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    prepared_particle_types = prepared_phase_ids.astype(np.int32) + 1
    phase_pairs = tuple(
        (first, second)
        for first in range(len(PHASE_NAMES))
        for second in range(first, len(PHASE_NAMES))
    )
    expected_components = tuple(
        f"{PHASE_NAMES[first]}-{PHASE_NAMES[second]}"
        for first, second in phase_pairs
    )
    modifier = CoordinationAnalysisModifier(
        cutoff=cutoff_A,
        number_of_bins=bins,
        partial=True,
        type_property="Prepared Phase",
    )
    progress(
        f"{branch_name}: per-phase RDF audit of {frame_count} frames "
        f"to {cutoff_A:.2f} A ({bins} bins, OVITO compiled backend)"
    )

    distance_A: np.ndarray | None = None
    for frame_index, (positions_A, cell_A) in enumerate(
        zip(trace.positions_A, trace.cell_vectors_A)
    ):
        atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
        data = ase_to_ovito(atoms)
        phase_property = data.particles_.create_property(
            "Prepared Phase", data=prepared_particle_types
        )
        for phase_id, phase_name in enumerate(PHASE_NAMES, start=1):
            phase_property.types.append(ParticleType(id=phase_id, name=phase_name))
        data.apply(modifier)
        table = data.tables["coordination-rdf"]
        components = tuple(table.y.component_names)
        if components != expected_components:
            raise RuntimeError(
                f"{branch_name}: OVITO returned RDF components {components}, expected "
                f"{expected_components}."
            )
        table_values = table.xy()
        if distance_A is None:
            distance_A = table_values[:, 0].copy()
        partial_rdf = table_values[:, 1:].T
        frame_rdf = np.zeros((len(PHASE_NAMES), bins), dtype=np.float64)
        for pair_index, (first, second) in enumerate(phase_pairs):
            frame_rdf[first] += phase_fractions[second] * partial_rdf[pair_index]
            if first != second:
                frame_rdf[second] += phase_fractions[first] * partial_rdf[pair_index]
        g_r[frame_index] = frame_rdf

    if distance_A is None:
        raise RuntimeError(
            f"{branch_name}: RDF analysis received an empty thermodynamic trace."
        )

    return PhaseRdfAnalysis(
        step=trace.step.copy(),
        time_ps=trace.step.astype(np.float64) * timestep_fs / 1000.0,
        distance_A=distance_A,
        phase_names=PHASE_NAMES,
        g_r=g_r,
    )


def _cyclic_smooth(profiles: np.ndarray, smoothing_bins: int) -> np.ndarray:
    half_width = smoothing_bins // 2
    smoothed = np.zeros_like(profiles)
    for offset in range(-half_width, half_width + 1):
        smoothed += np.roll(profiles, offset, axis=1)
    return smoothed / smoothing_bins


def _cyclic_distance(values: np.ndarray, reference: float) -> np.ndarray:
    return np.abs((values - reference + 0.5) % 1.0 - 0.5)


def _oriented_crossing(
    profile: np.ndarray,
    *,
    threshold: float,
    orientation: int,
    reference: float,
    branch_name: str,
    frame_step: int,
) -> float:
    bin_count = len(profile)
    centers = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
    candidates: list[float] = []
    for index in range(bin_count):
        next_index = (index + 1) % bin_count
        first_value = float(profile[index])
        second_value = float(profile[next_index])
        slope = second_value - first_value
        if orientation * slope <= 0.0:
            continue
        first_offset = first_value - threshold
        second_offset = second_value - threshold
        if first_offset * second_offset > 0.0:
            continue
        first_position = float(centers[index])
        second_position = float(centers[next_index])
        if next_index == 0:
            second_position += 1.0
        fraction = (threshold - first_value) / slope
        candidates.append((first_position + fraction * (second_position - first_position)) % 1.0)
    if not candidates:
        direction = "increasing" if orientation > 0 else "decreasing"
        raise RuntimeError(
            f"{branch_name}: no {direction} PTM-profile crossing of threshold "
            f"{threshold:.4f} exists at production step {frame_step}. Both phases must remain "
            "present throughout the fitted coexistence trajectory."
        )
    candidate_array = np.asarray(candidates, dtype=np.float64)
    candidate_distances = _cyclic_distance(candidate_array, reference)
    selected_index = int(np.argmin(candidate_distances))
    maximum_tracking_jump = 2.0 / bin_count
    if candidate_distances[selected_index] > maximum_tracking_jump:
        raise RuntimeError(
            f"{branch_name}: nearest oriented PTM-profile crossing jumps by "
            f"{candidate_distances[selected_index]:.5f} fractional cell at production step "
            f"{frame_step}, exceeding two profile bins ({maximum_tracking_jump:.5f}). The "
            "original planar front was lost or became ambiguous."
        )
    return float(candidate_array[selected_index])


def _unwrap_fractional_positions(values: np.ndarray) -> np.ndarray:
    unwrapped = np.empty_like(values)
    unwrapped[0] = values[0]
    for index in range(1, len(values)):
        delta = (values[index] - values[index - 1] + 0.5) % 1.0 - 0.5
        unwrapped[index] = unwrapped[index - 1] + delta
    return unwrapped


def _linear_fit(
    time_ps: np.ndarray, values_A: np.ndarray
) -> tuple[float, float, float, float]:
    design = np.column_stack((time_ps, np.ones_like(time_ps)))
    slope, intercept = np.linalg.lstsq(design, values_A, rcond=None)[0]
    fitted = slope * time_ps + intercept
    residual_sum = float(np.sum((values_A - fitted) ** 2))
    centered_sum = float(np.sum((values_A - np.mean(values_A)) ** 2))
    if centered_sum == 0.0:
        r_squared = 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
    else:
        r_squared = 1.0 - residual_sum / centered_sum
    residual_rms_A = float(np.sqrt(residual_sum / len(time_ps)))
    centered_time_sum = float(np.sum((time_ps - np.mean(time_ps)) ** 2))
    slope_standard_error = float(
        np.sqrt((residual_sum / (len(time_ps) - 2)) / centered_time_sum)
    )
    return float(slope), float(r_squared), slope_standard_error, residual_rms_A


def _thermodynamic_stationarity(
    equilibration_trace: ThermodynamicTrace,
    production_trace: ThermodynamicTrace,
    *,
    steady_mask: np.ndarray,
    target_temperature_K: float,
    target_pressure_GPa: float,
    maximum_temperature_error_K: float,
    maximum_pressure_error_GPa: float,
    branch_name: str,
) -> ThermodynamicStationarity:
    equilibration_tail_start = len(equilibration_trace.step) // 2
    equilibration_temperature = float(
        np.mean(equilibration_trace.temperature_K[equilibration_tail_start:])
    )
    equilibration_pressure = float(
        np.mean(equilibration_trace.pressure_GPa[equilibration_tail_start:])
    )
    startup_frame_count = min(5, len(production_trace.step))
    startup_temperature = float(
        np.mean(production_trace.temperature_K[:startup_frame_count])
    )
    startup_pressure = float(np.mean(production_trace.pressure_GPa[:startup_frame_count]))
    steady_temperature = production_trace.temperature_K[steady_mask]
    steady_pressure = production_trace.pressure_GPa[steady_mask]
    split = len(steady_temperature) // 2
    first_temperature = float(np.mean(steady_temperature[:split]))
    second_temperature = float(np.mean(steady_temperature[split:]))
    first_pressure = float(np.mean(steady_pressure[:split]))
    second_pressure = float(np.mean(steady_pressure[split:]))
    temperature_drift = second_temperature - first_temperature
    pressure_drift = second_pressure - first_pressure
    checks = (
        (
            "equilibration tail temperature",
            abs(equilibration_temperature - target_temperature_K),
            maximum_temperature_error_K,
            "K",
        ),
        (
            "equilibration tail pressure",
            abs(equilibration_pressure - target_pressure_GPa),
            maximum_pressure_error_GPa,
            "GPa",
        ),
        (
            "production startup temperature discontinuity",
            abs(startup_temperature - equilibration_temperature),
            maximum_temperature_error_K,
            "K",
        ),
        (
            "production startup pressure discontinuity",
            abs(startup_pressure - equilibration_pressure),
            maximum_pressure_error_GPa,
            "GPa",
        ),
        (
            "steady-window mean temperature",
            abs(float(np.mean(steady_temperature)) - target_temperature_K),
            maximum_temperature_error_K,
            "K",
        ),
        (
            "steady-window mean pressure",
            abs(float(np.mean(steady_pressure)) - target_pressure_GPa),
            maximum_pressure_error_GPa,
            "GPa",
        ),
        (
            "steady-window temperature block drift",
            abs(temperature_drift),
            maximum_temperature_error_K,
            "K",
        ),
        (
            "steady-window pressure block drift",
            abs(pressure_drift),
            maximum_pressure_error_GPa,
            "GPa",
        ),
    )
    for description, observed_error, maximum_error, unit in checks:
        if observed_error > maximum_error:
            raise RuntimeError(
                f"{branch_name}: {description} is {observed_error:.6f} {unit}, above the "
                f"allowed {maximum_error:.6f} {unit}. The selected velocity window is not "
                "thermodynamically stationary; extend equilibration or move the fit start."
            )
    return ThermodynamicStationarity(
        equilibration_tail_mean_temperature_K=equilibration_temperature,
        equilibration_tail_mean_pressure_GPa=equilibration_pressure,
        production_start_mean_temperature_K=startup_temperature,
        production_start_mean_pressure_GPa=startup_pressure,
        steady_state_mean_temperature_K=float(np.mean(steady_temperature)),
        steady_state_mean_pressure_GPa=float(np.mean(steady_pressure)),
        steady_state_temperature_block_drift_K=temperature_drift,
        steady_state_pressure_block_drift_GPa=pressure_drift,
    )


def analyze_transition(
    trace: ThermodynamicTrace,
    *,
    equilibration_trace: ThermodynamicTrace,
    chemical_symbol: str,
    timestep_fs: float,
    slab_bounds_fractional: tuple[float, float],
    profile_bins: int,
    profile_smoothing_bins: int,
    ptm_rmsd_cutoff: float,
    minimum_profile_contrast: float,
    minimum_velocity_fit_r_squared: float,
    target_pressure_GPa: float,
    maximum_temperature_error_K: float,
    maximum_pressure_error_GPa: float,
    branch: TransitionBranchConfig,
    progress: Callable[[str], None] = print,
) -> TransitionAnalysis:
    frame_count, atom_count, _ = trace.positions_A.shape
    progress(
        f"{branch.name}: spatial PTM audit of {frame_count} production frames "
        f"({atom_count} atoms/frame)"
    )
    structure_fractions = np.empty((frame_count, len(STRUCTURE_NAMES)), dtype=np.float64)
    crystalline_fraction = np.empty(frame_count, dtype=np.float64)
    liquid_slab_fraction = np.empty(frame_count, dtype=np.float64)
    solid_region_fraction = np.empty(frame_count, dtype=np.float64)
    crystalline_profile = np.empty((frame_count, profile_bins), dtype=np.float64)
    bin_centers = (np.arange(profile_bins, dtype=np.float64) + 0.5) / profile_bins
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    lower, upper = slab_bounds_fractional

    for frame_index, (positions_A, cell_A) in enumerate(
        zip(trace.positions_A, trace.cell_vectors_A)
    ):
        atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
        structure_types = _ptm_structure_types(atoms, ptm_rmsd_cutoff)
        counts = np.bincount(structure_types, minlength=len(STRUCTURE_NAMES))[: len(STRUCTURE_NAMES)]
        structure_fractions[frame_index] = counts / atom_count
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        crystalline_fraction[frame_index] = float(np.mean(crystalline))

        scaled_z = np.linalg.solve(np.asarray(cell_A).T, np.asarray(positions_A).T).T[:, 2] % 1.0
        liquid_slab = (scaled_z >= lower) & (scaled_z < upper)
        solid_region = ~liquid_slab
        liquid_slab_fraction[frame_index] = float(np.mean(crystalline[liquid_slab]))
        solid_region_fraction[frame_index] = float(np.mean(crystalline[solid_region]))

        bin_indices = np.floor(scaled_z * profile_bins).astype(np.int64)
        bin_counts = np.bincount(bin_indices, minlength=profile_bins)
        if np.any(bin_counts == 0):
            empty_bins = np.flatnonzero(bin_counts == 0).tolist()
            raise RuntimeError(
                f"{branch.name}: empty transition profile bins {empty_bins}; "
                f"profile_bins={profile_bins}, atom_count={atom_count}."
            )
        crystalline_counts = np.bincount(
            bin_indices,
            weights=crystalline.astype(np.float64),
            minlength=profile_bins,
        )
        crystalline_profile[frame_index] = crystalline_counts / bin_counts

    smoothed_profile = _cyclic_smooth(crystalline_profile, profile_smoothing_bins)
    baseline_mask = trace.step < branch.steady_state_start_step
    if not np.any(baseline_mask):
        raise RuntimeError(
            f"{branch.name}: no post-equilibration baseline frames occur before "
            f"steady_state_start_step={branch.steady_state_start_step}."
        )
    liquid_width = upper - lower
    solid_width = 1.0 - liquid_width
    liquid_core = (bin_centers >= lower + 0.25 * liquid_width) & (
        bin_centers <= upper - 0.25 * liquid_width
    )
    solid_center = (upper + 0.5 * solid_width) % 1.0
    solid_core = _cyclic_distance(bin_centers, solid_center) <= 0.25 * solid_width
    if not np.any(liquid_core) or not np.any(solid_core):
        raise RuntimeError(
            f"{branch.name}: profile_bins={profile_bins} does not resolve both bulk cores for "
            f"slab_bounds_fractional={slab_bounds_fractional}."
        )
    baseline_profile = np.mean(smoothed_profile[baseline_mask], axis=0)
    liquid_baseline = float(np.mean(baseline_profile[liquid_core]))
    solid_baseline = float(np.mean(baseline_profile[solid_core]))
    profile_contrast = solid_baseline - liquid_baseline
    if profile_contrast < minimum_profile_contrast:
        raise RuntimeError(
            f"{branch.name}: post-equilibration PTM profile contrast is {profile_contrast:.4f} "
            f"(solid core={solid_baseline:.4f}, liquid core={liquid_baseline:.4f}), below "
            f"analysis.minimum_profile_contrast={minimum_profile_contrast:.4f}. A spatial "
            "solid-liquid interface is not resolved."
        )
    profile_threshold = 0.5 * (solid_baseline + liquid_baseline)
    frame_profile_contrast = np.mean(smoothed_profile[:, solid_core], axis=1) - np.mean(
        smoothed_profile[:, liquid_core], axis=1
    )

    interface_positions = np.empty((frame_count, 2), dtype=np.float64)
    lower_reference, upper_reference = lower, upper
    for frame_index, profile in enumerate(smoothed_profile):
        lower_position = _oriented_crossing(
            profile,
            threshold=profile_threshold,
            orientation=-1,
            reference=lower_reference,
            branch_name=branch.name,
            frame_step=int(trace.step[frame_index]),
        )
        upper_position = _oriented_crossing(
            profile,
            threshold=profile_threshold,
            orientation=1,
            reference=upper_reference,
            branch_name=branch.name,
            frame_step=int(trace.step[frame_index]),
        )
        interface_positions[frame_index] = (lower_position, upper_position)
        lower_reference, upper_reference = lower_position, upper_position

    unwrapped_lower = _unwrap_fractional_positions(interface_positions[:, 0])
    unwrapped_upper = _unwrap_fractional_positions(interface_positions[:, 1])
    reference_cell = np.asarray(trace.cell_vectors_A[0], dtype=np.float64)
    reference_height_A = float(
        abs(np.linalg.det(reference_cell))
        / np.linalg.norm(np.cross(reference_cell[0], reference_cell[1]))
    )
    signed_interface_advance_A = np.column_stack(
        (
            (unwrapped_lower - unwrapped_lower[0]) * reference_height_A,
            (unwrapped_upper[0] - unwrapped_upper) * reference_height_A,
        )
    )
    mean_interface_advance_A = np.mean(signed_interface_advance_A, axis=1)

    time_ps = trace.step.astype(np.float64) * timestep_fs / 1000.0
    steady_mask = (trace.step >= branch.steady_state_start_step) & (
        trace.step <= branch.steady_state_end_step
    )
    steady_frame_count = int(np.count_nonzero(steady_mask))
    if steady_frame_count < 3:
        raise RuntimeError(
            f"{branch.name}: velocity fit requires at least three saved frames in steps "
            f"[{branch.steady_state_start_step}, {branch.steady_state_end_step}], found "
            f"{steady_frame_count}."
        )
    low_contrast_fit_frames = np.flatnonzero(
        steady_mask & (frame_profile_contrast < minimum_profile_contrast)
    )
    if len(low_contrast_fit_frames):
        first_index = int(low_contrast_fit_frames[0])
        raise RuntimeError(
            f"{branch.name}: PTM solid-core/liquid-core profile contrast falls to "
            f"{frame_profile_contrast[first_index]:.4f} at fitted production step "
            f"{int(trace.step[first_index])}, below analysis.minimum_profile_contrast="
            f"{minimum_profile_contrast:.4f}. A phase was exhausted or the planar interface "
            "lost spatial resolution; no velocity is reported through that interval."
        )
    stationarity = _thermodynamic_stationarity(
        equilibration_trace,
        trace,
        steady_mask=steady_mask,
        target_temperature_K=branch.temperature_K,
        target_pressure_GPa=target_pressure_GPa,
        maximum_temperature_error_K=maximum_temperature_error_K,
        maximum_pressure_error_GPa=maximum_pressure_error_GPa,
        branch_name=branch.name,
    )
    (
        fit_slope_A_per_ps,
        fit_r_squared,
        fit_slope_standard_error_A_per_ps,
        fit_residual_rms_A,
    ) = _linear_fit(
        time_ps[steady_mask], mean_interface_advance_A[steady_mask]
    )
    individual_fits = tuple(
        _linear_fit(
            time_ps[steady_mask], signed_interface_advance_A[steady_mask, index]
        )
        for index in range(2)
    )
    individual_slopes = np.asarray(
        [fit[0] for fit in individual_fits], dtype=np.float64
    )
    individual_fit_r_squared = np.asarray(
        [fit[1] for fit in individual_fits], dtype=np.float64
    )
    fitted_velocity_m_per_s = fit_slope_A_per_ps * 100.0
    individual_velocities_m_per_s = individual_slopes * 100.0
    if fit_r_squared < minimum_velocity_fit_r_squared:
        raise RuntimeError(
            f"{branch.name}: spatial interface-advance fit over production steps "
            f"[{branch.steady_state_start_step}, {branch.steady_state_end_step}] has "
            f"R^2={fit_r_squared:.4f}, below analysis.minimum_velocity_fit_r_squared="
            f"{minimum_velocity_fit_r_squared:.4f}. The fronts do not exhibit a resolved "
            "constant-velocity interval."
        )
    if np.any(individual_fit_r_squared < minimum_velocity_fit_r_squared):
        raise RuntimeError(
            f"{branch.name}: individual lower/upper interface fits have R^2="
            f"{individual_fit_r_squared.tolist()}, but both must reach "
            f"analysis.minimum_velocity_fit_r_squared="
            f"{minimum_velocity_fit_r_squared:.4f}. The mean would hide unresolved front "
            "fluctuations."
        )
    if (
        minimum_velocity_fit_r_squared > 0.0
        and individual_velocities_m_per_s[0] * individual_velocities_m_per_s[1] <= 0.0
    ):
        raise RuntimeError(
            f"{branch.name}: the two spatial interfaces have opposite or zero fitted "
            f"velocities {individual_velocities_m_per_s.tolist()} m/s. A mean front velocity "
            "is ambiguous; extend the stationary production interval."
        )

    fraction_change = float(crystalline_fraction[-1] - crystalline_fraction[0])
    net_advance_A = float(mean_interface_advance_A[-1] - mean_interface_advance_A[0])
    if branch.expected_direction == "growth":
        signed_change = fraction_change
        signed_velocity = fitted_velocity_m_per_s
    elif branch.expected_direction == "melting":
        signed_change = -fraction_change
        signed_velocity = -fitted_velocity_m_per_s
    else:
        signed_change = np.inf
        signed_velocity = np.inf
    direction_failed = (
        branch.expected_direction != "unconstrained"
        and branch.minimum_crystalline_fraction_change > 0.0
        and signed_velocity <= 0.0
    )
    if signed_change < branch.minimum_crystalline_fraction_change or direction_failed:
        raise RuntimeError(
            f"{branch.name}: expected_direction={branch.expected_direction!r}, but the "
            f"production trajectory has crystalline-fraction change {fraction_change:+.4f} "
            f"and fitted spatial interface velocity {fitted_velocity_m_per_s:+.3f} m/s. "
            f"The direction check requires a fraction change of at least "
            f"{branch.minimum_crystalline_fraction_change:.4f} and a velocity with the same "
            "sign. Adjust the temperature bracket or extend the stationary production run."
        )
    return TransitionAnalysis(
        step=trace.step.copy(),
        time_ps=time_ps,
        structure_fractions=structure_fractions,
        crystalline_fraction=crystalline_fraction,
        prepared_liquid_slab_crystalline_fraction=liquid_slab_fraction,
        prepared_solid_region_crystalline_fraction=solid_region_fraction,
        crystalline_profile=crystalline_profile,
        smoothed_crystalline_profile=smoothed_profile,
        profile_bin_centers_fractional=bin_centers,
        profile_threshold=profile_threshold,
        liquid_profile_baseline=liquid_baseline,
        solid_profile_baseline=solid_baseline,
        profile_contrast=frame_profile_contrast,
        interface_positions_fractional=interface_positions,
        signed_interface_advance_A=signed_interface_advance_A,
        mean_interface_advance_A=mean_interface_advance_A,
        net_crystalline_fraction_change=fraction_change,
        net_mean_interface_advance_A=net_advance_A,
        fitted_interface_velocity_m_per_s=fitted_velocity_m_per_s,
        individual_interface_velocities_m_per_s=individual_velocities_m_per_s,
        individual_interface_fit_r_squared=individual_fit_r_squared,
        velocity_fit_r_squared=fit_r_squared,
        velocity_fit_ols_standard_error_m_per_s=(
            fit_slope_standard_error_A_per_ps * 100.0
        ),
        velocity_fit_residual_rms_A=fit_residual_rms_A,
        velocity_fit_start_step=branch.steady_state_start_step,
        velocity_fit_end_step=branch.steady_state_end_step,
        stationarity=stationarity,
    )


def write_transition_visualization(
    path: Path,
    *,
    trace: ThermodynamicTrace,
    analysis: TransitionAnalysis,
    branch: TransitionBranchConfig,
    pressure_GPa: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.crystalline_fraction,
        color="#264653",
        label="whole cell",
    )
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.prepared_liquid_slab_crystalline_fraction,
        color="#e76f51",
        label="prepared liquid slab",
    )
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.prepared_solid_region_crystalline_fraction,
        color="#2a9d8f",
        label="prepared solid region",
    )
    axes[0, 0].set(xlabel="time (ps)", ylabel="PTM crystalline fraction", ylim=(0.0, 1.0))
    axes[0, 0].legend()

    axes[0, 1].plot(
        analysis.time_ps,
        analysis.mean_interface_advance_A,
        color="#6a4c93",
        label="mean of two fronts",
    )
    axes[0, 1].plot(
        analysis.time_ps,
        analysis.signed_interface_advance_A[:, 0],
        color="#457b9d",
        alpha=0.55,
        label="lower front",
    )
    axes[0, 1].plot(
        analysis.time_ps,
        analysis.signed_interface_advance_A[:, 1],
        color="#e76f51",
        alpha=0.55,
        label="upper front",
    )
    fit_start_ps = analysis.velocity_fit_start_step * (
        analysis.time_ps[-1] / analysis.step[-1]
    )
    fit_end_ps = analysis.velocity_fit_end_step * (
        analysis.time_ps[-1] / analysis.step[-1]
    )
    axes[0, 1].axvspan(fit_start_ps, fit_end_ps, color="#6a4c93", alpha=0.10)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(
        xlabel="production time (ps)",
        ylabel="signed interface advance in reference cell (Å)",
    )
    axes[0, 1].legend()

    temperature_axis = axes[1, 0]
    pressure_axis = temperature_axis.twinx()
    temperature_axis.plot(analysis.time_ps, trace.temperature_K, color="#f4a261")
    temperature_axis.axhline(branch.temperature_K, color="#f4a261", linestyle="--")
    pressure_axis.plot(analysis.time_ps, trace.pressure_GPa, color="#457b9d", alpha=0.8)
    pressure_axis.axhline(pressure_GPa, color="#457b9d", linestyle="--")
    temperature_axis.set(xlabel="time (ps)", ylabel="temperature (K)")
    pressure_axis.set_ylabel("pressure (GPa)")

    axes[1, 1].imshow(
        analysis.smoothed_crystalline_profile,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.0, 1.0, analysis.time_ps[0], analysis.time_ps[-1]),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[1, 1].plot(
        analysis.interface_positions_fractional[:, 0],
        analysis.time_ps,
        color="white",
        linewidth=1.2,
    )
    axes[1, 1].plot(
        analysis.interface_positions_fractional[:, 1],
        analysis.time_ps,
        color="white",
        linewidth=1.2,
    )
    axes[1, 1].set(xlabel="fractional position along interface normal", ylabel="time (ps)")
    figure.suptitle(
        f"{branch.name}: direct solid–liquid coexistence at {branch.temperature_K:.0f} K\n"
        f"Δcrystalline={analysis.net_crystalline_fraction_change:+.3f}, "
        f"spatial-fit velocity={analysis.fitted_interface_velocity_m_per_s:+.1f} m/s, "
        f"R²={analysis.velocity_fit_r_squared:.3f}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_structure_slice_visualization(
    path: Path,
    *,
    trace: ThermodynamicTrace,
    chemical_symbol: str,
    timestep_fs: float,
    reference_planes_fractional: tuple[float, ...],
    simulation_name: str,
    temperature_K: float,
    ptm_rmsd_cutoff: float,
) -> None:
    selected_frames = (0, len(trace.step) // 2, len(trace.step) - 1)
    numbers = np.full(
        trace.positions_A.shape[1],
        atomic_numbers[chemical_symbol],
        dtype=np.int32,
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))

    for axis, frame_index in zip(axes, selected_frames):
        atoms = Atoms(
            numbers=numbers,
            positions=trace.positions_A[frame_index],
            cell=trace.cell_vectors_A[frame_index],
            pbc=True,
        )
        positions_A = atoms.get_positions(wrap=True)
        cell_A = np.asarray(atoms.cell, dtype=np.float64)
        cell_lengths_A = np.linalg.norm(cell_A, axis=1)
        scaled_positions = atoms.get_scaled_positions(wrap=True)
        slice_half_width_A = max(2.5, 0.08 * float(cell_lengths_A[1]))
        slice_mask = (
            np.abs(scaled_positions[:, 1] - 0.5) * cell_lengths_A[1]
            <= slice_half_width_A
        )
        structure_types = _ptm_structure_types(atoms, ptm_rmsd_cutoff)
        for structure_id, (structure_name, color) in enumerate(
            zip(STRUCTURE_NAMES, STRUCTURE_COLORS)
        ):
            mask = slice_mask & (structure_types == structure_id)
            axis.scatter(
                positions_A[mask, 0],
                positions_A[mask, 2],
                s=5.0,
                alpha=0.82,
                linewidths=0.0,
                color=color,
                label=structure_name.upper(),
                rasterized=True,
            )
        for boundary in reference_planes_fractional:
            axis.axhline(
                boundary * cell_lengths_A[2],
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        crystalline_fraction = float(
            np.mean(np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES))
        )
        time_ps = float(trace.step[frame_index] * timestep_fs / 1000.0)
        axis.set(
            xlim=(0.0, cell_lengths_A[0]),
            ylim=(0.0, cell_lengths_A[2]),
            xlabel="x (Å)",
            ylabel="z (Å)",
            title=f"t={time_ps:.2f} ps, crystalline={crystalline_fraction:.3f}",
        )
        axis.set_aspect("equal", adjustable="box")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(STRUCTURE_NAMES),
    )
    audit_note = "PTM colors are audit observables"
    if reference_planes_fractional:
        audit_note += "; dashed lines mark the prepared interfaces"
    figure.suptitle(
        f"{simulation_name}: central-y structure slices at {temperature_K:.0f} K\n"
        f"{audit_note}",
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.15, top=0.82, wspace=0.22)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_phase_rdf_visualization(
    path: Path,
    *,
    analysis: PhaseRdfAnalysis,
    branch: TransitionBranchConfig,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(analysis.phase_names),
        figsize=(16.0, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    selected_frames = (0, len(analysis.step) // 2, len(analysis.step) - 1)
    colors = ("#264653", "#e9c46a", "#e76f51")
    heatmap_limit = float(np.quantile(analysis.g_r, 0.995))

    for phase_index, phase_name in enumerate(analysis.phase_names):
        curve_axis = axes[0, phase_index]
        for frame_index, color in zip(selected_frames, colors):
            curve_axis.plot(
                analysis.distance_A,
                analysis.g_r[frame_index, phase_index],
                color=color,
                label=f"{analysis.time_ps[frame_index]:.2f} ps",
            )
        curve_axis.set_title(phase_name.replace("_", " "))
        curve_axis.set_ylabel("center-conditioned g(r)")
        curve_axis.legend()

        heatmap_axis = axes[1, phase_index]
        heatmap = heatmap_axis.imshow(
            analysis.g_r[:, phase_index],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                0.0,
                float(analysis.distance_A[-1]),
                float(analysis.time_ps[0]),
                float(analysis.time_ps[-1]),
            ),
            vmin=0.0,
            vmax=heatmap_limit,
            cmap="magma",
        )
        heatmap_axis.set(xlabel="r (A)", ylabel="time (ps)")
        figure.colorbar(heatmap, ax=heatmap_axis, label="g(r)")

    figure.suptitle(
        f"{branch.name}: RDF evolution by initial phase provenance at "
        f"{branch.temperature_K:.0f} K\n"
        "Central atoms follow their prepared labels; neighbors include every Al atom"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_phase_rdf_overview(path: Path, branch_images: dict[str, Path]) -> None:
    figure, axes = plt.subplots(
        1,
        len(branch_images),
        figsize=(16.0, 6.0),
        constrained_layout=True,
    )
    for axis, image_path in zip(np.atleast_1d(axes), branch_images.values()):
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
    figure.suptitle("MACE direct-coexistence RDF evolution by initial phase provenance")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_structure_slice_overview(path: Path, branch_images: dict[str, Path]) -> None:
    figure, axes = plt.subplots(
        len(branch_images),
        1,
        figsize=(17.0, 6.0 * len(branch_images)),
        constrained_layout=True,
    )
    for axis, (branch_name, image_path) in zip(np.atleast_1d(axes), branch_images.items()):
        axis.imshow(plt.imread(image_path))
        axis.set_title(branch_name)
        axis.axis("off")
    figure.suptitle("MACE direct-coexistence atomic structure slices")
    figure.savefig(path, dpi=160)
    plt.close(figure)
