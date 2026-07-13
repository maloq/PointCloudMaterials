from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.constraints import FixAtoms, FixCom
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from .config import GeneratorConfig


@dataclass(frozen=True)
class ThermodynamicTrace:
    step: np.ndarray
    temperature_K: np.ndarray
    pressure_GPa: np.ndarray
    volume_A3: np.ndarray
    potential_energy_eV_per_atom: np.ndarray
    positions_A: np.ndarray
    cell_vectors_A: np.ndarray

@dataclass(frozen=True)
class SimulatedSystems:
    solid: Atoms
    liquid: Atoms
    interface: Atoms
    solid_trace: ThermodynamicTrace
    liquid_trace: ThermodynamicTrace
    interface_trace: ThermodynamicTrace
    liquid_slab_bounds_fractional: tuple[float, float]


class _TraceRecorder:
    def __init__(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.step: list[int] = []
        self.temperature_K: list[float] = []
        self.pressure_GPa: list[float] = []
        self.volume_A3: list[float] = []
        self.potential_energy_eV_per_atom: list[float] = []
        self.positions_A: list[np.ndarray] = []
        self.cell_vectors_A: list[np.ndarray] = []

    def sample(self, step: int) -> None:
        atom_count = len(self.atoms)
        self.step.append(step)
        self.temperature_K.append(float(self.atoms.get_temperature()))
        self.pressure_GPa.append(
            float(
                -np.trace(
                    self.atoms.get_stress(voigt=False, include_ideal_gas=True)
                )
                / 3.0
                / units.GPa
            )
        )
        self.volume_A3.append(float(self.atoms.get_volume()))
        self.potential_energy_eV_per_atom.append(
            float(self.atoms.get_potential_energy() / atom_count)
        )
        self.positions_A.append(
            np.asarray(self.atoms.get_positions(wrap=True), dtype=np.float32)
        )
        self.cell_vectors_A.append(np.asarray(self.atoms.cell, dtype=np.float64))

    def finish(self, stage: str) -> ThermodynamicTrace:
        if not self.volume_A3:
            self.sample(0)
        arrays = ThermodynamicTrace(
            step=np.asarray(self.step, dtype=np.int64),
            temperature_K=np.asarray(self.temperature_K, dtype=np.float64),
            pressure_GPa=np.asarray(self.pressure_GPa, dtype=np.float64),
            volume_A3=np.asarray(self.volume_A3, dtype=np.float64),
            potential_energy_eV_per_atom=np.asarray(
                self.potential_energy_eV_per_atom, dtype=np.float64
            ),
            positions_A=np.stack(self.positions_A),
            cell_vectors_A=np.stack(self.cell_vectors_A),
        )
        for name, values in (
            ("step", arrays.step),
            ("temperature_K", arrays.temperature_K),
            ("pressure_GPa", arrays.pressure_GPa),
            ("volume_A3", arrays.volume_A3),
            ("potential_energy_eV_per_atom", arrays.potential_energy_eV_per_atom),
            ("positions_A", arrays.positions_A),
            ("cell_vectors_A", arrays.cell_vectors_A),
        ):
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Non-finite thermodynamic values during {stage}: field={name}, "
                    f"values={values.tolist()}."
                )
        return arrays


def build_initial_solid(config: GeneratorConfig) -> Atoms:
    unit_cell = bulk(
        config.system.chemical_symbol,
        crystalstructure=config.system.crystal_structure,
        a=config.system.initial_lattice_constant_A,
        cubic=True,
    )
    atoms = unit_cell.repeat(config.system.repetitions)
    atoms.pbc = True
    return atoms


def _initialize_velocities(atoms: Atoms, temperature_K: float, rng: np.random.Generator) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms, preserve_temperature=True)


def _run_npt(
    atoms: Atoms,
    *,
    config: GeneratorConfig,
    temperature_K: float,
    steps: int,
    stage: str,
    initialize_velocities: bool,
    rng: np.random.Generator,
    progress: Callable[[str], None],
) -> ThermodynamicTrace:
    if initialize_velocities:
        _initialize_velocities(atoms, temperature_K, rng)
    recorder = _TraceRecorder(atoms)
    if steps:
        dynamics = IsotropicMTKNPT(
            atoms,
            timestep=config.dynamics.timestep_fs * units.fs,
            temperature_K=temperature_K,
            pressure_au=config.dynamics.pressure_GPa * units.GPa,
            tdamp=config.dynamics.thermostat_time_fs * units.fs,
            pdamp=config.dynamics.barostat_time_fs * units.fs,
        )
        dynamics.attach(
            lambda: recorder.sample(dynamics.nsteps),
            interval=config.dynamics.sample_interval,
        )
        progress(
            f"{stage}: {steps} NPT steps at {temperature_K:.1f} K and "
            f"{config.dynamics.pressure_GPa:.4f} GPa"
        )
        dynamics.run(steps)
        if steps % config.dynamics.sample_interval:
            recorder.sample(steps)
    return recorder.finish(stage)


def _run_nvt(
    atoms: Atoms,
    *,
    config: GeneratorConfig,
    temperature_K: float,
    steps: int,
    stage: str,
    initialize_velocities: bool,
    rng: np.random.Generator,
    progress: Callable[[str], None],
) -> ThermodynamicTrace:
    if initialize_velocities:
        _initialize_velocities(atoms, temperature_K, rng)
    recorder = _TraceRecorder(atoms)
    original_constraints = list(atoms.constraints)
    atoms.set_constraint([*original_constraints, FixCom()])
    try:
        if steps:
            dynamics = Langevin(
                atoms,
                timestep=config.dynamics.timestep_fs * units.fs,
                temperature_K=temperature_K,
                friction=1.0 / (config.dynamics.thermostat_time_fs * units.fs),
                fixcm=False,
                rng=rng,
            )
            dynamics.attach(
                lambda: recorder.sample(dynamics.nsteps),
                interval=config.dynamics.sample_interval,
            )
            progress(f"{stage}: {steps} NVT steps at {temperature_K:.1f} K")
            dynamics.run(steps)
            if steps % config.dynamics.sample_interval:
                recorder.sample(steps)
    finally:
        atoms.set_constraint(original_constraints)
    return recorder.finish(stage)


def _quench(
    atoms: Atoms,
    *,
    config: GeneratorConfig,
    rng: np.random.Generator,
    progress: Callable[[str], None],
    stage_prefix: str,
    use_npt: bool,
) -> None:
    total_steps = config.dynamics.quench_steps
    if total_steps == 0:
        return
    stage_count = min(config.dynamics.quench_stages, total_steps)
    base_steps, remainder = divmod(total_steps, stage_count)
    temperatures = np.linspace(
        config.dynamics.melt_temperature_K,
        config.dynamics.target_temperature_K,
        stage_count + 1,
        dtype=np.float64,
    )[1:]
    runner = _run_npt if use_npt else _run_nvt
    for index, temperature_K in enumerate(temperatures):
        steps = base_steps + (1 if index < remainder else 0)
        runner(
            atoms,
            config=config,
            temperature_K=float(temperature_K),
            steps=steps,
            stage=f"{stage_prefix}.quench_{index + 1:02d}",
            initialize_velocities=False,
            rng=rng,
            progress=progress,
        )


def _liquid_slab_mask(atoms: Atoms, fraction: float) -> tuple[np.ndarray, tuple[float, float]]:
    lower = 0.5 - fraction / 2.0
    upper = 0.5 + fraction / 2.0
    scaled_z = atoms.get_scaled_positions(wrap=True)[:, 2]
    return (scaled_z >= lower) & (scaled_z < upper), (lower, upper)


def simulate_systems(
    config: GeneratorConfig,
    calculator: object,
    *,
    random_seed: int,
    checkpoint_prefix: str,
    progress: Callable[[str], None],
) -> SimulatedSystems:
    from .checkpoints import CheckpointStore

    checkpoints = CheckpointStore(config)
    seed_sequence = np.random.SeedSequence(random_seed)
    solid_seed, liquid_seed, interface_seed = seed_sequence.spawn(3)
    solid_rng = np.random.default_rng(solid_seed)
    liquid_rng = np.random.default_rng(liquid_seed)
    interface_rng = np.random.default_rng(interface_seed)

    solid_stage = f"{checkpoint_prefix}.solid"
    liquid_stage = f"{checkpoint_prefix}.liquid"
    interface_stage = f"{checkpoint_prefix}.interface"

    solid_checkpoint = checkpoints.load(solid_stage)
    if solid_checkpoint is None:
        solid = build_initial_solid(config)
        solid.calc = calculator
        solid_trace = _run_npt(
            solid,
            config=config,
            temperature_K=config.dynamics.target_temperature_K,
            steps=config.dynamics.solid_equilibration_steps,
            stage="solid.equilibrate",
            initialize_velocities=True,
            rng=solid_rng,
            progress=progress,
        )
        solid.wrap()
        checkpoints.save(solid_stage, solid, solid_trace)
    else:
        progress(f"{solid_stage}: loaded checkpoint from {checkpoints.directory}")
        solid, solid_trace = solid_checkpoint.atoms, solid_checkpoint.trace
        solid.calc = calculator

    liquid_checkpoint = checkpoints.load(liquid_stage)
    if liquid_checkpoint is None:
        liquid = solid.copy()
        liquid.calc = calculator
        _run_npt(
            liquid,
            config=config,
            temperature_K=config.dynamics.melt_temperature_K,
            steps=config.dynamics.melt_steps,
            stage="liquid.melt",
            initialize_velocities=True,
            rng=liquid_rng,
            progress=progress,
        )
        _quench(
            liquid,
            config=config,
            rng=liquid_rng,
            progress=progress,
            stage_prefix="liquid",
            use_npt=True,
        )
        liquid_trace = _run_npt(
            liquid,
            config=config,
            temperature_K=config.dynamics.target_temperature_K,
            steps=config.dynamics.target_equilibration_steps,
            stage="liquid.equilibrate",
            initialize_velocities=False,
            rng=liquid_rng,
            progress=progress,
        )
        liquid.wrap()
        checkpoints.save(liquid_stage, liquid, liquid_trace)
    else:
        progress(f"{liquid_stage}: loaded checkpoint from {checkpoints.directory}")
        liquid, liquid_trace = liquid_checkpoint.atoms, liquid_checkpoint.trace
        liquid.calc = calculator

    interface_checkpoint = checkpoints.load(interface_stage)
    if interface_checkpoint is None:
        interface = solid.copy()
        interface.calc = calculator
        liquid_fraction = config.system.liquid_slab_fraction
        mixture_volume_A3 = (
            (1.0 - liquid_fraction) * solid.get_volume()
            + liquid_fraction * liquid.get_volume()
        )
        volume_scale = (mixture_volume_A3 / interface.get_volume()) ** (1.0 / 3.0)
        interface.set_cell(np.asarray(interface.cell) * volume_scale, scale_atoms=True)
        liquid_mask, slab_bounds = _liquid_slab_mask(
            interface, liquid_fraction
        )
        interface.set_constraint(FixAtoms(mask=~liquid_mask))
        _run_nvt(
            interface,
            config=config,
            temperature_K=config.dynamics.melt_temperature_K,
            steps=config.dynamics.melt_steps,
            stage="interface.melt_slab",
            initialize_velocities=True,
            rng=interface_rng,
            progress=progress,
        )
        interface.set_constraint()
        # The 650 K solid-liquid boundary is a moving growth front, not an equilibrium
        # interface. Resetting velocities implements the reference trajectory's rapid quench.
        _initialize_velocities(
            interface, config.dynamics.target_temperature_K, interface_rng
        )
        interface_trace = _run_nvt(
            interface,
            config=config,
            temperature_K=config.dynamics.target_temperature_K,
            steps=config.dynamics.interface_evolution_steps,
            stage="interface.evolve_after_rapid_quench",
            initialize_velocities=False,
            rng=interface_rng,
            progress=progress,
        )
        interface.wrap()
        checkpoints.save(
            interface_stage,
            interface,
            interface_trace,
            metadata={"slab_bounds_fractional": list(slab_bounds)},
        )
    else:
        progress(f"{interface_stage}: loaded checkpoint from {checkpoints.directory}")
        interface, interface_trace = interface_checkpoint.atoms, interface_checkpoint.trace
        interface.calc = calculator
        slab_values = interface_checkpoint.metadata["slab_bounds_fractional"]
        if not isinstance(slab_values, list) or len(slab_values) != 2:
            raise ValueError(
                "Interface checkpoint slab_bounds_fractional must be a two-item list, "
                f"got {slab_values!r}."
            )
        slab_bounds = (float(slab_values[0]), float(slab_values[1]))

    return SimulatedSystems(
        solid=solid,
        liquid=liquid,
        interface=interface,
        solid_trace=solid_trace,
        liquid_trace=liquid_trace,
        interface_trace=interface_trace,
        liquid_slab_bounds_fractional=slab_bounds,
    )
