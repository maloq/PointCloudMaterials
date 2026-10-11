"""Benchmark minimizers against the same existing FastNi force callback."""
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize.precon import Exp, PreconFIRE, PreconLBFGS

from src.experiment_runner.artifacts import json_digest, write_json
from src.simulation.eam_gpu import relaxation_fast
from src.simulation.relaxation import _write_input
from src.research.ni_relaxation_repeatability.backend import CanonicalCallback


class TimedForces:
    """Count actual energy/force callbacks, including line-search evaluations."""
    def compute(self, tags, positions):
        begin = time.perf_counter()
        result = super().compute(tags, positions)
        self.callback_seconds = getattr(self, 'callback_seconds', 0.) + time.perf_counter() - begin
        return result


class SortedForces(TimedForces, CanonicalCallback):
    pass


class RawForces(TimedForces, relaxation_fast.FixedBoxCallback):
    pass


class GPUCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.tags = np.arange(1, callback.atoms + 1, dtype=np.int32)

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        f, e, _ = self.callback.compute(self.tags, atoms.positions)
        self.results = dict(energy=e, free_energy=e, forces=f.copy())


class ComponentNorm:
    """Match production's max force component, rather than ASE's atom norm."""
    def gradient_converged(self, gradient):
        return np.abs(gradient).max() < self.fmax


class ComponentLBFGS(ComponentNorm, PreconLBFGS):
    pass


class ComponentFIRE(ComponentNorm, PreconFIRE):
    pass


def warmup(q, box, potential, directory):
    native = SimpleNamespace(get_natoms=lambda: len(q),
        extract_box=lambda: (np.zeros(3), box, 0., 0., 0., [1, 1, 1], 0))
    callback = SortedForces(native, potential, directory)
    callback.compute(np.arange(1, len(q)+1, dtype=np.int32), q)
    del callback
    gc.collect()


def native_fire(q, box, potential, profile, directory):
    directory = Path(directory)
    n = len(q)
    trajectory = SimpleNamespace(positions=q[None], box_low=np.zeros((1, 3)),
        box_high=box[None], atom_ids=np.arange(1, n+1), atom_types=np.ones(n, dtype=int))
    _, _, data = _write_input(trajectory, 0, directory, {})
    text = f'''units metal
atom_style atomic
boundary p p p
read_data {data}
mass 1 58.71
pair_style eam/fs
pair_coeff * * {potential} Ni
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 100
thermo_style custom step pe fmax fnorm
thermo_modify flush yes format float %.16g
timestep {profile.get('timestep_ps', .001)}
min_style fire
min_modify norm inf {profile.get('min_modify', '')}
minimize 0.0 {profile['tolerance']} 20000 100000
variable final_force equal fmax
variable final_energy equal pe
print "FINAL_FORCE ${{final_force}}"
print "FINAL_ENERGY ${{final_energy}}"
write_dump all custom relaxed.dump id type x y z modify sort id format line "%d %d %.17g %.17g %.17g"
'''
    (directory/'in.lammps').write_text(text)
    callbacks = []
    cls = RawForces if profile['name'] == 'raw_fire' else SortedForces
    def factory(*args):
        callback = cls(*args)
        callbacks.append(callback)
        return callback
    previous = relaxation_fast.FixedBoxCallback
    relaxation_fast.FixedBoxCallback = factory
    try:
        begin = time.perf_counter()
        relaxation_fast.execute(directory/'in.lammps', directory/'log.lammps', potential,
            expected_sha256=json_digest(relaxation_fast.implementation()))
        elapsed = time.perf_counter() - begin
    finally:
        relaxation_fast.FixedBoxCallback = previous
    log = (directory/'log.lammps').read_text().splitlines()
    fmax = float(next(s.split()[1] for s in reversed(log) if s.startswith('FINAL_FORCE ')))
    energy = float(next(s.split()[1] for s in reversed(log) if s.startswith('FINAL_ENERGY ')))
    xyz = np.loadtxt(directory/'relaxed.dump', skiprows=9)[:, 2:5]
    stats = dict(solver_seconds=elapsed, force_calls=callbacks[0].calls,
        callback_seconds=callbacks[0].callback_seconds, fmax_eV_A=fmax, energy_eV=energy,
        converged=fmax < profile['tolerance'])
    del callbacks
    return xyz, stats


def ase_minimize(q, box, potential, profile, directory, time_limit):
    native = SimpleNamespace(get_natoms=lambda: len(q),
        extract_box=lambda: (np.zeros(3), box, 0., 0., 0., [1, 1, 1], 0))
    callback = SortedForces(native, potential, directory)
    atoms = Atoms('Ni'*len(q), positions=q.copy(), cell=box, pbc=True)
    atoms.calc = GPUCalculator(callback)
    begin = time.perf_counter()
    precon = None if profile['name'] == 'lbfgs' else Exp(A=3, solver='pyamg')
    if profile['name'] == 'precon_fire':
        opt = ComponentFIRE(atoms, precon=precon, use_armijo=True,
            logfile=str(directory/'optimizer.log'), maxmove=.1)
    else:
        opt = ComponentLBFGS(atoms, precon=precon, use_armijo=True, maxstep=.1,
            logfile=str(directory/'optimizer.log'))
    def progress():
        forces = atoms.get_forces()
        write_json(directory/'optimizer-progress.json', dict(iteration=opt.nsteps,
            force_calls=callback.calls, seconds=time.perf_counter()-begin,
            fmax_eV_A=float(abs(forces).max()), energy_eV=atoms.get_potential_energy()))
        if time.perf_counter()-begin > time_limit:
            raise TimeoutError('Declared per-case optimizer time budget exhausted')
    opt.attach(progress, interval=10)
    failure = None
    try:
        opt.run(fmax=profile['tolerance'], steps=20000)
    except Exception as error:
        failure = repr(error)
    elapsed = time.perf_counter()-begin
    forces = atoms.get_forces()
    fmax = float(abs(forces).max())
    stats = dict(solver_seconds=elapsed, force_calls=callback.calls,
        callback_seconds=callback.callback_seconds, fmax_eV_A=fmax,
        energy_eV=atoms.get_potential_energy(), iterations=opt.nsteps,
        converged=fmax < profile['tolerance'], failure=failure)
    return atoms.positions.copy(), stats
