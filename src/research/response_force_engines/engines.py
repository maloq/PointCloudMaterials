"""Persistent, single-process force engines. Positions are in Å, forces eV/Å."""
import ctypes
import numpy as np
from ase import Atoms

from src.project_runtime.paths import resolve_path
from src.research.local_response.common import sha


def checked_files(spec):
    files = {name: resolve_path(item['path']) for name, item in spec['files'].items()}
    for name, path in files.items():
        if sha(path) != spec['files'][name]['sha256']:
            raise ValueError(f'Changed {name} potential: {path}')
    return files


class UNEP:
    def __init__(self, spec, q, half_box):
        from calorine.calculators import CPUNEP
        self.half_box = half_box
        self.cutoff = 6.0  # pinned UNEP-v1-main header: cutoff 6 5
        self.atoms = Atoms('Al' * len(q), positions=q + half_box,
                           cell=[2 * half_box] * 3, pbc=True)
        self.calculator = CPUNEP(str(checked_files(spec)['model']))

    def __call__(self, q):
        # CPUNEP ignores ASE pbc=False. This explicit vacuum makes its periodic
        # evaluation equivalent to an isolated cluster, also during motion.
        if np.abs(q).max() >= self.half_box - self.cutoff:
            raise ValueError('UNEP vacuum margin exhausted')
        self.atoms.positions[:] = q + self.half_box
        self.calculator.calculate(self.atoms, properties=['energy', 'forces'])
        return float(self.calculator.results['energy']), self.calculator.results['forces'].copy()

    def close(self):
        pass


class MEAM:
    def __init__(self, spec, q, half_box):
        from lammps import lammps
        files = checked_files(spec)
        self.half_box, self.n = half_box, len(q)
        self.lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        self.lmp.commands_list([
            'units metal', 'atom_style atomic', 'boundary f f f',
            'atom_modify map array sort 0 0.0',
            f'region domain block {-half_box} {half_box} {-half_box} {half_box} {-half_box} {half_box} units box',
            'create_box 1 domain', 'mass 1 26.9815385'])
        created = self.lmp.create_atoms(self.n, list(range(1, self.n + 1)),
                                       [1] * self.n, q.ravel().tolist())
        if created != self.n:
            raise ValueError(f'LAMMPS created {created}/{self.n} atoms')
        self.lmp.commands_list([
            'pair_style meam', f'pair_coeff * * {files["library"]} Al {files["parameters"]} Al',
            'neighbor 0.6 bin', 'neigh_modify every 1 delay 0 check no',
            'thermo_style custom step atoms pe', 'thermo_modify lost error'])

    def __call__(self, q):
        if np.abs(q).max() >= self.half_box - 1:
            raise ValueError('LAMMPS atom reached the open box boundary')
        x = np.ascontiguousarray(q, dtype=np.float64)
        self.lmp.scatter_atoms('x', 1, 3, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # Always rebuild neighbors for externally supplied coordinates. No LAMMPS
        # time integrator is installed: identical BAOAB runs outside both engines.
        self.lmp.command('run 0 pre yes post no')
        if int(self.lmp.get_natoms()) != self.n:
            raise ValueError('LAMMPS lost atoms')
        force = np.ctypeslib.as_array(self.lmp.gather_atoms('f', 1, 3),
                                     shape=(self.n * 3,)).reshape(self.n, 3).copy()
        return float(self.lmp.get_thermo('pe')), force

    def close(self):
        self.lmp.close()


def make_engine(spec, q, half_box):
    return {'unep': UNEP, 'meam': MEAM}[spec['engine']](spec, q, half_box)
