"""Original LAMMPS MEAM forces in a fixed orthorhombic periodic cell."""
import ctypes
import numpy as np
from lammps import lammps

from src.research.response_force_engines.engines import checked_files


class PeriodicLAMMPS:
    def __init__(self, files, q, box, comm=None):
        self.n=len(q)
        self.box=np.asarray(box,dtype=np.float64)
        self.lmp=lammps(cmdargs=['-screen','none','-log','none'],comm=comm)
        self.lmp.commands_list([
            'units metal','atom_style atomic','boundary p p p',
            'atom_modify map array sort 0 0.0',
            f'region domain block 0 {box[0]} 0 {box[1]} 0 {box[2]} units box',
            'create_box 1 domain','mass 1 26.9815385'])
        points=np.asarray(q,dtype=np.float64)%self.box
        self.lmp.create_atoms(self.n,list(range(1,self.n+1)),[1]*self.n,points.ravel().tolist())
        if int(self.lmp.get_natoms())!=self.n:
            raise ValueError('LAMMPS periodic atom count differs')
        self.lmp.commands_list(['pair_style meam',f'pair_coeff * * {files["library"]} Al {files["parameters"]} Al',
            'neighbor 0.6 bin','neigh_modify every 1 delay 0 check no',
            'thermo_style custom step atoms pe','thermo_modify lost error'])

    def __call__(self,q):
        points=np.ascontiguousarray(q%self.box,dtype=np.float64)
        self.lmp.scatter_atoms('x',1,3,points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command('run 0 pre yes post no')
        if int(self.lmp.get_natoms())!=self.n:
            raise ValueError('LAMMPS lost an atom')
        forces=np.ctypeslib.as_array(self.lmp.gather_atoms('f',1,3),shape=(3*self.n,)).reshape(self.n,3).copy()
        return float(self.lmp.get_thermo('pe')),forces

    def close(self):
        self.lmp.close()
