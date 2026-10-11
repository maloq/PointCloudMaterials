"""Ni GPU EAM forces and scalar virial inside native LAMMPS isotropic NPT.

The isotropic barostat uses the virial trace only. This adapter deliberately
does not expose physical normal or shear stresses, or anisotropic cell control.
"""
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from src.experiment_runner.artifacts import write_json
from .potential import FastNi


class IsotropicNi(FastNi):
    """Reuse neighbors under a checked affine cell deformation.

    If s is the smallest box stretch and u the largest displacement from the
    affinely transformed build positions, any omitted pair remains at least
    s*(rc+skin)-2*u apart. Rebuild before this lower bound reaches rc. Native
    atom-tag reordering also invalidates the list. This checks every force call.
    """
    def support(self,q):
        x=q.detach().to(self.dtype).contiguous()
        if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite Ni coordinates')
        rebuild=self.reference is None or not np.array_equal(self.reference_tags,self.current_tags)
        if not rebuild:
            ratio=self.box/self.reference_box
            displacement=x-self.reference*ratio
            displacement-=self.box*torch.round(displacement/self.box)
            bound=ratio.min()*self.cutoff-2*displacement.square().sum(-1).max().sqrt()
            rebuild=bool(bound<=self.data.cutoff+1e-4)
        if rebuild:
            self._build(x);self.reference_box=self.box.clone();self.reference_tags=self.current_tags.copy()
        return x


class Callback:
    def __init__(self,native,potential,directory):
        self.native=native;self.directory=Path(directory);self.atoms=int(native.get_natoms())
        low,high,xy,yz,xz,periodic,_=native.extract_box()
        if (xy,yz,xz)!=(0.,0.,0.) or list(periodic)!=[1,1,1]:
            raise ValueError('Ni adapter requires an orthorhombic periodic cell')
        self.model=IsotropicNi(potential,self.atoms,box=np.subtract(high,low),dtype=torch.float32)
        self.calls=0

    def __call__(self,caller,step,nlocal,tags,positions,forces):
        try:
            if nlocal!=self.atoms:raise ValueError('Exactly one LAMMPS rank is required')
            low,high,xy,yz,xz,_,_=self.native.extract_box()
            if (xy,yz,xz)!=(0.,0.,0.):raise ValueError('Cell tilt is unsupported')
            box=np.subtract(high,low)
            q=torch.tensor(np.remainder(positions-low,box),device='cuda',dtype=torch.float64)
            # Affine displacement/box bound and actual tags govern list reuse.
            # Box and neighbor pointers remain stable for force graph replay.
            m=self.model;m.box.copy_(torch.as_tensor(box,device=m.device,dtype=m.dtype))
            m.current_tags=np.asarray(tags)
            energy,force,virial,_=m.evaluate(q)
            forces[:]=force.cpu().numpy()
            self.native.fix_external_set_energy_global('gpu_ni',float(energy))
            self.native.fix_external_set_virial_global('gpu_ni',[float(virial)/3]*3+[0.,0.,0.])
            self.calls+=1
        except BaseException:
            error=traceback.format_exc()
            write_json(self.directory/'callback-failure.json',dict(step=int(step),error=error))
            print(error,flush=True)
            # ctypes callbacks otherwise swallow Python exceptions and continue MD.
            os._exit(72)


def install(native,potential,directory):
    native.commands_string('pair_style zero 1.0\npair_coeff * *\nfix gpu_ni all external pf/callback 1 1\nfix_modify gpu_ni energy yes virial yes')
    callback=Callback(native,potential,directory)
    native.set_fix_external_callback('gpu_ni',callback)
    return callback
