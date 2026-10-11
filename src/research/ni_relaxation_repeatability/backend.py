"""Private callback variants; production FastNi/FIRE files remain unchanged."""
import argparse
import functools
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from src.experiment_runner.artifacts import file_hash, json_digest, write_json
from src.simulation.eam_gpu import relaxation_fast as original
from src.simulation.eam_gpu.potential import FastNi


@functools.lru_cache(maxsize=1)
def extension():
    path=Path(__file__).with_name('sort_neighbors.cu')
    return load('pcm_ni_canonical_'+file_hash(path)[:12], [str(path)],
                extra_cuda_cflags=['-O3','-lineinfo'], verbose=False)


def implementation():
    return dict(original=original.implementation(), files={
        name:file_hash(Path(__file__).with_name(name))
        for name in ['backend.py','sort_neighbors.cu']})


class CanonicalCallback(original.FixedBoxCallback):
    def __init__(self,native,potential,directory,*,dtype=torch.float32):
        super().__init__(native,potential,directory)
        if dtype==torch.float64:
            self.model=FastNi(potential,self.atoms,box=self.box,dtype=dtype)
            for name in ['x','reference','origin','fp','empty','f']:
                setattr(self,name,getattr(self,name).to(dtype))
        self.sorter=extension()
        self.device_tags=torch.empty(self.atoms,device=self.model.device,dtype=torch.int32)

    def compute(self,tags,positions):
        if self.tags is None or not np.array_equal(self.tags,tags):
            self.device_tags.copy_(torch.from_numpy(np.asarray(tags,dtype=np.int32).copy()))
        return super().compute(tags,positions)

    def _evaluate(self):
        m=self.model
        self.x.copy_(self.device_in[:-1].view(self.atoms,3))
        self.error.zero_()
        d=self.x-self.reference;d-=m.box*torch.round(d/m.box)
        bound=m.cutoff-2*d.square().sum(-1).max().sqrt()
        self.rebuild.copy_((bound<=m.data.cutoff+1e-4)|(self.device_in[-1]!=0))
        m.native.dynamic_neighbors(self.x,self.reference,m.box,self.origin,
            *self.grid,m.atoms,m.cutoff,m.skin,self.bin_counts,self.bins,
            self.atom_cells,self.neighbors,self.count,self.rebuild,self.rebuild_count,self.error)
        self.sorter.sort_neighbors(self.neighbors,self.count,self.device_tags,self.rebuild,self.error)
        m.native.evaluate(self.x,self.empty,m.box,True,self.neighbors,self.count,
            m.rho,m.embed,m.pair,m.data.dr,m.data.drho,m.data.cutoff,
            self.fp,self.empty,self.e,self.w,self.f,self.empty,self.error)
        self.packed[:-4].copy_(self.f.flatten())
        self.packed[-4].copy_(self.e.sum());self.packed[-3].copy_(self.w.sum())
        self.packed[-2:].copy_(torch.cat((self.error,self.rebuild_count)))


def execute(directory,settings,mode):
    directory=Path(directory).resolve()
    actual=implementation()
    command=settings['lammps_command']
    expected=command[command.index('--implementation-sha256')+1]
    if json_digest(actual)!=expected:raise ValueError('Repeatability implementation changed')
    potential=settings['potential_files'][0]
    # This substitution exists only inside this diagnostic process. The frozen
    # production workers and shared source files never see it.
    saved=original.FixedBoxCallback
    try:
        if mode!='raw32':
            original.FixedBoxCallback=functools.partial(CanonicalCallback,
                dtype=torch.float64 if mode=='canonical64' else torch.float32)
        result=original.execute(str(directory/'in.lammps'),str(directory/'log.lammps'),
            potential,expected_sha256=json_digest(original.implementation()))
    finally:
        original.FixedBoxCallback=saved
    result.update(diagnostic_mode=mode,diagnostic_implementation=actual,
        force_dtype='float64' if mode=='canonical64' else 'float32',
        neighbor_order='ascending atom ID' if mode!='raw32' else 'production bin insertion order')
    write_json(directory/'in.gpu.json',result)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['raw32','canonical32','canonical64'],required=True)
    ap.add_argument('--implementation-sha256',required=True)
    ap.add_argument('--potential',required=True)
    ap.add_argument('-in',dest='input',required=True)
    ap.add_argument('-log',default='log.lammps');a=ap.parse_args()
    if Path(a.input).name!='in.lammps' or Path(a.log).name!='log.lammps':
        raise ValueError('This diagnostic adapter uses in.lammps and log.lammps')
    execute(Path(a.input).resolve().parent,dict(potential_files=[a.potential],
        lammps_command=['--implementation-sha256',a.implementation_sha256]),a.mode)


if __name__=='__main__':main()
