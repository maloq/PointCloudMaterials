"""LAMMPS periodic-force checks and full-cell long-time reference responses."""
import ctypes
import gc
import time
import numpy as np
import torch
from lammps import lammps
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.local_response.gates import error
from src.research.response_performance.oracle import force_and_tangent
from .common import FAMILY,root,read,save,sha,write_json
from .data import prepare,initial
from .oracle import Oracle


class PeriodicReference:
    def __init__(self,c,q,box):
        self.box=box.cpu().numpy();self.n=len(q)
        self.lmp=lammps(cmdargs=['-screen','none','-log','none'])
        self.lmp.commands_list(['units metal','atom_style atomic','boundary p p p','atom_modify map array sort 0 0.0',
            f'region domain block 0 {self.box[0]} 0 {self.box[1]} 0 {self.box[2]} units box',
            'create_box 1 domain','mass 1 26.9815385'])
        positions=np.remainder(q.cpu().numpy(),self.box)
        count=self.lmp.create_atoms(self.n,list(range(1,self.n+1)),[1]*self.n,positions.ravel().tolist())
        if count!=self.n:raise ValueError('Periodic LAMMPS reference lost initial atoms')
        library=resolve_path(c['meam_files']['library']);parameters=resolve_path(c['meam_files']['parameters'])
        self.lmp.commands_list(['pair_style meam',f'pair_coeff * * {library} Al {parameters} Al',
            'neighbor 0.6 bin','neigh_modify every 1 delay 0 check no','thermo_style custom step atoms pe','thermo_modify lost error'])

    def __call__(self,q):
        x=np.ascontiguousarray(np.remainder(q,self.box),dtype=np.float64)
        self.lmp.scatter_atoms('x',1,3,x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.lmp.command('run 0 pre yes post no')
        f=np.ctypeslib.as_array(self.lmp.gather_atoms('f',1,3),shape=(3*self.n,)).reshape(-1,3).copy()
        return float(self.lmp.get_thermo('pe')),torch.from_numpy(f)


def gate(c):
    from .collection import normalizer,publish
    out=root(c)/'technical/gates';out.mkdir(parents=True,exist_ok=True)
    identity=read(root(c)/'technical/binding.json')['identity']
    if (out/'complete.json').exists():
        if read(out/'complete.json')['identity']!=identity:raise ValueError('Periodic gate identity changed')
        return
    states=prepare(c);norm=normalizer(c);rows=[];boundary=[]
    reference=resolve_path(c['open_reference'])
    if sha(reference/'technical/binding.json')!=c['open_reference_binding_sha256']:
        raise ValueError('Open-environment reference changed')
    old=read(reference/'technical/binding.json')['config']
    for key in ['horizons_fs','timestep_fs','friction_fs','temperature_K','velocity_seed','noise_seed',
                'feature_seed','velocity_scale_A_fs','geometry_reference_identity']:
        if c[key]!=old[key]:raise ValueError('Open/periodic physics or target map mismatch: '+key)
    for index in c['gate_parents']:
        state=states[index];q,v,basis,ids=initial(c,state,0,None)
        if state['role']!='train':raise ValueError('Periodic gate must use train sources')
        oracle=Oracle(c,state,ids,norm,periodic=True)
        checks=out/f'force-parent{index}.json'
        if checks.exists():row=read(checks)
        else:
            cpu=PeriodicReference(c,q,state['box'])
            try:
                ce,cf=cpu(q.cpu().numpy());x=q.flatten()[None]
                energy=float(oracle.potential(x));force,tangent=force_and_tangent(oracle.potential,x,basis[0].flatten(0,1)[None,:,:2])
                eps=.0001;fd=[]
                for j in range(2):
                    _,plus=cpu((q+eps*basis[0,...,j]).cpu().numpy())
                    _,minus=cpu((q-eps*basis[0,...,j]).cpu().numpy());fd.append((plus-minus)/(2*eps))
                fe=error(force.flatten().cpu(),cf.flatten());he=error(tangent[0].cpu(),torch.stack(fd,-1).flatten(0,1))
                row=dict(parent=index,atoms=len(q),energy_per_atom_error=abs(energy-ce)/len(q),
                    force_rms=fe['rms'],force_response_relative=he['relative'])
                if row['energy_per_atom_error']>1e-8 or row['force_rms']>1e-8 or row['force_response_relative']>.002:
                    raise ValueError(f'Periodic MEAM disagrees with LAMMPS: {row}')
                write_json(checks,row);print('Periodic force gate',row,flush=True)
            finally:cpu.lmp.close()
        rows.append(row)
        seed=[c['noise_seed']-100-index]
        # Integration and velocity tangents already passed1ps open-cell FD. Check
        # the new periodic force/feature geometry in a complete20fs trajectory.
        if index==c['gate_parents'][0] and not (out/'periodic-path-audit.json').exists():
            short=dict(c,horizons_fs=[20]);probe=Oracle(short,state,ids,norm,periodic=True)
            value,response=probe.simulate(q,v,basis,seed);audit=[]
            for j in [0,2]:
                eps=.0001
                plus,_=probe.simulate(q+eps*basis[0,...,j],v+eps*basis[1,...,j],basis[...,:0],seed)
                minus,_=probe.simulate(q-eps*basis[0,...,j],v-eps*basis[1,...,j],basis[...,:0],seed)
                e=error((plus-minus)/(2*eps),response[...,j]);audit.append(dict(direction=j,**e))
                if e['relative']>.02:raise ValueError(f'Periodic path derivative failed: {audit}')
            again,_=probe.simulate(q,v,basis[...,:0],seed)
            if error(again,value)['maximum_absolute']>1e-9:raise ValueError('Periodic value reuse audit failed')
            write_json(out/'periodic-path-audit.json',dict(checks=audit,value_reuse=error(again,value)))
            del probe;gc.collect();torch.cuda.empty_cache()
        path=out/f'periodic-parent{index}.pt'
        if path.exists():result=torch.load(path,weights_only=False)
        else:
            began=time.monotonic();value,response=oracle.simulate(q,v,basis,seed)
            result=dict(values=value,responses=response,seconds=time.monotonic()-began,atoms=len(q))
            save(path,result);publish(c,f'gates/{path.name}',result)
            print('Periodic1ps reference',index,result['seconds'],flush=True)
        opath=reference/'technical/gates'/f'parent{index}-radius48.pt'
        if sha(opath)!=c['open_reference_queries'][str(index)]:raise ValueError('Open48A query changed')
        cropped=torch.load(opath,weights_only=False)
        for h,horizon in enumerate(c['horizons_fs']):
            s=slice(h*128,(h+1)*128);ve=error(cropped['values'][:,s],result['values'][:,s])
            for j,kind in [(slice(0,2),'position'),(slice(2,4),'velocity')]:
                he=error(cropped['responses'][:,s,j],result['responses'][:,s,j])
                boundary.append(dict(parent=index,horizon_fs=horizon,direction=kind,value_rms=ve['rms'],
                    response_relative=he['relative'],passed=ve['rms']<=c['halo_value_tolerance'] and he['relative']<=c['halo_response_tolerance']))
        write_metric_rows(rows,root(c)/'analyses/numerical-gates-v1',family=FAMILY,name='periodic-force')
        write_metric_rows(boundary,root(c)/'analyses/numerical-gates-v1',family=FAMILY,name='environment')
        del oracle,q,v,basis;gc.collect();torch.cuda.empty_cache()
    cropped=all(r['passed']for r in boundary)
    write_json(out/'complete.json',dict(state='complete',identity=identity,radius_A=48 if cropped else None,
        boundary='validated48A open acceleration'if cropped else 'full native periodic cell',horizons_fs=c['horizons_fs']))
