"""Finite displacements: actual forward changes versus local tangent predictions."""
import gc
import math
import time
import torch

from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.local_response.oracle import Teacher, LocalFeatures, OpenPotential, LocalOracle
from src.research.local_response.data import environment
from .common import FAMILY, root, reference, source_config, read, sha, write_json, archive_payload, read_payload


class CaptureFeatures(LocalFeatures):
    """Add raw12-observable changes; keep positions only in transient GPU memory."""
    def __call__(self,initial,path):
        self.last_path=path.detach().clone()
        phi=super().__call__(initial,path)
        anchor=self.observed(initial)
        change=torch.stack([self.observed(path[:,i])-anchor for i in range(path.shape[1])],1)
        return torch.cat((phi,change.flatten(1)),-1)


def minimum_local_distance(q):
    x=q.reshape(-1,3)
    distances=torch.cdist(x[:80],x,compute_mode='donot_use_mm_for_euclid_dist')
    ids=torch.arange(80,device=x.device);distances[ids,ids]=math.inf
    return float(distances.min())


def difference_rms(a,b):
    return float((a.double()-b.double()).square().mean().sqrt())


def metric(actual,linear):
    a,b=actual.double().flatten(),linear.double().flatten();den=a.norm().clamp_min(1e-12)
    return dict(actual_rms=float(a.square().mean().sqrt()),linear_rms=float(b.square().mean().sqrt()),
        error_rms=float((a-b).square().mean().sqrt()),relative_error=float((a-b).norm()/den),
        cosine=float(a@b/(a.norm()*b.norm()).clamp_min(1e-24)))


def prepare(c):
    states=torch.load(reference(c)/'technical/parents.pt',weights_only=False)
    rows=[]
    for index in c['forward']['parent_indices']:
        state=states[index]
        if state['role']!='train':raise ValueError('Displacement selection must use train ancestors only')
        x=state['q'];basis=torch.zeros(len(x),3,2,dtype=torch.float64);basis[:80]=state['basis']
        for eps in c['forward']['epsilons_A']:
            for direction in range(2):
                for sign in [-1,1]:
                    minimum=minimum_local_distance(x+sign*eps*basis[...,direction])
                    if minimum<c['forward']['initial_close_contact_limit_A']:
                        raise ValueError(f'Unphysical initial contact: parent{index},epsilon{eps},direction{direction},sign{sign},r={minimum}')
                    rows.append(dict(parent=index,epsilon_A=eps,direction=direction,sign=sign,
                        per_atom_rms_A=eps/math.sqrt(79),max_atom_displacement_A=float(eps*basis[...,direction].norm(dim=-1).max()),
                        minimum_local_distance_A=minimum))
    write_metric_rows(rows,root(c)/'analyses/forward-v1',family=FAMILY,name='initial-disturbances')


def run(c):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=source_config(c);fc=c['forward']
    if [int(x*cfg['oracle']['timestep_fs']) for x in cfg['oracle']['horizons_steps']]!=fc['horizons_fs']:
        raise ValueError('Forward horizon contract differs from original oracle')
    states=torch.load(reference(c)/'technical/parents.pt',weights_only=False)
    teacher=Teacher(cfg['oracle']);identity=read(root(c)/'technical/binding.json')['identity']
    deadline=allocation_deadline(reserve_seconds=240)
    for index in fc['parent_indices']:
        state=states[index]
        for radius in fc['radii_A']:
            cached_path=reference(c)/'technical/gates'/f'parent{index}-radius{radius}.pt'
            old=torch.load(cached_path,weights_only=False)
            if len(old['seeds'])!=fc['shots']:raise ValueError('Original gate shot count changed')
            q,basis,rows=environment(state,radius,torch.float32)
            features=CaptureFeatures(cfg['oracle'],torch.float32)
            potential=OpenPotential(teacher,len(rows),cfg['oracle']['skin_A'])
            oracle=LocalOracle(potential,features,cfg['oracle'],rows,state['original_atoms'])
            empty=q.new_empty(len(q),0)
            with torch.no_grad():energy0=float(potential(q))
            for shot,seed in enumerate(old['seeds']):
                baseline_name=f'parent{index}/radius{radius}/shot{shot}-baseline.pt'
                cached=read_payload(c,baseline_name)
                if time.time()>=deadline:raise TimeoutError('Forward stage reached allocation reserve')
                begin=time.monotonic()
                result=oracle.query(q,basis if cached is None else empty,[seed])
                torch.cuda.synchronize();seconds=time.monotonic()-begin
                base_path=features.last_path[0].reshape(len(fc['horizons_fs']),-1,3).clone()
                if not torch.isfinite(base_path).all():raise FloatingPointError('Nonfinite unperturbed path')
                value=result.values[0].cpu()
                if difference_rms(value[:256],old['values'][shot])>1e-6:
                    raise ValueError('Unperturbed RFF execution differs from original gate')
                if cached is None:
                    response=result.responses[0].cpu()
                    if difference_rms(response[:256],old['responses'][shot])>1e-5:
                        raise ValueError('Unperturbed RFF tangent differs from original gate')
                    cached=dict(identity=identity,parent=index,radius_A=radius,shot=shot,seed=seed,
                        values=value,responses=response,seconds=seconds,atoms=len(q)//3,
                        reference_sha256=sha(cached_path),energy_eV=energy0)
                    archive_payload(c,baseline_name,cached)
                elif difference_rms(cached['values'],value)>1e-6:
                    raise ValueError('Resumed central forward path changed')
                for eps in fc['epsilons_A']:
                    for direction in range(fc['directions']):
                        for sign in [-1,1]:
                            name=f'parent{index}/radius{radius}/shot{shot}-eps{eps:g}-dir{direction}-sign{sign}.pt'
                            if read_payload(c,name) is not None:continue
                            if time.time()>=deadline:raise TimeoutError('Forward stage reached allocation reserve')
                            initial=q+sign*eps*basis[:,direction]
                            with torch.no_grad():energy=float(potential(initial))
                            minimum=minimum_local_distance(initial)
                            if minimum<fc['initial_close_contact_limit_A']:
                                raise ValueError(f'Close contact in {name}: {minimum}')
                            print(dict(stage='forward',parent=index,radius_A=radius,shot=shot,
                                epsilon_A=eps,per_atom_rms_A=eps/math.sqrt(79),direction=direction,sign=sign),flush=True)
                            begin=time.monotonic();result=oracle.query(initial,empty,[seed]);torch.cuda.synchronize()
                            seconds=time.monotonic()-begin
                            path=features.last_path[0].reshape(len(fc['horizons_fs']),-1,3)
                            if not torch.isfinite(path).all():raise FloatingPointError(f'Nonfinite path: {name}')
                            relative=path[:,1:80]-path[:,:1]
                            base_relative=base_path[:,1:80]-base_path[:,:1]
                            divergence=(relative-base_relative).square().sum(-1).mean(-1).sqrt()
                            start_relative=initial.reshape(-1,3)[1:80]-initial.reshape(-1,3)[:1]
                            motion=(relative-start_relative).square().sum(-1).mean(-1).sqrt()
                            payload=dict(identity=identity,parent=index,radius_A=radius,shot=shot,seed=seed,
                                epsilon_A=eps,direction=direction,sign=sign,atoms=len(q)//3,
                                values=result.values[0].cpu(),seconds=seconds,minimum_initial_distance_A=minimum,
                                initial_energy_change_eV=energy-energy0,
                                local_divergence_rms_A=divergence.cpu(),local_motion_rms_A=motion.cpu(),
                                minimum_observed_distance_A=[minimum_local_distance(p) for p in path])
                            archive_payload(c,name,payload)
                            write_json(root(c)/'technical/forward-progress.json',
                                {k:v for k,v in payload.items() if not torch.is_tensor(v)})
            del oracle,features,potential,q,basis,base_path,result;gc.collect();torch.cuda.empty_cache()
    summarize(c)
    write_json(root(c)/'technical/forward-complete.json',dict(state='complete',identity=identity,
        perturbed_trajectories=len(fc['parent_indices'])*len(fc['radii_A'])*fc['shots']*len(fc['epsilons_A'])*fc['directions']*2))


def summarize(c):
    fc=c['forward'];rows=[];boundary=[];secants=[]
    blocks=[('RFF_20fs',0,128),('RFF_100fs_joint',128,256),('radial_20fs',256,264),
            ('angular_20fs',264,268),('radial_100fs',268,276),('angular_100fs',276,280)]
    physical=[]
    for index in fc['parent_indices']:
        for radius in fc['radii_A']:
            for shot in range(fc['shots']):
                baseline=read_payload(c,f'parent{index}/radius{radius}/shot{shot}-baseline.pt')
                if baseline is None:continue
                for eps in fc['epsilons_A']:
                    for direction in range(fc['directions']):
                        pair={sign:read_payload(c,f'parent{index}/radius{radius}/shot{shot}-eps{eps:g}-dir{direction}-sign{sign}.pt') for sign in [-1,1]}
                        if radius==fc['radii_A'][0] and all(p is not None for p in pair.values()):
                            larger={sign:read_payload(c,f'parent{index}/radius{fc["radii_A"][-1]}/shot{shot}-eps{eps:g}-dir{direction}-sign{sign}.pt') for sign in [-1,1]}
                            if all(p is not None for p in larger.values()):
                                for block,lo,hi in blocks[:2]:
                                    a=(pair[1]['values'][lo:hi]-pair[-1]['values'][lo:hi])/(2*eps)
                                    b=(larger[1]['values'][lo:hi]-larger[-1]['values'][lo:hi])/(2*eps)
                                    error=float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-12))
                                    secants.append(dict(parent=index,shot=shot,epsilon_A=eps,direction=direction,
                                        block=block,radius_A=radius,reference_A=fc['radii_A'][-1],
                                        secant_relative_error=error,passed=error<=fc['response_relative_tolerance']))
                        for sign,p in pair.items():
                            if p is None:continue
                            for block,lo,hi in blocks:
                                actual=p['values'][lo:hi]-baseline['values'][lo:hi]
                                linear=sign*eps*baseline['responses'][lo:hi,direction]
                                rows.append(dict(parent=index,radius_A=radius,shot=shot,epsilon_A=eps,
                                    per_atom_rms_A=eps/math.sqrt(79),direction=direction,sign=sign,block=block,
                                    **metric(actual,linear)))
                            for i,horizon in enumerate(fc['horizons_fs']):
                                physical.append(dict(parent=index,radius_A=radius,shot=shot,epsilon_A=eps,
                                    per_atom_rms_A=eps/math.sqrt(79),direction=direction,sign=sign,horizon_fs=horizon,
                                    local_divergence_rms_A=float(p['local_divergence_rms_A'][i]),
                                    divergence_amplification=float(p['local_divergence_rms_A'][i])/(eps/math.sqrt(79)),
                                    local_motion_rms_A=float(p['local_motion_rms_A'][i]),
                                    minimum_initial_distance_A=p['minimum_initial_distance_A'],
                                    minimum_observed_distance_A=p['minimum_observed_distance_A'][i],
                                    initial_energy_change_eV=p['initial_energy_change_eV'],seconds=p['seconds']))
                            if radius==fc['radii_A'][0]:
                                ref=read_payload(c,f'parent{index}/radius{fc["radii_A"][-1]}/shot{shot}-eps{eps:g}-dir{direction}-sign{sign}.pt')
                                if ref is not None:
                                    for block,lo,hi in blocks[:2]:
                                        rms=difference_rms(p['values'][lo:hi],ref['values'][lo:hi])
                                        boundary.append(dict(parent=index,shot=shot,epsilon_A=eps,direction=direction,
                                            sign=sign,block=block,radius_A=radius,reference_A=fc['radii_A'][-1],
                                            value_rms=rms,passed=rms<=fc['value_rms_tolerance']))
    out=root(c)/'analyses/forward-v1'
    for name,values in [('linearization',rows),('physical-response',physical),('boundary',boundary),('boundary-secants',secants)]:
        if values:write_metric_rows(values,out,family=FAMILY,name=name)
    if not rows:raise ValueError('No forward outputs to summarize')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    df=pd.DataFrame(rows);df=df[df.radius_A==max(fc['radii_A'])]
    if len(df):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for block in ['RFF_20fs','RFF_100fs_joint','radial_100fs','angular_100fs']:
            d=df[df.block==block].groupby('per_atom_rms_A').relative_error.agg(['median','min','max'])
            axes[0].plot(d.index,100*d['median'],'o-',label=block)
            axes[0].fill_between(d.index,100*d['min'],100*d['max'],alpha=.08)
        axes[0].set(xscale='log',yscale='log',xlabel='Initial RMS displacement per atom (Å)',
                    ylabel='Forward change versus tangent: relative error (%)')
        axes[0].axhline(10,color='grey',ls='--');axes[0].legend(fontsize=8)
        pf=pd.DataFrame(physical);pf=pf[pf.radius_A==max(fc['radii_A'])]
        for horizon,d in pf.groupby('horizon_fs'):
            d=d.groupby('per_atom_rms_A').local_divergence_rms_A.mean()
            axes[1].plot(d.index,d.values,'o-',label=f'{horizon}fs')
        axes[1].plot(d.index,d.index,'--',color='grey',label='Initial displacement')
        axes[1].set(xscale='log',yscale='log',xlabel='Initial RMS displacement per atom (Å)',
                    ylabel='Local trajectory divergence RMS (Å)');axes[1].legend()
        fig.suptitle('Actual forward simulations ·36Å moving environments · common noise')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/'plots'/f'finite-displacement.{ext}',dpi=180)
        plt.close(fig)
