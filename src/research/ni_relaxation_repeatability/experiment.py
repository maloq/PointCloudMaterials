"""Matched numerical assays of force ordering, minima and response stability."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from src.experiment_runner.artifacts import file_hash, json_digest, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.simulation.relaxation import relax_frame
from src.simulation.eam_gpu.relaxation_fast import FixedBoxCallback
from src.simulation.eam_gpu.benchmark import reference
from src.data.conversion.relaxation import read_relaxed, convert
from src.data.relaxed_targets.worker import publish
from src.research.ni_response.model import Motion
from src.research.ni_response.data import resolve_path
from . import backend

FAMILY='ni_quench_repeatability'


def read(path):return json.loads(Path(path).read_text())


def prepare(c):
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    parent=Path(c['parent']);old=read(parent/'technical/plan.json')
    audit=read(parent/'technical/repeatability-audit-node60.json')
    response_jobs={job['key']:job for job in old['response_jobs']};jobs=[]
    for key in audit['jobs']:
        job=response_jobs[key]
        folder=parent/'technical/physical-bank'/key
        jobs.append(dict(key=key,source=job['source_key'],folder=str(folder),
            restart_sha256=file_hash(folder/'restart.pt'),response_sha256=file_hash(folder/'response.pt')))
    payload=dict(config=c,jobs=jobs,implementation=backend.implementation(),
        parent_plan_sha256=file_hash(parent/'technical/plan.json'),
        audit_sha256=file_hash(parent/'technical/repeatability-audit-node60.json'),
        parent_config_sha256=file_hash(Path(read(parent/'technical/launch.json')['bundle'])/'config.json'))
    payload['identity']=json_digest(payload)
    target=out/'technical/plan.json'
    if target.exists() and read(target)!=payload:raise ValueError('Changed repeatability plan')
    write_json(target,payload)
    return payload


def inputs(job):
    folder=Path(job['folder'])
    for name in ['restart','response']:
        if file_hash(folder/(name+'.pt'))!=job[name+'_sha256']:raise ValueError('Changed physical bank')
    return (torch.load(folder/'restart.pt',weights_only=False,map_location='cpu'),
            torch.load(folder/'response.pt',weights_only=False,map_location='cpu'))


def force_audit(c,p,pc):
    out=Path(c['output']);rows=[];proof=[]
    for job in p['jobs']:
        md,_=inputs(job);q=md['future'][0,0,:,:3].numpy();box=md['box'];n=len(q)
        native=SimpleNamespace(get_natoms=lambda:n,extract_box=lambda:(np.zeros(3),box,0.,0.,0.,[1,1,1],0))
        ids=np.arange(1,n+1,dtype=np.int32)
        cpu_e,cpu_f,_,_=reference(q%box,box,pc['potential_files'][0])
        for mode in ['raw32','canonical32','canonical64']:
            callback=(FixedBoxCallback(native,pc['potential_files'][0],out) if mode=='raw32'
                else backend.CanonicalCallback(native,pc['potential_files'][0],out,
                    dtype=torch.float64 if mode=='canonical64' else torch.float32))
            forces=[];energies=[];members=[];orders=[]
            for repeat in range(8):
                callback.tags=None # Force an independent neighbor rebuild.
                f,e,_=callback.compute(ids,q);forces.append(f.copy());energies.append(e)
                count=callback.count.cpu().numpy();nb=callback.neighbors.cpu().numpy()
                valid=np.arange(256)[None,:]<count[:,None]
                masked=np.where(valid,nb,n)
                orders.append(hashlib.sha256(masked.tobytes()).hexdigest())
                members.append(hashlib.sha256(np.sort(masked,axis=1).tobytes()).hexdigest())
            a=np.asarray(forces);delta=a[1:]-a[:1]
            permutation=np.random.default_rng(261009).permutation(n)
            f,_,_=callback.compute(ids[permutation],q[permutation]);restored=np.empty_like(f);restored[permutation]=f
            row=dict(source=job['source'],mode=mode,rebuilds=8,
                force_repeat_rms_eV_A=float(np.sqrt(np.mean(delta**2))),
                force_repeat_max_eV_A=float(abs(delta).max()),
                bitwise_equal_forces=bool(np.all(a==a[:1])),
                unique_neighbor_orders=len(set(orders)),identical_membership=len(set(members))==1,
                atom_permutation_force_rms_eV_A=float(np.sqrt(np.mean((restored-a[0])**2))),
                cpu_force_rms_eV_A=float(np.sqrt(np.mean((a[0]-cpu_f)**2))),
                cpu_force_max_eV_A=float(abs(a[0]-cpu_f).max()),
                cpu_energy_error_meV_atom=float((energies[0]-cpu_e)*1000/n))
            if not row['identical_membership']:raise ValueError('Neighbor membership changed at identical coordinates')
            rows.append(row);proof.append(dict(source=job['source'],mode=mode,orders=orders,membership=members))
            write_json(out/'technical/force-progress.json',dict(state='running',rows=rows))
            print(row,flush=True)
            del callback;gc.collect();torch.cuda.empty_cache()
    write_metric_rows(rows,out/'analyses/forces',family=FAMILY,name='repeatability',columns=list(rows[0]))
    write_json(out/'technical/force-complete.json',dict(identity=p['identity'],rows=len(rows),proof=proof))


def quench(c,p,pc,job,md,qv,profile,variant,repeat):
    name=f"{job['key']}/{profile}/{variant}-r{repeat}"
    archive=Path(c['archive'])/name;work=Path(c['scratch'])/name
    if (archive/'complete.json').exists():
        done=read(archive/'complete.json')
        if read(archive/'source/manifest.json')['plan']!=p['identity']:raise ValueError('Diagnostic archive belongs to another plan')
        if file_hash(archive/'patches.npz')!=done['patches_sha256']:raise ValueError('Changed diagnostic patch')
        with np.load(archive/'patches.npz') as a:return a['positions'],done
    work.mkdir(parents=True,exist_ok=True);source=work/'source';source.mkdir(exist_ok=True)
    write_json(source/'manifest.json',dict(plan=p['identity'],job=job['key'],variant=variant,
        input_sha256=json_digest(qv[:,:3].tolist()),input='exact float64 saved MD checkpoint'))
    n=len(qv);box=md['box'];trajectory=SimpleNamespace(root=source,positions=qv[None,:,:3],
        box_low=np.zeros((1,3)),box_high=box[None],atom_ids=md['atom_ids'],
        atom_types=np.ones(n,dtype=np.int32),atom_count=n,timesteps=np.array([0],dtype=np.int64))
    spec=c['profiles'][profile];settings=dict(pc,force_tolerance=spec['force_tolerance'])
    settings['lammps_command']=['python','-m','src.research.ni_relaxation_repeatability.backend',
        '--mode',spec['mode'],'--implementation-sha256',json_digest(p['implementation']),
        '--potential',settings['potential_files'][0]]
    def runner(directory,recipe):
        previous=Path.cwd()
        try:
            os.chdir(directory);backend.execute(directory,recipe,spec['mode'])
        finally:os.chdir(previous)
    start=time.monotonic()
    try:
        if not (work/'metadata.json').exists():
            relax_frame(trajectory,0,work,settings,reuse=False,runner=runner)
        if not (work/'patches.npz').exists():
            xyz,_=read_relaxed(work)
            # Save both native coordinates and local offsets before conversion
            # deletes the dump; a continuation uses these exact saved values.
            np.savez_compressed(work/'native-quench-state.npz',positions=xyz,box=box,atom_ids=md['atom_ids'])
            q=xyz[md['neighbors']]-xyz[md['neighbors'][0]];q-=box*np.rint(q/box)
            np.savez_compressed(work/'patches.npz',positions=q.astype(np.float32)[None])
        with np.load(work/'patches.npz') as saved:patch=saved['positions']
        meta=read(work/'metadata.json')
        if not (work/'conversion.json').exists():
            convert(work,delete_source=True,local_cloud_dtype='float32')
        done=dict(plan_identity=p['identity'],profile=profile,variant=variant,repeat=repeat,source=job['source'],
            input_sha256=meta['input_sha256'],fmax_eV_A=meta['fmax_eV_per_A'],energy_eV=meta['energy_eV'],
            seconds=meta['seconds'],total_seconds=time.monotonic()-start,
            patches_sha256=file_hash(work/'patches.npz'),native_state_sha256=file_hash(work/'native-quench-state.npz'))
        write_json(work/'complete.json',done);publish(work,archive)
        return patch,done
    except Exception:
        write_json(work/'failure.json',dict(error=traceback.format_exc(),time=time.time()))
        failure=Path(c['archive'])/'failures'/name/str(time.time_ns())
        publish(work,failure)
        raise


def collect(c,p,pc,lane):
    out=Path(c['output']);torch.set_num_threads(2)
    parent_config=read(Path(read(Path(c['parent'])/'technical/launch.json')['bundle'])/'config.json')
    model=Motion(parent_config).cuda().eval().requires_grad_(False)
    rows=[]
    with torch.no_grad():
        for global_index in range(lane,len(p['jobs']),2):
            job=p['jobs'][global_index]
            md,bank=inputs(job);qv=md['future'][0,0].numpy();factor=parent_config['length_factor']
            results={}
            profiles=['raw32','canonical32','canonical32_tight']+(['canonical64'] if global_index==0 else [])
            for profile in profiles:
                for repeat in range(2):
                    patch,done=quench(c,p,pc,job,md,qv,profile,'base',repeat)
                    z=model.geometry(torch.from_numpy(patch).cuda()*factor).cpu().numpy()
                    results[(profile,repeat)]=(patch,z,done)
                    write_json(out/f'technical/progress-{lane}.json',dict(stage='quench',source=job['source'],profile=profile,repeat=repeat))
                a,za,da=results[(profile,0)];b,zb,db=results[(profile,1)]
                row=dict(source=job['source'],profile=profile,comparison='independent_repeat',
                    coordinate_rms_A=float(np.sqrt(np.mean((b-a)**2))),
                    embedding_rms=float(np.sqrt(np.mean((zb-za)**2))),
                    energy_difference_meV_atom=(db['energy_eV']-da['energy_eV'])*1000/len(qv),
                    seconds_first=da['seconds'],seconds_second=db['seconds'],
                    bitwise_equal_patches=bool(np.array_equal(a,b)))
                rows.append(row);print(row,flush=True)
            # Cross-recipe agreement and CPU convergence at each baseline minimum.
            for profile in profiles:
                a,za,da=results[('canonical32',0)];b,zb,db=results[(profile,0)]
                path=Path(c['archive'])/job['key']/profile/'base-r0'
                with np.load(path/'native-quench-state.npz') as archive:
                    e,f,_,_=reference(archive['positions']%md['box'],md['box'],pc['potential_files'][0])
                rows.append(dict(source=job['source'],profile=profile,comparison='versus_canonical32',
                    coordinate_rms_A=float(np.sqrt(np.mean((b-a)**2))),embedding_rms=float(np.sqrt(np.mean((zb-za)**2))),
                    energy_difference_meV_atom=(db['energy_eV']-da['energy_eV'])*1000/len(qv),
                    cpu_fmax_eV_A=float(abs(f).max()),cpu_energy_error_meV_atom=(db['energy_eV']-e)*1000/len(qv)))
            for amplitude in [1e-7,1e-4]:
                perturb=np.random.default_rng(261009).normal(size=(80,3));perturb-=perturb.mean(0)
                perturb*=amplitude/np.sqrt(np.mean(np.sum(perturb**2,-1)))
                changed=qv.copy();changed[md['neighbors'],:3]+=perturb
                patch,done=quench(c,p,pc,job,md,changed,'canonical32',f'jitter-{amplitude:g}A',0)
                z=model.geometry(torch.from_numpy(patch).cuda()*factor).cpu().numpy()
                a,za,da=results[('canonical32',0)]
                rows.append(dict(source=job['source'],profile='canonical32',comparison=f'input_jitter_{amplitude:g}A',
                    coordinate_rms_A=float(np.sqrt(np.mean((patch-a)**2))),embedding_rms=float(np.sqrt(np.mean((z-za)**2))),
                    energy_difference_meV_atom=(done['energy_eV']-da['energy_eV'])*1000/len(qv)))
            # Same saved ten signed MD endpoints, now with canonical quenches.
            codes=[]
            for shot in range(2):
                values=[]
                for variant in range(5):
                    if shot==0 and variant==0:patch=results[('canonical32',0)][0]
                    else:patch,_=quench(c,p,pc,job,md,md['future'][shot,variant].numpy(),
                        'canonical32',f'shot-{shot}-variant-{variant}',0)
                    values.append(model.geometry(torch.from_numpy(patch).cuda()*factor).cpu())
                codes.append(torch.cat(values))
            new=torch.stack(codes).mean(0)
            old=model.geometry(bank['future'].flatten(0,1).cuda()).reshape(2,5,128).mean(0).cpu()
            for direction,a,b in [('position',1,2),('velocity',3,4)]:
                # response_epsilon=.5, so denominator 2*epsilon=1.
                denominator=2*parent_config['student_difference']['response_epsilon']
                dr=(new[b]-new[a])/denominator;old_dr=(old[b]-old[a])/denominator
                rows.append(dict(source=job['source'],profile='canonical32',comparison='signed_response_'+direction,
                    response_rms=float(dr.square().mean().sqrt()),old_response_rms=float(old_dr.square().mean().sqrt()),
                    response_difference_rms=float((dr-old_dr).square().mean().sqrt()),
                    response_cosine=float(torch.nn.functional.cosine_similarity(dr,old_dr,dim=0))))
            write_json(out/f'technical/rows-{lane}.json',rows)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    write_metric_rows(rows,out/f'analyses/lane-{lane}',family=FAMILY,name='quenches',columns=columns)
    write_json(out/f'technical/complete-{lane}.json',dict(identity=p['identity'],state='complete',rows=len(rows)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','forces','quenches'])
    ap.add_argument('--config',required=True);ap.add_argument('--lane',type=int,default=0)
    a=ap.parse_args();c=read(a.config)
    c={**c,**{k:str(resolve_path(c[k])) for k in ['output','parent','scratch','archive']}}
    out=Path(c['output'])
    torch.set_num_threads(2)
    try:
        p=prepare(c) if a.stage=='prepare' else read(out/'technical/plan.json')
        if json_digest(backend.implementation())!=json_digest(p['implementation']):raise ValueError('Changed diagnostic backend')
        pc=read(Path(read(Path(c['parent'])/'technical/launch.json')['bundle'])/'config.json')['quench_settings']
        if a.stage=='forces':force_audit(c,p,pc)
        elif a.stage=='quenches':collect(c,p,pc,a.lane)
        else:print(p['identity'])
    except Exception:
        write_json(out/f'technical/failed-{a.stage}-{a.lane}.json',dict(error=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__':main()
