"""Detached, checkpointed full-cell continuations for the recovery assay."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from ase import units

from src.data.fixed_cohort.protocol import sha, write_json
from src.data.trajectories.lammps import write_temporal_lammps_binary, TemporalLAMMPSBinaryTrajectory
from src.project_runtime.paths import resolve_path
from src.postprocessing.descriptors.ptm import full_labels
from src.research.meam_md.dynamics import archive_file
from src.research.nucleus_recovery import roots, read, analyze_branch


def files(c, root):
    directory = root/'potential';directory.mkdir(exist_ok=True)
    result = {k:resolve_path(v) for k,v in c['meam_files'].items()}
    for source in result.values():
        target = directory/source.name
        if not target.exists():shutil.copy2(source,target)
        if sha(target)!=sha(source):raise ValueError('Potential copy changed')
    return result


def arrays(root, count, atoms):
    result = []
    for filename,dtype,shape in [('positions.building.npy',np.float32,(count,atoms,3)),
                               ('boxes.npy',np.float32,(count,3)),('labels.npy',np.uint8,(count,atoms))]:
        path = root/filename
        value = np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
        if value.shape!=shape or value.dtype!=dtype:raise ValueError('Changed continuation observations')
        result.append(value)
    return result


def save_state(root, archive, **values):
    path = root/'state.building.npz'
    np.savez(path,**values)
    path.replace(root/'state.npz')
    archive_file(root/'state.npz',archive/'state.npz')


def observe(c, root, archive, observations, index, q, box, **state):
    if not np.isfinite(q).all() or not np.isfinite(box).all():
        raise FloatingPointError('Nonfinite continuation state')
    positions,boxes,labels = observations
    wrapped = np.mod(q.astype(float),box.astype(float))
    positions[index] = np.mod(wrapped.astype(np.float32),box.astype(np.float32))
    boxes[index] = box
    # Label native generated coordinates before archive quantization.
    labels[index] = full_labels(wrapped,box,c['ptm']['rmsd_cutoff'])
    for value in observations:value.flush()
    save_state(root,archive,index=index,q=q,box=box,**state)


def progress(root, task, first_step, step, total, start, warmup):
    elapsed = time.monotonic()-start
    rate = (step-first_step)/elapsed if step>first_step else None
    write_json(root/'progress.json',dict(task=task['name'],step=step,total_steps=total,
        warmup_seconds=warmup,elapsed_seconds=elapsed,steps_per_second=rate,
        remaining_seconds=(total-step)/rate if rate else None,
        state_dtype='float32' if task['integrator']=='baoab' else 'native_float64',
        force_dtype='float32',energy_reduction_dtype='float64',updated_unix=time.time()))


def baoab(c, plan, task, state, event, root, archive, initial, observations, models, deadline):
    from src.research.meam_gpu.potential import FastAl
    device = torch.device('cuda');torch.set_num_threads(1)
    box = initial['box'];atoms=len(initial['q'])
    checkpoint = root/'state.npz'
    rng = torch.Generator(device=device).manual_seed(task['seed'])
    if checkpoint.exists():
        with np.load(checkpoint) as values:
            q = torch.tensor(values['q'],device=device,dtype=torch.float32)
            p = torch.tensor(values['p'],device=device,dtype=torch.float32)
            index = int(values['index']);rng.set_state(torch.tensor(values['rng'],dtype=torch.uint8))
    else:
        q = torch.tensor(initial['q'],device=device,dtype=torch.float32)
        v = torch.tensor(initial['v_A_ps'],device=device,dtype=torch.float32)/1000.
        p = v*(c['mass_amu']/units.fs)
        index = 0
        observe(c,root,archive,observations,index,q.cpu().numpy(),box,p=p.cpu().numpy(),rng=rng.get_state().cpu().numpy(),
                identity=plan['identity'])
    sample = round(state['cadence_ps']*1000/c['timestep_fs'])
    steps = round(state['duration_ps']*1000/c['timestep_fs'])
    if steps%sample:raise ValueError('Continuation misses observation endpoint')
    first = index*sample
    started = time.monotonic()
    if 'baoab' not in models:
        models['baoab'] = FastAl(files(c,root),atoms,box=box,dtype=torch.float32,compile_tangents=False)
    potential = models['baoab']
    potential.periodic_box.copy_(torch.tensor(box,device=device,dtype=torch.float32))
    potential.topology.reference = None
    _, force = potential.force(q)
    torch.cuda.synchronize();warmup=time.monotonic()-started;started=time.monotonic()
    print(task['name'],'warmup',round(warmup,2),'seconds; steps',steps,flush=True)
    dt=c['timestep_fs']*units.fs
    decay=math.exp(-c['timestep_fs']/c['friction_fs'])
    amplitude=math.sqrt(c['mass_amu']*units.kB*task['temperature_K']*(1-decay**2))
    for step in range(first+1,steps+1):
        p.add_(force,alpha=dt/2)
        q.add_(p,alpha=dt/(2*c['mass_amu']))
        p.mul_(decay).add_(torch.randn(q.shape,device=device,dtype=torch.float32,generator=rng),alpha=amplitude)
        q.add_(p,alpha=dt/(2*c['mass_amu']))
        _,force=potential.force(q)
        p.add_(force,alpha=dt/2)
        if step%100==0:
            progress(root,task,first,step,steps,started,warmup)
        if step%sample==0:
            observe(c,root,archive,observations,step//sample,q.cpu().numpy(),box,
                p=p.cpu().numpy(),rng=rng.get_state().cpu().numpy(),identity=plan['identity'])
            if time.time()>=deadline:return False
    progress(root,task,first,steps,steps,started,warmup)
    return True


def npt(c, plan, task, state, event, root, archive, initial, observations, models, deadline):
    from lammps import lammps
    from src.research.meam_gpu.npt import install
    settings=c['source_npt'];files(c,root)
    torch.set_num_threads(1)
    native=lammps(cmdargs=['-log',str(root/'lammps.log'),'-screen','none'])
    start_step=round(state['start_ps']*1000/settings['timestep_fs'])
    sample=round(state['cadence_ps']*1000/settings['timestep_fs'])
    total=round(state['duration_ps']*1000/settings['timestep_fs'])
    checkpoint=root/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as values:index=int(values['index'])
        restart=root/f'native-{index:04d}.restart.bin'
        native.command('read_restart '+str(restart))
    else:
        index=0;box=initial['box']
        native.commands_string(f'''units metal
atom_style atomic
boundary p p p
atom_modify map array
region cell block 0 {box[0]:.17g} 0 {box[1]:.17g} 0 {box[2]:.17g} units box
create_box 1 cell
mass 1 {settings['mass_amu']}
''')
        created=native.create_atoms(len(initial['q']),initial['atom_ids'],np.ones(len(initial['q']),np.int32),
            np.mod(initial['q'].astype(float),box).ravel(),v=initial['v_A_ps'].astype(float).ravel())
        if created!=len(initial['q']):raise ValueError('LAMMPS did not create the full saved cell')
        native.command(f'reset_timestep {start_step}')
    if 'npt_callback' not in models:
        callback=install(native,root);models['npt_callback']=callback
    else:
        callback=models['npt_callback'];callback.native=native;callback.directory=root;callback.calls=0;callback.started=time.monotonic()
        native.commands_string('pair_style zero 1.0\npair_coeff * *\nfix gpu_meam all external pf/callback 1 1\nfix_modify gpu_meam energy yes virial yes')
        native.set_fix_external_callback('gpu_meam',callback)
    native.commands_string(f'''neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings['timestep_fs']/1000}
fix remove_drift all momentum {settings['momentum_every_steps']} linear 1 1 1
fix ensemble all npt temp {task['temperature_K']} {task['temperature_K']} {settings['thermostat_ps']} iso {settings['pressure_bar']} {settings['pressure_bar']} {settings['barostat_ps']}
thermo {sample}
thermo_style custom step temp pe ke press vol
thermo_modify flush yes lost error
''')
    started=time.monotonic();native.command('run 0');warmup=time.monotonic()-started;started=time.monotonic()
    def save(index):
        ids=native.numpy.extract_atom('id')[:len(initial['q'])]
        order=np.argsort(ids)
        if not np.array_equal(ids[order],initial['atom_ids']):raise ValueError('Native atom identities changed')
        low,high,xy,yz,xz,_,_=native.extract_box()
        if (xy,yz,xz)!=(0.,0.,0.):raise ValueError('Unexpected tilted NPT cell')
        q=native.numpy.extract_atom('x')[:len(ids)][order].copy()-low
        v=native.numpy.extract_atom('v')[:len(ids)][order].copy()
        restart=root/f'native-{index:04d}.restart.bin'
        native.command('write_restart '+str(restart))
        archive_file(restart,archive/restart.name)
        observe(c,root,archive,observations,index,q,np.subtract(high,low),v_A_ps=v,identity=plan['identity'])
    if not checkpoint.exists():save(0)
    first=index*sample
    print(task['name'],'warmup',round(warmup,2),'seconds; steps',total,flush=True)
    try:
        for j in range(index+1,total//sample+1):
            native.command(f'run {sample} pre no post no')
            save(j);progress(root,task,first,j*sample,total,started,warmup)
            if time.time()>=deadline:return False
    finally:
        native.close()
    return True


def publish(c, plan, task, state, event, root, archive, initial):
    if not (root/'result.json').exists():
        analyze_branch(c,plan,task,state,event,root)
    binary=root/'trajectory_binary_float32'
    if not (root/'float16_conversion.json').exists():
        if not binary.exists():
            q=np.load(root/'positions.building.npy',mmap_mode='r')
            boxes=np.load(root/'boxes.npy')
            dt=c['timestep_fs'] if task['integrator']=='baoab' else c['source_npt']['timestep_fs']
            timesteps=np.rint((state['start_ps']+np.arange(len(q))*state['cadence_ps'])*1000/dt).astype(np.int64)
            write_temporal_lammps_binary(binary,positions=q,timesteps=timesteps,box_low=np.zeros_like(boxes),box_high=boxes,
                atom_ids=initial['atom_ids'],atom_types=np.ones(len(initial['q']),np.int32),atom_columns=('id','type','x','y','z'),
                source=dict(protocol=c['protocol'],plan_identity=plan['identity'],task=task,source_manifest=event['raw_manifest_sha256'],timestep_fs=dt),
                provenance=dict(original_source=event['raw'],start_frame=state['start_frame'],native_state='state.npz; native-*.restart.bin for NPT',
                    labels='PTM on native generated states before archive float16 rounding'),consume_positions_file=root/'positions.building.npy')
            # Release the moved NFS mapping before the converter retires its file.
            del q
        with (root/'conversion.log').open('w') as stream:
            subprocess.run([sys.executable,'scripts/convert_trajectory.py','temporal-storage',str(binary),'--delete-source'],check=True,stdout=stream,stderr=subprocess.STDOUT)
    TemporalLAMMPSBinaryTrajectory.load(root/'trajectory_binary_float16').verify_checksums()
    for path in (root/'trajectory_binary_float16').iterdir():archive_file(path,archive/'trajectory_binary_float16'/path.name)
    for name in ['float16_conversion.json','labels.npy','boxes.npy','state.npz','graph.npz','events.json','divergence.json','result.json','integration-complete.json','input.json','progress.json']:
        archive_file(root/name,archive/name)
    write_json(archive/'complete.json',dict(task=task['name'],plan_identity=plan['identity'],result_sha256=sha(archive/'result.json'),completed_unix=time.time()))


def run_branch(c, plan, task, state, event, models, deadline):
    _,scratch,store=roots(c)
    root=scratch/'branches'/task['name'];archive=store/'branches'/task['name']
    root.mkdir(parents=True,exist_ok=True);archive.mkdir(parents=True,exist_ok=True)
    (root/'technical').mkdir(exist_ok=True)
    for key,path in c['meam_files'].items():
        if sha(resolve_path(path))!=plan['potential_sha256'][key]:
            raise ValueError('Potential file changed after preparation: '+key)
    path=Path(state['initial_path'])
    if sha(path)!=state['initial_sha256']:raise ValueError('Changed saved initial snapshot')
    with np.load(path) as data:initial={k:data[k].copy() for k in data.files}
    quantization={}
    for field,enabled in [('q',task['variant'] in ('q_float16','qv_float16')),('v_A_ps',task['variant'] in ('v_float16','qv_float16'))]:
        original=initial[field].copy()
        if enabled:initial[field]=initial[field].astype(np.float16).astype(np.float32)
        delta=initial[field].astype(float)-original.astype(float)
        quantization[field]=dict(rms=float(np.sqrt(np.mean(delta**2))),maximum=float(np.abs(delta).max()))
    receipt=dict(task=task,state=state,event=event['key'],plan_identity=plan['identity'],quantization=quantization,
        host=os.uname().nodename,pid=os.getpid(),allocation=os.environ.get('SLURM_JOB_ID'),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        dynamics='float32 Q,p,noise,forces; float64 scalar energy' if task['integrator']=='baoab' else 'original native NPT integrator float64; GPU force arithmetic float32; bath state reset',
        velocity_source='actual simultaneous saved full-cell velocities, angstrom/ps',encoder_input=None,predictor_input=None)
    write_json(root/'input.json',receipt);archive_file(root/'input.json',archive/'input.json')
    if not (root/'integration-complete.json').exists():
        count=state['stop_frame']-state['start_frame']+1
        observations=arrays(root,count,event['atom_count'])
        runner=baoab if task['integrator']=='baoab' else npt
        finished=runner(c,plan,task,state,event,root,archive,initial,observations,models,deadline)
        if not finished:return False
        write_json(root/'integration-complete.json',dict(task=task['name'],plan_identity=plan['identity'],observations=count,completed_unix=time.time()))
        del observations
    publish(c,plan,task,state,event,root,archive,initial)
    print('completed',task['name'],json.dumps(read(root/'result.json')),flush=True)
    return True
