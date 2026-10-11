"""Native fixed-box Nose-Hoover control using the existing FastAl callback.

Native integration is float64, matching the NPT reference; it is explicitly
separate from the unchanged float32 BAOAB response integrator.
"""
import time

import numpy as np
import torch

from src.research import nucleus_recovery_md as md


def nvt(c, plan, task, state, case, root, archive, initial, observations, models, deadline):
    from lammps import lammps
    from src.research.meam_gpu.npt import install

    settings=c['source_npt']
    md.files(c,root)
    torch.set_num_threads(1)
    native=lammps(cmdargs=['-log',str(root/'lammps.log'),'-screen','none'])
    start_step=round(state['start_ps']*1000/settings['timestep_fs'])
    sample=round(state['cadence_ps']*1000/settings['timestep_fs'])
    total=round(state['duration_ps']*1000/settings['timestep_fs'])
    if total%sample:raise ValueError('NVT observation endpoint is not an exact step')
    checkpoint=root/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as values:index=int(values['index'])
        native.command('read_restart '+str(root/f'native-{index:04d}.restart.bin'))
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
        if created!=len(initial['q']):raise ValueError('NVT did not create the full saved cell')
        native.command(f'reset_timestep {start_step}')
    if 'nvt_callback' not in models:
        callback=install(native,root);models['nvt_callback']=callback
    else:
        callback=models['nvt_callback'];callback.native=native;callback.directory=root
        callback.calls=0;callback.started=time.monotonic()
        native.commands_string('pair_style zero 1.0\npair_coeff * *\nfix gpu_meam all external pf/callback 1 1\nfix_modify gpu_meam energy yes virial yes')
        native.set_fix_external_callback('gpu_meam',callback)
    commands=f'''neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep {settings['timestep_fs']/1000}
fix remove_drift all momentum {settings['momentum_every_steps']} linear 1 1 1
fix ensemble all nvt temp {task['temperature_K']} {task['temperature_K']} {settings['thermostat_ps']}
thermo {sample}
thermo_style custom step temp pe ke press vol
thermo_modify flush yes lost error
'''
    (root/'nvt-controls.in').write_text(commands)
    md.archive_file(root/'nvt-controls.in',archive/'nvt-controls.in')
    native.commands_string(commands)
    started=time.monotonic();native.command('run 0')
    warmup=time.monotonic()-started;started=time.monotonic()

    def save(index):
        ids=native.numpy.extract_atom('id')[:len(initial['q'])]
        order=np.argsort(ids)
        if not np.array_equal(ids[order],initial['atom_ids']):raise ValueError('NVT atom identities changed')
        low,high,xy,yz,xz,_,_=native.extract_box()
        if (xy,yz,xz)!=(0.,0.,0.) or not np.array_equal(np.subtract(high,low),initial['box']):
            raise ValueError('Fixed-box NVT cell changed')
        q=native.numpy.extract_atom('x')[:len(ids)][order].copy()-low
        v=native.numpy.extract_atom('v')[:len(ids)][order].copy()
        restart=root/f'native-{index:04d}.restart.bin'
        native.command('write_restart '+str(restart))
        md.archive_file(restart,archive/restart.name)
        md.observe(c,root,archive,observations,index,q,np.subtract(high,low),v_A_ps=v,identity=plan['identity'])

    if not checkpoint.exists():save(0)
    first=index*sample
    print(task['name'],'warmup',round(warmup,2),'seconds; steps',total,flush=True)
    try:
        for j in range(index+1,total//sample+1):
            native.command(f'run {sample} pre no post no')
            save(j);md.progress(root,task,first,j*sample,total,started,warmup)
            if time.time()>=deadline:return False
    finally:
        native.close()
        md.archive_file(root/'lammps.log',archive/f'lammps-through-{j if total>first else index:04d}.log')
    return True
