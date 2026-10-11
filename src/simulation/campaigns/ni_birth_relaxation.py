"""Pinned Ni birth windows, shared full-cell GPU quenches and detached workers."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, allocation_deadline
from src.project_runtime.paths import REPO, resolve_path, machine
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions, lock, publish, verify_archive
from src.data import relaxation_index as index
from src.simulation.relaxation import relax_frame
from src.simulation.eam_gpu.relaxation import implementation


def read(path):
    return json.loads(Path(path).read_text())


def frame_sha(raw, frame):
    h = hashlib.sha256()
    for value in (raw.positions[frame], raw.box_low[frame], raw.box_high[frame],
                  raw.timesteps[frame:frame+1], raw.atom_ids, raw.atom_types):
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def settings(c, *, potential_file=None):
    potential = potential_file or str(resolve_path(c['potential']))
    if sha(potential) != c['potential_sha256']:
        raise ValueError('Ni generating potential changed')
    return dict(c['relaxation'], potential_files=[potential],
        pair_commands=['pair_style eam/fs', f'pair_coeff * * {potential} Ni'],
        lammps_command=[sys.executable, '-m', 'src.simulation.eam_gpu.relaxation',
            '--implementation-sha256', digest(implementation()), '--potential', potential])


def prepare(c):
    root = resolve_path(c['output']); target = root/'technical/plan.json'
    if target.exists():
        old = read(target)
        if old['config'] != c:
            raise ValueError('Ni relaxation configuration changed')
        return old
    spec = index.recipe(settings(c))
    sources = {}; tasks = {}; windows = []; inputs = []; coverage = []
    for name in c['catalogues']:
        folder = resolve_path(name)
        complete = read(folder/'technical/complete.json')
        if complete['state'] != 'complete':
            raise ValueError(f'Unsealed Ni birth catalogue: {folder}')
        source_plan = folder/'technical/plans.json'
        table = folder/'analyses/births-v1/tables/events.csv'
        payload = read(source_plan)
        if payload['campaign']['potential_sha256'] != c['potential_sha256']:
            raise ValueError(f'Wrong generating potential: {folder}')
        events = [e for e in csv.DictReader(table.open()) if
                  e['detector'] == 'ptm' and e['origin'] == 'emergence_from_liquid'
                  and e['establishment_confirmation_ps']]
        if len(events) != complete['established_ptm_liquid_births']:
            raise ValueError(f'Birth count differs from completion: {folder}')
        inputs.append(dict(plan=str(source_plan), plan_sha256=sha(source_plan),
                           events=str(table), events_sha256=sha(table), births=len(events)))
        all_sources = {s['id']: s for s in payload['parent']['sources']}
        for sid, s in all_sources.items():
            coverage.append(dict(source=sid, dataset=s['dataset'], ancestor=s['ancestry_group'],
                                 role=s['role'], births=sum(e['source']==sid for e in events)))
        for sid in sorted({e['source'] for e in events}):
            s = all_sources[sid]
            if sid in sources:
                raise ValueError(f'Duplicate source across catalogues: {sid}')
            if s['material'] != 'Ni' or s['potential'] != ['ni-mendelev2012-eam']:
                raise ValueError(f'Unexpected Ni source provenance: {sid}')
            timeline = np.asarray(s['times_ps'], dtype=np.float64)
            np.testing.assert_allclose(np.diff(timeline), .1, rtol=0, atol=1e-8)
            parts = [ShootingBinaryTrajectory.load(p['path']) for p in s['parts']]
            for raw, part in zip(parts, s['parts'], strict=True):
                if sha(raw.root/'manifest.json') != part['manifest_sha256']:
                    raise ValueError(f'Changed native input: {raw.root}')
                np.testing.assert_allclose(np.diff(raw.timesteps)*part['timestep_fs']/1000,
                                           .1, rtol=0, atol=1e-10)
            sources[sid] = {k:s[k] for k in ('id','dataset','parts','ancestry_group','role','temperature_K','input_identity')}
            for event in [e for e in events if e['source']==sid]:
                onset = float(event[c['window_anchor']]); first = int(event['first_frame'])
                if abs(timeline[first]-onset)>1e-8:
                    raise ValueError(f'Birth frame/time mismatch: {sid}, {event["track"]}')
                event_id = f'{sid}-ptm-{event["track"]}-{event["episode"]}'
                window = dict(event_id=event_id, source=sid, event=event, tasks=[], missing_offsets_ps=[])
                for offset in np.arange(-c['before_ps']*10, c['after_ps']*10+1, dtype=int)/10:
                    when = onset + offset
                    near = int(np.searchsorted(timeline, when-1e-8))
                    if near >= len(timeline) or abs(timeline[near]-when)>1e-8:
                        window['missing_offsets_ps'].append(float(offset)); continue
                    part_id, frame = s['frames'][near]; raw = parts[part_id]; part=s['parts'][part_id]
                    if abs(float(raw.timesteps[frame])*part['timestep_fs']/1000-when)>1e-8:
                        raise ValueError(f'Actual saved timestep differs from catalogue: {sid}/{frame}')
                    task_id = f'{sid}-p{part_id}-f{frame:06d}'
                    if task_id not in tasks:
                        meta = dict(source_manifest_sha256=part['manifest_sha256'], source_frame=frame,
                                    source_timestep=int(raw.timesteps[frame]))
                        available = [e for e in index.candidates(meta, spec=spec) if index.available(e)]
                        tasks[task_id] = dict(id=task_id, source=sid, part=part_id, frame=frame,
                            time_ps=float(when), source_timestep=int(raw.timesteps[frame]),
                            frame_sha256=frame_sha(raw,frame), reused_candidates=[e['id'] for e in available])
                    window['tasks'].append(dict(task=task_id, offset_ps=float(offset)))
                windows.append(window)
    ordered = list(tasks.values()); np.random.default_rng(c['seed']).shuffle(ordered)
    plan = dict(config=c, sources=sources, tasks=ordered, windows=windows, inputs=inputs,
                coverage=coverage, settings=settings(c), recipe=spec,
                summary=dict(births=len(windows), sources=len(sources), unique_cells=len(tasks),
                    requested_event_frames=sum(len(w['tasks']) for w in windows),
                    missing_event_frames=sum(len(w['missing_offsets_ps']) for w in windows),
                    existing_exact_recipe_cells=sum(bool(t['reused_candidates']) for t in ordered),
                    roles=dict(Counter(s['role'] for s in sources.values()))))
    plan['identity'] = digest(plan)
    write_json(target, plan)
    from src.simulation.campaigns.birth_sources import register_entry
    register_entry(c['dataset_id'], dict(root='archive', path=c['archive_relative'], kind='simulation',
        dependencies=sorted({s['dataset'] for s in sources.values()})+['potential-ni-mendelev2012','relaxation-index-v1'],
        metadata=dict(title='Ni birth windows: full-cell GPU FIRE relaxation', materials=['Ni'],
            potential_ids=['ni-mendelev2012-eam'], role='relaxed_configurations', classification='planned',
            description='Native 0.1 ps windows from -6 to +6 ps around catalogue first_time_ps of established PTM liquid-origin roots. Full periodic fixed-box FIRE; overlapping source frames deduplicated.',
            ancestry='Paired views of original Ni source frames; original ancestry and roles retained; no new independent melts.',
            precision='Float64 native minimizer/states, float32 FastNi forces, float64 scalar energy. Verified float16 archive plus shared native restart for full-precision neighborhoods.',
            evidence=[str(target)], plan_identity=plan['identity'], **plan['summary'])))
    print(json.dumps(plan['summary']), flush=True)
    return plan


def load_plan(c):
    plan=read(resolve_path(c['output'])/'technical/plan.json')
    if plan['config']!=c or plan['settings']!=settings(c, potential_file=plan['settings']['potential_files'][0]):
        raise ValueError('Frozen relaxation plan or execution changed')
    return plan


def produce(plan, task, *, execution_settings=None, runner=None):
    c=plan['config']; tech=resolve_path(c['output'])/'technical'
    receipt=tech/'cells'/(task['id']+'.json')
    if receipt.exists():
        done=read(receipt)
        if done['identity']!=plan['identity']:raise ValueError('Wrong completed cell identity')
        verify_archive(Path(done['archive']));return done
    source=plan['sources'][task['source']];part=source['parts'][task['part']]
    raw=ShootingBinaryTrajectory.load(part['path']); frame=task['frame']
    if sha(raw.root/'manifest.json')!=part['manifest_sha256'] or frame_sha(raw,frame)!=task['frame_sha256']:
        raise ValueError(f'Pinned source frame changed: {task["id"]}')
    work=resolve_path(c['scratch'])/'cells'/task['id']
    archive=resolve_path(c['archive'])/'cells'/task['id']
    if not archive.exists():
        if not (work/'metadata.json').exists():
            absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count,)
            absolute.positions=AbsolutePositions(raw)
            relax_frame(absolute,frame,work,execution_settings or plan['settings'],runner=runner)
        meta=read(work/'metadata.json')
        if meta['source_manifest_sha256']!=part['manifest_sha256'] or meta['source_frame']!=frame:
            raise ValueError('Wrong relaxed cell after recovery')
        if not (work/'conversion.json').exists():
            subprocess.run([sys.executable,str(REPO/'scripts/convert_trajectory.py'),'relaxation',str(work),
                '--delete-source','--local-cloud-dtype','none'],check=True,stdout=subprocess.DEVNULL)
        publish(work,archive)
    verify_archive(archive)
    meta=read(archive/'metadata.json')
    done=dict(identity=plan['identity'],task=task,archive=str(archive),
        fmax_eV_per_A=meta['fmax_eV_per_A'],energy_eV=meta['energy_eV'],seconds=meta['seconds'],
        native_index_key=index.identity(meta,meta['index_recipe']),reuse=meta.get('reuse'),
        metadata_sha256=sha(archive/'metadata.json'),conversion_sha256=sha(archive/'conversion.json'))
    write_json(receipt,done);return done


def audit(c):
    """One real production quench; check native EAM endpoint forces/energy."""
    plan=load_plan(c);task=plan['tasks'][0];done=produce(plan,task)
    from lammps import lammps
    native=index.root()/'cells'/done['native_index_key']/'relaxed.restart'
    log=resolve_path(c['output'])/'technical/native-endpoint-audit.log'
    model=lammps(cmdargs=['-log',str(log),'-screen','none'])
    try:
        model.commands_string(f'units metal\natom_style atomic\nread_restart {native}\n'+
            '\n'.join(plan['settings']['pair_commands'])+'\nrun 0')
        fmax=float(np.abs(model.numpy.extract_atom('f')[:int(model.get_natoms())]).max())
        energy=float(model.get_thermo('pe'))
    finally:model.close()
    if fmax>c['relaxation']['force_tolerance']+1e-4 or abs(energy-done['energy_eV'])/32000>1e-5:
        raise ValueError(f'Ni quench endpoint disagrees with native EAM: fmax={fmax}, E={energy}')
    write_json(resolve_path(c['output'])/'technical/audit.json',dict(state='passed',task=task['id'],
        identity=plan['identity'],native_fmax_eV_per_A=fmax,native_energy_eV=energy,
        gpu_fmax_eV_per_A=done['fmax_eV_per_A'],gpu_energy_eV=done['energy_eV'],
        seconds=done['seconds'],job=os.environ.get('SLURM_JOB_ID')))


def worker(c, lane):
    plan=load_plan(c);root=resolve_path(c['output'])/'technical'
    proof=read(root/'audit.json')
    if proof['state']!='passed' or proof['identity']!=plan['identity']:raise ValueError('Quench audit missing')
    deadline=allocation_deadline(reserve_seconds=120);failures=[];completed=0
    tasks=plan['tasks'][lane::c['workers']]
    for task in tasks:
        if time.time()+c['relaxation']['frame_timeout_seconds']>=deadline:break
        with lock(root/'locks'/(task['id']+'.lock')) as acquired:
            if not acquired:raise RuntimeError(f'Concurrent quench of {task["id"]}')
            write_json(root/'workers'/f'{lane}.json',dict(state='running',task=task['id'],
                completed=completed,total=len(tasks),job=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname()))
            try:
                result=produce(plan,task);completed+=1
                print(json.dumps(dict(task=task['id'],seconds=result['seconds'],completed=completed,total=len(tasks))),flush=True)
            except Exception:
                failure=dict(task=task,error=traceback.format_exc());failures.append(task['id'])
                write_json(root/'failures'/(task['id']+'.json'),failure)
                work=resolve_path(c['scratch'])/'cells'/task['id']
                dest=resolve_path(c['archive'])/'failures'/task['id']
                if work.exists() and not dest.exists():publish(work,dest)
                print(json.dumps(failure),flush=True)
                if len(failures)>=3:break
    state='complete' if completed==len(tasks) else ('failed' if failures else 'deadline')
    write_json(root/'workers'/f'{lane}.json',dict(state=state,completed=completed,total=len(tasks),
        failures=failures,job=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname()))
    if state!='complete':raise RuntimeError(f'Ni relaxation lane {lane}: {state}; see recorded unfinished cells')


def status(c):
    plan=load_plan(c);tech=resolve_path(c['output'])/'technical'
    done=[read(p) for p in (tech/'cells').glob('*.json')]
    if any(v['identity']!=plan['identity'] for v in done):raise ValueError('Mixed completed cell identities')
    result=dict(state='complete' if len(done)==len(plan['tasks']) else 'incomplete',
        identity=plan['identity'],**plan['summary'],completed_cells=len(done),remaining_cells=len(plan['tasks'])-len(done),
        failures=[p.stem for p in (tech/'failures').glob('*.json')],
        worker_states=[read(p) for p in sorted((tech/'workers').glob('*.json'))])
    write_json(tech/'status.json',result)
    if result['state']=='complete':write_json(tech/'complete.json',result)
    print(json.dumps(result),flush=True)


def submit(c):
    plan=prepare(c);tech=resolve_path(c['output'])/'technical'
    if (tech/'launch.json').exists():raise FileExistsError('Ni relaxation already submitted')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','configs','scripts','docs/metrics'))
    host=machine();host['catalog']=str(bundle.root/'configs/datasets.json');write_json(tech/'machine.json',host)
    cuda=host['execution']['gpu_meam_portable_cuda_root']
    env=dict(PCM_PROJECT_ROOT=str(bundle.root),PCM_MACHINE_CONFIG=str(tech/'machine.json'),PYTHONPATH=str(bundle.root),
        CUDA_HOME=cuda,PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',UCX_TLS='tcp,self',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MAX_JOBS='2')
    receipt=dict(identity=plan['identity'],jobs={},code=str(bundle.root),
        created_at=datetime.now(timezone.utc).isoformat(),purpose='Current chat: user-authorized Ni GPU relaxation, -6 to +6 ps birth windows',
        implementation=implementation(),summary=plan['summary'])
    queue=SlurmQueue(tech,bundle,__name__,env,tech/'launch.json',receipt,'NI-RELAX')
    queue.module='src.simulation.campaigns.ni_birth_relaxation'
    with queue.submission():
        audit_job=queue.submit('audit',['--gres=gpu:1','--cpus-per-task=4','--mem=16G','--time=00:30:00',
            '--exclude=node52'],partition=c['partitions'])
        jobs=[]
        for lane in range(c['workers']):
            jobs.append(queue.submit(f'worker-{lane}',['--gres=gpu:1','--cpus-per-task=4','--mem=16G',
                '--time='+c['walltime'],'--exclude=node52'],partition=c['partitions'],
                dependency='afterok:'+audit_job,command_stage='worker',arguments=('--lane',str(lane))))
        queue.submit('status',['--cpus-per-task=1','--mem=4G','--time=00:15:00'],dependency='afterany:'+':'.join(jobs))
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','submit','audit','worker','status'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--lane',type=int,default=0)
    args=parser.parse_args();c=read(resolve_path(args.config))
    if args.stage=='worker':worker(c,args.lane)
    else:globals()[args.stage](c)


if __name__=='__main__':main()
