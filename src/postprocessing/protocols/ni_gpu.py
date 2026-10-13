"""Ni GPU descriptors followed by the unchanged CPU chronological lineage search."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import getpass
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, machine, resolve_path
from src.postprocessing import extraction
from src.postprocessing.io import read
from . import ni
from .ni_queue import active_jobs, execute


def campaign(settings):
    original = read(resolve_path(settings['campaign_config']))
    path = resolve_path(original['launch'])/'plan.json'
    if sha(path) != settings['campaign_plan_sha256']:
        raise ValueError('Ni GPU search campaign plan changed')
    plan = read(path)
    if plan['config'] != original or digest({k:v for k,v in plan.items() if k!='identity'}) != plan['identity']:
        raise ValueError('Ni GPU search source population changed')
    # Keep the original MD recipe; bind a separately declared observation window.
    derived = dict(original, analysis=settings['analysis'], cache=settings['cache'])
    if settings.get('analysis_extent') == 'full_saved_trajectory':
        if 'observation_cap_ps' in settings:
            raise ValueError('Full-trajectory Ni analysis cannot declare an observation cap')
        derived['observation_window'] = dict(mode='full_saved_trajectory',total_cap_ps=None,
            cooling_ps=original['cooling_ps'],simulation_hold_cap_ps=settings['simulation_hold_cap_ps'],
            no_birth='Valid censored observation at the actual saved endpoint; not a failed run')
    if 'observation_cap_ps' in settings:
        cap = settings['observation_cap_ps']
        if cap <= original['cooling_ps'] or cap > original['cooling_ps']+original['hold_ps']:
            raise ValueError('Ni observation cap lies outside the declared hold')
        if abs(cap/original['sample_interval_ps']-round(cap/original['sample_interval_ps'])) > 1e-8:
            raise ValueError('Ni observation cap is not a native saved observation')
        derived['observation_window'] = dict(total_cap_ps=cap,cooling_ps=original['cooling_ps'],
            hold_cap_ps=cap-original['cooling_ps'],no_birth='Valid censored observation; not a failed run')
    return original, derived, plan


def reviewed_backend(settings):
    from src.postprocessing import gpu_extract
    audit_path = Path(__file__).resolve().parents[3]/settings['numerical_audit']
    if sha(audit_path) != settings['numerical_audit_sha256']:
        raise ValueError('GPU descriptor numerical audit changed')
    audit = read(audit_path)
    if settings['execution'] != 'resident' or settings['ptm_ordering'] != 'voronoi':
        raise ValueError('This campaign requests the reviewed resident/voronoi GPU producer')
    root = Path(gpu_extract.__file__).resolve().parents[2]
    files = {str(Path(p).relative_to(root)): value for p,value in gpu_extract.implementation().items()}
    reviewed = audit['final_sequence']['implementation']
    for name,value in files.items():
        key = str(REPO/name)
        if reviewed.get(key) != value:
            raise ValueError(f'GPU extraction implementation differs from its audit: {name}')
    return dict(name='gpu-extract', execution=settings['execution'], ptm_ordering=settings['ptm_ordering'],
                numerical_audit=settings['numerical_audit'], numerical_audit_sha256=settings['numerical_audit_sha256'],
                implementation=files, implementation_audit_section='final_sequence',
                lineage='unchanged CPU chronological tracking')


def plans(c, sources, coverage, backend):
    payload = ni.plans(c, sources, coverage)
    parent = payload['parent']; parent['descriptor_backend'] = backend
    parent['identity'] = digest({k:v for k,v in parent.items() if k!='identity'})
    tracking = payload['tracking']; tracking['parent_identity'] = parent['identity']
    tracking['identity'] = digest({k:v for k,v in tracking.items() if k!='identity'})
    return payload


def source_plan(settings, record, backend):
    original,c,_ = campaign(settings); tech = resolve_path(c['analysis'])/'technical'
    path = tech/'source-plans'/(record['id']+'.json')
    if path.exists():
        payload = ni.resume_payload(c, read(path))
        if payload['parent']['descriptor_backend'] != backend:
            raise ValueError('Changed GPU producer for an existing Ni source plan')
        source = payload['parent']['sources'][0]
        saved = tech/'source-inputs'/(record['id']+'.json')
        if read(saved)['completion_sha256'] != sha(resolve_path(original['archive'])/record['id']/'complete.json'):
            raise ValueError('Completed Ni source changed since GPU preparation')
    else:
        source = ni.source(original, record)  # Full checksums and actual native timeline.
        if settings.get('analysis_extent') == 'full_saved_trajectory':
            from .ni_saved_tail import include_saved_tail
            source = include_saved_tail(original,record,source)
        if 'observation_window' in c:
            window = c['observation_window']; raw_frames = source['frame_count']
            raw_identity = source['input_identity']; raw_end = source['times_ps'][-1]
            n = (raw_frames if window['total_cap_ps'] is None else
                 min(raw_frames,round(window['total_cap_ps']/source['native_cadence_ps'])+1))
            source['frames'] = source['frames'][:n]; source['times_ps'] = source['times_ps'][:n]
            source.update(sampled_frames=list(range(n)),views={'010':list(range(n))},frame_count=n)
            if n != raw_frames:
                source['input_identity'] = digest(dict(arrays=source['parts'][0]['array_sha256'],frames=source['frames']))
            outcome = read(resolve_path(original['archive'])/record['id']/'complete.json')
            source['observation_window'] = dict(window,raw_frame_count=raw_frames,raw_end_time_ps=raw_end,
                raw_input_identity=raw_identity,raw_termination=outcome['termination'],
                observed_end_time_ps=source['times_ps'][-1],
                observed_hold_ps=max(0,source['times_ps'][-1]-original['cooling_ps']),
                reached_hold_cap=(None if window['total_cap_ps'] is None else source['times_ps'][-1]==window['total_cap_ps']))
        coverage = [dict(source=source['id'], state='complete', frames=source['frame_count'],
                         temperature_K=record['temperature_K'])]
        payload = plans(c, [source], coverage, backend)
        saved = tech/'source-inputs'/(record['id']+'.json')
        write_json(saved, dict(input_identity=source['input_identity'], verified_raw_arrays=True,
                    completion_sha256=sha(resolve_path(original['archive'])/record['id']/'complete.json'),
                    parts=source['parts']))
        write_json(path, payload)
    if source['id'] != record['id'] or source['role'] != record['role']:
        raise ValueError('Ni GPU source identity or ancestry role changed')
    return payload, path, saved


def verify_cache(payload, source):
    """Bind GPU receipts to lineage inputs without relabelling them as CPU output."""
    root = extraction.source_root(payload['parent'], source)
    contract = read(root/'extraction.json'); definition = contract['definition']
    backend = payload['parent']['descriptor_backend']; config = payload['parent']['config']
    expected = dict(source_id=source['id'], input_identity=source['input_identity'],
                    frames=source['sampled_frames'], descriptor_config=config,
                    ptm_ordering=backend['ptm_ordering'], execution=backend['execution'],
                    chunk_frames=config['chunk_frames'])
    if (contract['identity'] != digest(definition) or contract['source'] != source
            or any(definition[key] != value for key,value in expected.items())):
        raise ValueError(f'GPU cache does not belong to the Ni source plan: {root}')
    code_root = Path(__file__).resolve().parents[3]
    if definition['implementation'] != {str(code_root/name):value for name,value in backend['implementation'].items()}:
        raise ValueError('GPU cache executing implementation changed')
    complete = read(root/'complete.json'); frames = source['sampled_frames']; chunk = config['chunk_frames']
    if complete['identity'] != contract['identity'] or complete['frames'] != len(frames):
        raise ValueError('Incomplete or foreign GPU extraction')
    bounds = [(i,min(i+chunk,len(frames))) for i in range(0,len(frames),chunk)]
    if [(r['start'],r['stop']) for r in complete['chunks']] != bounds:
        raise ValueError('GPU chunks do not cover the exact source timeline')
    for start,stop in bounds:
        name = f'clusters-{start:06d}-{stop:06d}'
        receipt = read(root/(name+'.json'))
        if (receipt != complete['chunks'][start//chunk] or receipt['identity'] != contract['identity']
                or receipt['frames'] != frames[start:stop] or receipt['layout'] != 'detector_arrays_v1'
                or sha(root/(name+'.npz')) != receipt['sha256']):
            raise ValueError(f'Changed GPU chunk: {root/name}')
    return dict(source=source['id'], extraction=str(root/'extraction.json'),
                extraction_sha256=sha(root/'extraction.json'), complete_sha256=sha(root/'complete.json'))


def prepare(settings):
    original,c,plan = campaign(settings); backend = reviewed_backend(settings)
    check_metric_docs(family='ni_births')
    launch = resolve_path(settings['launch']); launch.mkdir(parents=True, exist_ok=True)
    target = launch/'preparation.json'
    identity = dict(settings=settings, campaign_identity=plan['identity'], backend=backend)
    if target.exists() and read(target) != identity:
        raise ValueError('Ni GPU search preparation changed')
    write_json(target, identity)
    available = []
    for record in plan['records']:
        if (resolve_path(original['archive'])/record['id']/'complete.json').exists():
            payload,path,_ = source_plan(settings, record, backend)
            available.append(dict(source=record['id'], plan=str(path), sha256=sha(path),
                                  frames=payload['parent']['sources'][0]['frame_count']))
    write_json(launch/'preflight.json', dict(checked_at=datetime.now(timezone.utc).isoformat(),
                completed_sources=available, planned_sources=len(plan['records']),
                backend=backend, note='Only completed inputs prepared; no CUDA extraction or new numerical audit executed'))
    return identity


def submit(settings):
    original,c,plan = campaign(settings); launch = resolve_path(settings['launch'])
    if (launch/'launch.json').exists(): raise FileExistsError('Ni GPU search already submitted')
    preparation = read(launch/'preparation.json')
    if preparation != dict(settings=settings, campaign_identity=plan['identity'], backend=reviewed_backend(settings)):
        raise ValueError('Changed prepared Ni GPU search')
    parent = resolve_path(original['launch']); owner = read(parent/'launch.json'); dispatch = read(parent/'dispatch.json')
    if owner['identity'] != plan['identity']:
        raise ValueError('Ni search campaign ownership changed')
    controller_job = None
    if 'replaces_gpu_search_receipt' in settings:
        previous_path = resolve_path(settings['replaces_gpu_search_receipt'])
        if sha(previous_path) != settings['replaces_gpu_search_sha256']:
            raise ValueError('Previous GPU search ownership receipt changed')
        previous = read(previous_path)
        if (previous['campaign_identity'] != plan['identity']
                or dispatch['gpu_search_receipt'] != str(previous_path)
                or dispatch['jobs']['analyse'] != previous['jobs']['track']
                or dispatch['jobs']['gpu-extract'] != previous['jobs']['extract']):
            raise ValueError('Current Ni GPU search dispatch differs from its ownership receipt')
        old_jobs = [previous['jobs']['extract'],previous['jobs']['track']]
        simulation_jobs = previous['simulation_jobs']
        if 'producer_handoff_receipt' in settings:
            handoff_path = resolve_path(settings['producer_handoff_receipt'])
            if sha(handoff_path) != settings['producer_handoff_sha256']:
                raise ValueError('Ni producer handoff receipt changed')
            handoff = read(handoff_path)
            if (handoff['campaign_identity'] != plan['identity']
                    or handoff['analysis_original_receipt'] != str(previous_path)
                    or handoff['previous_job'] not in simulation_jobs):
                raise ValueError('Ni producer handoff does not belong to this source population')
            simulation_jobs = [job for job in simulation_jobs if job != handoff['previous_job']]
            simulation_jobs.extend(handoff['jobs'][name] for name in ['resume','fresh'])
        control_path = resolve_path(settings['hard_stop_receipt'])
        if sha(control_path) != settings['hard_stop_receipt_sha256']:
            raise ValueError('Hard-stop controller receipt changed')
        control = read(control_path); policy = read(control_path.parent/'policy.json')
        simulation_cap = (original['cooling_ps']+settings['simulation_hold_cap_ps']
                          if settings.get('analysis_extent')=='full_saved_trajectory' else settings['observation_cap_ps'])
        if control['campaign_identity'] != plan['identity'] or policy['total_cap_ps'] != simulation_cap:
            raise ValueError('Hard-stop controller and declared simulation policy differ')
        controller_job = control['jobs']['run']
    else:
        old_jobs = [owner['jobs']['analyse']]
        if dispatch['jobs']['analyse'] != old_jobs[0]:
            raise ValueError('Ni CPU search ownership or current dispatch changed')
        simulation_jobs = dispatch['analysis_dependency'].removeprefix('afterany:').split(':')
    from src.simulation.campaigns.ni_births.allocation import job_state
    if any(job_state(job) != 'PENDING' for job in old_jobs):
        raise ValueError('Replace only receipt-owned pending Ni search jobs')
    contract = check_metric_docs(family='ni_births')['ni_births']
    write_json(launch/'previous-dispatch.json', dispatch)
    code = ExecutionBundle.freeze(REPO, launch/'code', settings,
        directories=('src','configs','scripts','docs/metrics'),
        files=tuple((REPO/name,name) for name in contract['files']) +
              ((resolve_path(settings['numerical_audit']),settings['numerical_audit']),))
    host = machine(); host['catalog'] = str(code.root/'configs/datasets.json'); write_json(launch/'machine.json',host)
    cuda = host['execution']['gpu_meam_portable_cuda_root']
    environment = dict(PCM_PROJECT_ROOT=str(REPO),PCM_MACHINE_CONFIG=str(launch/'machine.json'),
        PYTHONPATH=str(code.root),CUDA_HOME=cuda,
        PATH=cuda+'/bin:'+str(Path(sys.executable).parent)+':/usr/bin:/bin',
        TORCH_EXTENSIONS_DIR=str(resolve_path(settings['cache'])/'torch-extensions'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        MAX_JOBS='4',QT_QPA_PLATFORM='offscreen')
    active = set(subprocess.check_output(['squeue','--user',getpass.getuser(),'-h','-o','%A'],text=True).split())
    waiting = [job for job in simulation_jobs if job in active]
    # Finished jobs may already be purged from Slurm; preserve their original
    # IDs in provenance but attach dependencies only to still-active producers.
    dependencies = 'afterany:'+':'.join(waiting) if waiting else None
    if controller_job in active:
        dependencies = ','.join(filter(None,[dependencies,'afterok:'+controller_job]))
    elif controller_job is not None and read(control_path.parent/'status.json')['state'] != 'complete':
        raise ValueError('Hard-stop controller ended without a completion receipt')
    receipt = dict(campaign_identity=plan['identity'],jobs={},code=str(code.root),
        replaced_jobs=old_jobs,simulation_jobs=simulation_jobs,simulation_dependency=dependencies,
        hard_stop_controller_job=controller_job,
        created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Current chat: user-requested GPU Ni search with the explicitly configured observation window')
    queue = SlurmQueue(launch,code,'src.postprocessing.protocols.ni_gpu',environment,
                       launch/'launch.json',receipt,'NI-GPU-SEARCH')
    released = False
    held = []
    try:
        for job in old_jobs:
            subprocess.run(['scontrol','hold',job],check=True); held.append(job)
        if any(job_state(job) != 'PENDING' for job in old_jobs):
            raise ValueError('Ni search started during replacement')
        with queue.submission():
            gpu = queue.submit('extract',['--hold','--gres=gpu:1',f'--cpus-per-task={settings["gpu_cpus"]}',
                '--mem='+settings['gpu_memory'],'--time='+settings['gpu_walltime']],
                partition=settings['gpu_partitions'],dependency=dependencies)
            tracking = queue.submit('track',['--hold',f'--cpus-per-task={settings["tracking_workers"]}',
                '--mem='+settings['tracking_memory'],'--time='+settings['tracking_walltime']],dependency='afterok:'+gpu)
        write_json(launch/'replacement.json',dict(replaced_jobs=old_jobs,new_gpu_job=gpu,
            new_tracking_job=tracking,simulation_dependency=dependencies,
            previous_dispatch_sha256=sha(launch/'previous-dispatch.json'),state='prepared'))
        subprocess.run(['scancel',*old_jobs],check=True)
        dispatch['jobs']['analyse'] = tracking; dispatch['jobs']['gpu-extract'] = gpu
        dispatch.update(analysis_dependency='afterok:'+gpu,gpu_extraction_dependency=dependencies,
                        gpu_search_receipt=str(launch/'launch.json'),replaced_analysis_jobs=old_jobs)
        if controller_job is not None:
            dispatch['jobs']['hard-stop'] = controller_job
            dispatch['observation_window'] = c['observation_window']
        write_json(parent/'dispatch.json',dispatch)
        for job in [gpu,tracking]:
            released = True; subprocess.run(['scontrol','release',job],check=True)
        write_json(launch/'replacement.json',dict(read(launch/'replacement.json'),state='submitted'))
    except BaseException:
        if not released:
            for job in receipt['jobs'].values():subprocess.run(['scancel',job],check=True)
            for job in held:
                if job_state(job)=='PENDING':subprocess.run(['scontrol','release',job],check=True)
        raise
    print(json.dumps(receipt),flush=True)


def extract(settings):
    from src.postprocessing.gpu_extract import extract as gpu_extract
    import torch
    original,c,plan = campaign(settings); backend = reviewed_backend(settings)
    launch = resolve_path(settings['launch']); receipt = read(launch/'launch.json')
    if active_jobs(receipt['simulation_jobs']): raise ValueError('Ni simulation producers are still active')
    if receipt['hard_stop_controller_job'] is not None:
        controller = resolve_path(settings['hard_stop_receipt'])
        if (sha(controller) != settings['hard_stop_receipt_sha256']
                or read(controller.parent/'status.json')['state'] != 'complete'):
            raise ValueError('Ni hard-stop controller has not finalized its publication')
    root = resolve_path(c['analysis']); tech = root/'technical'; tech.mkdir(parents=True,exist_ok=True)
    with (tech/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=[];coverage=[];references={};verified=[]
        for record in plan['records']:
            sid=record['id']; folder=resolve_path(original['archive'])/sid
            if not (folder/'complete.json').exists():
                state=read(folder/'outcome.json')['state'] if (folder/'outcome.json').exists() else 'not_started'
                coverage.append(dict(source=sid,state=state,frames=0,temperature_K=record['temperature_K']));continue
            payload,path,input_receipt = source_plan(settings,record,backend); source=payload['parent']['sources'][0]
            config=payload['parent']['config']
            arguments=dict(source=source,frames=source['sampled_frames'],config=config,
                output=str(extraction.source_root(payload['parent'],source)),chunk_frames=config['chunk_frames'],
                prefetch=settings['prefetch'],compression_workers=settings['compression_workers'],
                ptm_ordering=settings['ptm_ordering'],execution=settings['execution'],input_receipt=str(input_receipt))
            write_json(tech/'gpu-configs'/(sid+'.json'),arguments)
            gpu_extract(**arguments);verified.append(verify_cache(payload,source))
            sources.append(source);coverage.extend(payload['parent']['coverage'])
            references[sid]=dict(path=str(path),sha256=sha(path))
            write_json(tech/'gpu-progress.json',dict(state='running',completed_sources=len(sources),
                planned_sources=len(plan['records']),last_source=sid,gpu=torch.cuda.get_device_name()))
        population=plans(c,sources,coverage,backend);population['source_plans']=references
        path=tech/'plans.json'
        if path.exists() and read(path)!=population:raise ValueError('Ni GPU search population changed on resume')
        write_json(path,population)
        write_json(tech/'gpu-complete.json',dict(state='complete',plans_sha256=sha(path),sources=verified,
            completed_sources=len(sources),planned_sources=len(plan['records']),gpu=torch.cuda.get_device_name(),
            job=os.environ.get('SLURM_JOB_ID')))
        write_json(tech/'gpu-progress.json',dict(state='complete',completed_sources=len(sources),planned_sources=len(plan['records'])))


def track(settings):
    _,c,_=campaign(settings);backend=reviewed_backend(settings);tech=resolve_path(c['analysis'])/'technical'
    with (tech/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completion=read(tech/'gpu-complete.json');path=tech/'plans.json'
        if sha(path)!=completion['plans_sha256']:raise ValueError('GPU population changed before tracking')
        population=ni.resume_payload(c,read(path));tasks=[];bindings=[]
        for source in population['parent']['sources']:
            reference=population['source_plans'][source['id']]
            if sha(reference['path'])!=reference['sha256']:raise ValueError('Changed GPU source plan')
            payload=ni.resume_payload(c,read(reference['path']))
            if payload['parent']['descriptor_backend']!=backend:raise ValueError('GPU backend changed before tracking')
            bindings.append(verify_cache(payload,source))
            for detector in payload['parent']['config']['detectors']:
                tasks.append((reference['path'],reference['sha256'],'track',dict(source=source['id'],cadence='010',detector=detector)))
        if bindings!=completion['sources']:raise ValueError('GPU extraction binding changed before tracking')
        write_json(tech/'tracking-progress.json',dict(state='running',completed_tasks=0,total_tasks=len(tasks)))
        with ProcessPoolExecutor(max_workers=settings['tracking_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,_ in enumerate(pool.map(execute,tasks),1):
                write_json(tech/'tracking-progress.json',dict(state='running',completed_tasks=i,total_tasks=len(tasks)))
        ni.collect(c,population)
        if 'observation_window' in c:
            observation_outcomes(c,population)
        write_json(tech/'tracking-progress.json',dict(state='complete',completed_tasks=len(tasks),total_tasks=len(tasks)))


def observation_outcomes(c, population):
    """Retain every analyzed source/detector, including explicit zero counts."""
    root = resolve_path(c['analysis']); rows = []
    for source in population['parent']['sources']:
        window = source['observation_window']
        for detector in population['parent']['config']['detectors']:
            events = read(root/'technical/sources'/source['id']/'010'/detector/'events.json')
            births = [e for e in events if e['origin']=='emergence_from_liquid'
                      and e['establishment_confirmation_ps'] is not None]
            hold_births = [e for e in births if e['first_time_ps']>=c['cooling_ps']]
            rows.append(dict(source=source['id'],temperature_K=source['temperature_K'],role=source['role'],
                detector=detector,frames=source['frame_count'],observed_end_time_ps=window['observed_end_time_ps'],
                analysis_capped=window['total_cap_ps'] is not None,
                observed_hold_ps=window['observed_hold_ps'],reached_hold_cap=window['reached_hold_cap'],
                raw_end_time_ps=window['raw_end_time_ps'],raw_termination=window['raw_termination'],
                event_records=len(events),established_liquid_births=len(births),
                established_isothermal_liquid_births=len(hold_births),
                outcome='established_liquid_birth_observed' if births else 'no_established_liquid_birth_observed',
                hold_outcome='established_liquid_birth_observed' if hold_births else 'no_established_liquid_birth_observed',
                zero_birth_interpretation='No established birth observed within this window; later outcomes unobserved'))
    write_metric_rows(rows,root/'analyses/births-v1',family='ni_births',name='observation_outcomes')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['prepare','submit','extract','track']);parser.add_argument('--config',required=True)
    args=parser.parse_args();settings=read(resolve_path(args.config))
    {'prepare':prepare,'submit':submit,'extract':extract,'track':track}[args.action](settings)


if __name__=='__main__':main()
