"""Known-nucleus recovery from saved full-cell Q,V; no model fitting.

Original and shooting protocols keep their own cadence and ancestry. The
historical graph prefix is retained to distinguish births from crystal arrivals.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.execution import ExecutionBundle, allocation_deadline, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.postprocessing.protocols import ancestry
from src.postprocessing.protocols.extract import raw_source
from src.postprocessing.io import frame_geometry
from src.postprocessing.descriptors.ptm import full_labels

FAMILY = 'nucleus_recovery'


def read(path):
    return json.loads(Path(path).read_text())


def roots(c):
    return tuple(resolve_path(c[k]) for k in ('output', 'scratch', 'archive'))


class Reference:
    def __init__(self, event):
        self.event = event
        self.raw = ShootingBinaryTrajectory.load(event['raw'])
        if sha(self.raw.root/'manifest.json') != event['raw_manifest_sha256']:
            raise ValueError('Changed reference trajectory: '+event['key'])
        self.folder = Path(event['audit_folder'])
        if sha(self.folder/'graph.npz') != event['graph_sha256']:
            raise ValueError('Changed original lineage graph')
        with np.load(self.folder/'graph.npz') as values:
            self.graph = {k: values[k] for k in values.files}
        self.ptm = read(self.folder/'ptm-complete.json')
        self.chunk = None
        self.rows = np.empty(0, dtype=int)

    def labels(self, f):
        size = self.event['chunk_frames']
        start = f//size*size
        if self.chunk != start:
            name = f'ptm-{start:04d}-{min(start+size,len(self.raw.positions)):04d}.npz'
            if sha(self.folder/name) != self.ptm['files'][name]:
                raise ValueError('Changed stored PTM chunk: '+name)
            with np.load(self.folder/name) as values:
                self.chunk_labels = values['labels'].copy()
            self.chunk = start
        return self.chunk_labels[f-start]

    def frame(self, f):
        q, box = frame_geometry(self.raw, f)
        local, _ = ancestry.components(q, box, self.labels(f), self.event['lineage'])
        return q, box, np.where(local > 0, local+self.graph['start'][f], 0)


def prepare(c):
    check_metric_docs(family=FAMILY)
    out, scratch, archive = roots(c)
    target = out/'technical/plan.json'
    if target.exists():
        plan = read(target)
        if plan['config'] != c:
            raise ValueError('Existing recovery plan differs')
        return plan
    cache = resolve_path(c['birth_cache'])
    if read(cache/'manifest.json')['identity'] != c['birth_identity']:
        raise ValueError('Changed birth cohort')
    released = read(cache/'plan.json')
    rows = np.load(cache/'rows.npz')
    positive = np.flatnonzero((rows['role']=='test') & (rows['label']==1))
    pairs = sorted(set(zip(rows['source'][positive].tolist(), rows['event'][positive].tolist())))
    events = []
    for source, event_id in pairs:
        item = next(s for s in released['sources'] if s['id']==source)
        if item['role'] != 'test':
            raise ValueError('Birth row and source roles disagree')
        raw = raw_source(item)
        folder = resolve_path(c['original_audit'])/'technical/sources'/str(source)
        event = next(e for e in read(folder/'events.json') if e['event_id']==event_id and e['threshold']=='primary')
        candidates = [i for i in positive if rows['source'][i]==source and rows['event'][i]==event_id]
        # Earliest eligible appearance, then atom ID: fixed before any new MD.
        i = min(candidates, key=lambda j: (int(rows['appearance_frame'][j]), int(rows['atom'][j]), str(rows['id'][j])))
        events.append(dict(key=f'original-{source}-e{event_id}', panel='original', source=str(source),
            ancestry=item['lineage'], role=item['role'], temperature_K=item['audit_metadata']['temperature_K'],
            cadence_ps=.75, chunk_frames=32, raw=str(raw.root), raw_manifest_sha256=item['manifest_sha256'],
            audit_folder=str(folder), graph_sha256=sha(folder/'graph.npz'), target=event,
            appearance_frame=int(rows['appearance_frame'][i]), appearance_atom=int(rows['atom'][i]),
            appearance_row=str(rows['id'][i]), lineage=c['lineage']))
    shooting = read(resolve_path(c['shooting_audit'])/'technical/plan.json')
    for selection in c['precision_panel']:
        item = next(s for s in shooting['sources'] if s['id']==selection['source'])
        if len(item['parts']) != 1 or item['parts'][0]['storage_dtype'] != 'float32':
            raise ValueError('Precision panel must use one original float32 trajectory')
        if item['frames'] != [[0,f] for f in range(item['frame_count'])]:
            raise ValueError('Unexpected shooting source frame mapping')
        folder = resolve_path(c['shooting_audit'])/'technical/sources'/item['id']
        event = next(e for e in read(folder/'events.json') if e['event_id']==selection['event'])
        events.append(dict(key=item['id']+f'-e{selection["event"]}', panel='precision', source=item['id'],
            ancestry=item['ancestry_group'], role=item['role'], temperature_K=item['temperature_K'],
            cadence_ps=item['cadence_ps'], chunk_frames=16, raw=item['parts'][0]['path'],
            raw_manifest_sha256=item['parts'][0]['manifest_sha256'], audit_folder=str(folder),
            graph_sha256=sha(folder/'graph.npz'), target=event, lineage=c['lineage']))
    states, tasks = [], []
    for e in events:
        ref = Reference(e)
        raw, target_event = ref.raw, e['target']
        if target_event['kind'] != 'isolated_establishment':
            raise ValueError('Target is not an isolated birth')
        if raw.manifest['velocity_units'] != 'angstrom_per_ps':
            raise ValueError('Unsupported velocity units')
        if e['panel']=='precision' and (raw.positions.dtype!=np.float32 or raw.velocities.dtype!=np.float32):
            raise ValueError('Precision panel is not originally float32 Q,V')
        timeline = raw.timesteps*3./1000.
        if not np.allclose(np.diff(timeline), e['cadence_ps'], rtol=0, atol=1e-7):
            raise ValueError('Wrong native observation spacing')
        birth = target_event['birth_frame']
        stop = target_event['confirmation_frame']+round(c['tail_after_confirmation_ps']/e['cadence_ps'])
        if stop >= len(timeline):
            raise ValueError('Missing reference confirmation tail')
        e.update(stop_frame=stop, position_dtype=str(raw.positions.dtype), velocity_dtype=str(raw.velocities.dtype),
                 threshold=dict(size=c['establishment_atoms'], persistence_frames=1+round(c['persistence_ps']/e['cadence_ps'])),
                 atom_count=len(raw.atom_ids), target_birth_ps=float(timeline[birth]))
        reference_events, _, _ = ancestry.establish(ref.graph, e['threshold'])
        reproduced = next(x for x in reference_events if x['event_id']==target_event['event_id'])
        for k in ('birth_frame','confirmation_frame','birth_size','birth_node'):
            if reproduced[k] != target_event[k]:
                raise ValueError('Saved graph no longer reproduces historical birth: '+k)
        q, box, dense = ref.frame(birth)
        members = np.flatnonzero(dense==target_event['birth_node'])
        if len(members) != target_event['birth_size']:
            raise ValueError('Stored PTM no longer reproduces target birth core')
        e['core_rows'] = members.tolist()
        e['reference_reproduced'] = True
        offsets = c['birth_offsets_ps'] if e['panel']=='original' else c['precision_offsets_ps']
        starts = [(f'birth-minus-{offset:g}ps', birth-round(offset/e['cadence_ps'])) for offset in offsets]
        if e['panel']=='original':
            starts.append(('preappearance', e['appearance_frame']-round(c['preappearance_offset_ps']/e['cadence_ps'])))
        for offset_name, start in starts:
            if start < 0 or start >= birth:
                raise ValueError('Invalid prebirth start')
            name = e['key']+'-'+offset_name
            q, box = frame_geometry(raw, start)
            v = np.array(raw.velocities[start], dtype=np.float32)
            state = dict(name=name, event=e['key'], offset=offset_name, start_frame=start, stop_frame=stop,
                start_ps=float(timeline[start]), birth_offset_ps=float(timeline[birth]-timeline[start]),
                duration_ps=float(timeline[stop]-timeline[start]), cadence_ps=e['cadence_ps'],
                initial_path=str(archive/'initial'/f'{name}.npz'), panel=e['panel'])
            path = Path(state['initial_path']);path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, q=q.astype(np.float32), v_A_ps=v, box=box.astype(np.float32), atom_ids=raw.atom_ids)
            state['initial_sha256'] = sha(path)
            states.append(state)
            variants = ['as_saved'] if e['panel']=='original' else c['precision_variants']
            for replicate in range(c['noise_replicates']):
                seed = c['seed']+int(digest(dict(state=name, replicate=replicate))[:8],16)%1000000000
                for variant in variants:
                    temperatures = [e['temperature_K']]
                    if e['panel']=='original' and e['temperature_K'] != c['response_temperature_K']:
                        temperatures.append(c['response_temperature_K'])
                    for temperature in temperatures:
                        arms = ['sourceT'] if temperature==e['temperature_K'] else []
                        if temperature==c['response_temperature_K'] and e['panel']=='original':
                            arms.append('response450K')
                        name_t = f'{name}-{variant}-baoab-T{temperature:g}-r{replicate}'
                        tasks.append(dict(name=name_t, state=name, integrator='baoab', variant=variant,
                            temperature_K=temperature, arms=arms, replicate=replicate, seed=seed,
                            priority=(0 if offset_name in ('birth-minus-0.75ps','birth-minus-0.3ps') else 1,
                                      replicate, len(states), variants.index(variant), -temperature)))
            if e['panel']=='original':
                tasks.append(dict(name=name+'-native-npt', state=name, integrator='npt', variant='as_saved',
                    temperature_K=e['temperature_K'], arms=['sourceNPT'], replicate=0, seed=None,
                    priority=(0 if offset_name=='birth-minus-0.75ps' else 1, 0, len(states), -1, 0)))
        print('prepared reference', e['key'], 'birth', e['target_birth_ps'], 'ps', flush=True)
    tasks.sort(key=lambda t: t['priority'])
    plan = dict(config=c, events=events, states=states, tasks=tasks,
        source_plan_sha256=sha(cache/'plan.json'), rows_sha256=sha(cache/'rows.npz'),
        shooting_plan_sha256=sha(resolve_path(c['shooting_audit'])/'technical/plan.json'),
        potential_sha256={k:sha(resolve_path(v)) for k,v in c['meam_files'].items()},
        coverage=dict(original_events=len(pairs), original_sources=len(set(x[0] for x in pairs)),
            precision_events=len(c['precision_panel']),
            deferred=['other original births', 'dense 0.1/0.01 ps descendants', 'million-atom trajectories',
                      'other shooting campaigns', 'relaxed views (different initial dynamics)']))
    plan['identity'] = digest(plan)
    write_json(target, plan)
    write_json(archive/'plan.json', plan)
    print('planned', len(states), 'states', len(tasks), 'continuations', flush=True)
    return plan


def graph_prefix(ref, start):
    old = ref.graph
    end = int(old['start'][start])+1
    graph = dict(frame=old['frame'][:end].tolist(), size=old['size'][:end].tolist(),
        start=old['start'][:start].tolist(), crystalline_atoms=old['crystalline_atoms'][:start].tolist(),
        edges=old['edges'][old['edges'][:,1]<end].tolist())
    previous = [ref.frame(f)[2] for f in range(max(0,start-2), start)]
    return graph, previous


def add_frame(graph, previous, q, box, labels, f, settings):
    local, sizes = ancestry.components(q, box, labels, settings)
    start = len(graph['size'])-1
    graph['start'].append(start)
    immediate = set()
    for lag, old in enumerate(reversed(previous),1):
        parent, child, count, strong = ancestry.overlap(old, local, graph['size'], sizes, settings)
        if lag==1:
            immediate = set(child[strong].tolist())
        else:
            retain = np.array([int(v) not in immediate for v in child],bool)
            parent,child,count,strong = (v[retain] for v in (parent,child,count,strong))
        graph['edges'].extend(zip(parent.tolist(), (child+start).tolist(), count.tolist(), strong.astype(int).tolist(), [lag]*len(parent)))
    dense = np.where(local>0, local+start, 0).astype(np.int32)
    graph['frame'].extend([f]*(len(sizes)-1));graph['size'].extend(sizes[1:].tolist())
    graph['crystalline_atoms'].append(int(np.isin(labels,[1,2,3]).sum()))
    previous.append(dense)
    del previous[:-(settings['maximum_missing_frames']+1)]
    return dense


def analyze_branch(c, plan, task, state, event, root):
    ref = Reference(event)
    q = np.load(root/'positions.building.npy', mmap_mode='r')
    boxes = np.load(root/'boxes.npy')
    labels = np.load(root/'labels.npy', mmap_mode='r')
    graph, previous = graph_prefix(ref, state['start_frame'])
    dense = []
    rms = []
    core = np.asarray(event['core_rows'])
    for j in range(len(q)):
        f = state['start_frame']+j
        points = np.mod(q[j].astype(float), boxes[j].astype(float))
        dense.append(add_frame(graph,previous,points,boxes[j],labels[j],f,c['lineage']))
        original, original_box = frame_geometry(ref.raw,f)
        delta = points/boxes[j]-original/original_box
        delta -= np.rint(delta)
        delta *= original_box
        rms.append(dict(frame=f,time_ps=f*event['cadence_ps'], all_atoms_rms_A=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
            birth_core_rms_A=float(np.sqrt(np.mean(np.sum(delta[core]**2,axis=1))))))
    graph = {k:np.asarray(v,np.int64 if k=='edges' else np.int32) for k,v in graph.items()}
    graph['edges'] = graph['edges'].reshape(-1,5)
    graph['center_node'] = np.empty((0,len(graph['start'])),np.int32)
    events, lineage_roots, _ = ancestry.establish(graph,event['threshold'])
    class Geometry:
        rows = np.empty(0,dtype=int)
        def frame(self, f):
            j = f-state['start_frame']
            return ref.frame(f) if j<0 else (np.mod(q[j].astype(float),boxes[j]),boxes[j],dense[j])
    geometry = Geometry()
    item = dict(id=task['name'],center_atom_ids=[],legacy_center_atom_ids=[])
    ancestry.locate_births(c,item,graph,events,lineage_roots,geometry)
    candidate_records = []
    for e in events:
        if e['confirmation_frame'] < state['start_frame']:
            continue
        points, box, nodes = geometry.frame(e['birth_frame'])
        member = np.flatnonzero(nodes==e['birth_node'])
        offsets = points[core]-points[core[0]]
        offsets -= box*np.rint(offsets/box)
        reference_center = np.mod(points[core[0]]+offsets.mean(0),box)
        delta = np.asarray(e['centroid_A'])-reference_center
        delta -= box*np.rint(delta/box)
        distance = float(np.linalg.norm(delta))
        shared = int(np.intersect1d(member,core).size)
        common = shared>=c['lineage']['minimum_shared_atoms'] and shared>=c['lineage']['minimum_overlap_fraction']*min(len(member),len(core))
        fresh = e['birth_frame']>state['start_frame'] and e['kind']=='isolated_establishment'
        error = (e['birth_frame']-event['target']['birth_frame'])*event['cadence_ps']
        candidate_records.append(dict(**e, same_site=distance<=c['site_radius_A'], same_core=common,
            site_distance_A=distance, shared_core_atoms=shared, fresh_isolated=fresh, timing_error_ps=error))
    fresh = [x for x in candidate_records if x['fresh_isolated']]
    site = [x for x in fresh if x['same_site']]
    core_events = [x for x in fresh if x['same_core']]
    result = dict(task=task['name'],state=state['name'],event=event['key'],source=event['source'],ancestry=event['ancestry'],
        panel=event['panel'],offset=state['offset'],birth_offset_ps=state['birth_offset_ps'],duration_ps=state['duration_ps'],
        integrator=task['integrator'],variant=task['variant'],temperature_K=task['temperature_K'],arms=task['arms'],
        replicate=task['replicate'],seed=task['seed'],
        same_site=int(bool(site)),same_core=int(bool(core_events)),any_new_nucleus=int(bool(fresh)),
        same_site_on_time=int(any(abs(x['timing_error_ps'])<=c['timing_tolerance_ps'] for x in site)),
        earliest_same_site_error_ps=min((x['timing_error_ps'] for x in site),default=None),
        initial_site_crystal_atoms=int(np.isin(labels[0,core],[1,2,3]).sum()),
        final_core_rms_A=rms[-1]['birth_core_rms_A'], plan_identity=plan['identity'])
    np.savez_compressed(root/'graph.npz',**graph)
    write_json(root/'events.json',candidate_records)
    write_json(root/'divergence.json',rms)
    write_json(root/'result.json',result)
    return result


def summarize(c):
    out, scratch, archive = roots(c)
    plan = read(out/'technical/plan.json')
    records = [read(archive/'branches'/t['name']/'result.json') for t in plan['tasks']
               if (archive/'branches'/t['name']/'complete.json').exists()]
    with (out/'technical/summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        groups = defaultdict(list)
        for r in records:
            for arm in r['arms']:
                groups[(r['panel'],r['offset'],r['variant'],arm)].append(r)
        summaries = []
        metric_names = ['same_site','same_site_on_time','same_core','any_new_nucleus']
        for key, values in sorted(groups.items()):
            # Publish a group mean only for events with all planned replicates.
            required = 1 if key[3]=='sourceNPT' else c['noise_replicates']
            by_event = defaultdict(list)
            for r in values:by_event[r['event']].append(r)
            ready = {k:v for k,v in by_event.items() if len(v)==required}
            row = dict(zip(['panel','offset','variant','arm'],key), completed_branches=len(values),
                completed_events=len(ready), observed_events=len(by_event), independent_ancestors=len({v[0]['ancestry'] for v in ready.values()}))
            for metric in metric_names:
                row[metric] = float(np.mean([np.mean([v[metric] for v in rs]) for rs in ready.values()])) if ready else None
                row[metric+'_ci_low'] = row[metric+'_ci_high'] = None
                if len(ready)>1:
                    sources = sorted({rs[0]['ancestry'] for rs in ready.values()})
                    rng = np.random.default_rng(c['seed'])
                    totals = np.array([sum(np.mean([r[metric] for r in rs]) for rs in ready.values() if rs[0]['ancestry']==s) for s in sources])
                    counts = np.array([sum(rs[0]['ancestry']==s for rs in ready.values()) for s in sources])
                    sampled = rng.integers(len(sources),size=(c['bootstrap_draws'],len(sources)))
                    draws = totals[sampled].sum(1)/counts[sampled].sum(1)
                    row[metric+'_ci_low'],row[metric+'_ci_high'] = map(float,np.quantile(draws,[.025,.975]))
            summaries.append(row)
        if records:write_metric_rows(({**r,'arms':'|'.join(r['arms'])} for r in records),out,family=FAMILY,name='branches')
        if summaries:write_metric_rows(summaries,out,family=FAMILY,name='recovery')
        pairs = []
        precise = [r for r in records if r['panel']=='precision']
        for r in precise:
            if r['variant']=='float32':continue
            baseline = next((b for b in precise if b['state']==r['state'] and b['seed']==r['seed'] and b['variant']=='float32'),None)
            if baseline is not None:
                pairs.append(dict(state=r['state'],event=r['event'],ancestry=r['ancestry'],offset=r['offset'],variant=r['variant'],seed=r['seed'],
                    **{m+'_delta':r[m]-baseline[m] for m in metric_names}))
        if pairs:write_metric_rows(pairs,out,family=FAMILY,name='paired-precision')
        statuses = []
        for t in plan['tasks']:
            p = out/'technical/tasks'/t['name']/'stage.json'
            if p.exists():statuses.append(read(p))
        progress = dict(completed=len(records),planned=len(plan['tasks']),
            running=sum(x['state']=='running' for x in statuses),
            failed=sum(x['state']=='failed' for x in statuses),
            coverage=plan['coverage'],updated_unix=time.time())
        write_json(out/'technical/progress.json',progress)
        if summaries:plot_summary(out,summaries)
        lines = ['# Saved-state nucleus recovery','',f"Completed: {len(records)}/{len(plan['tasks'])} continuations.",
            '', 'Full-cell Q,V restarts; known isolated nuclei; original 0.75 ps and shooting 0.3 ps cadences remain separate.',
            'Rates use only fully replicated events. Partial groups are preliminary; see branch rows and coverage.',
            'The NPT arm resets missing thermostat/barostat state. The BAOAB arms use float32 states and forces.',
            'Only the separate float32 shooting panel isolates added Q/V rounding using paired future noise.',
            '', '[Recovery table](tables/recovery.csv) · [Branch results](tables/branches.csv) · [Definitions](tables/METRICS.md)',
            '', 'Other original births, dense descendants, million-atom data, other shooting campaigns and relaxed views are not tested here.']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
    return progress


def plot_summary(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = sorted({r['panel'] for r in rows})
    fig,axes = plt.subplots(1,len(panels),figsize=(7*len(panels),4),squeeze=False)
    for ax,panel in zip(axes[0],panels):
        selected = [r for r in rows if r['panel']==panel and r['same_site'] is not None]
        offsets = sorted({r['offset'] for r in selected},key=lambda s: (s=='preappearance',float(s.split('-')[2][:-2]) if s!='preappearance' else 0))
        for arm,variant in sorted({(r['arm'],r['variant']) for r in selected}):
            values = [next((r for r in selected if r['offset']==o and r['arm']==arm and r['variant']==variant),None) for o in offsets]
            x = [i for i,v in enumerate(values) if v is not None]
            ax.plot(x,[values[i]['same_site'] for i in x],'o-',label=f'{arm} / {variant}')
        ax.set_xticks(range(len(offsets)),[o.replace('birth-minus-','−') for o in offsets],rotation=25)
        ax.set_ylim(-.03,1.03);ax.set_ylabel('Fraction recovering the same birth site')
        ax.set_title(panel+' — complete event replicate sets only');ax.legend(fontsize=8)
    fig.tight_layout();(out/'plots').mkdir(exist_ok=True)
    fig.savefig(out/'plots/recovery.png',dpi=170);fig.savefig(out/'plots/recovery.svg');plt.close(fig)


def worker(c, lane):
    from src.research.nucleus_recovery_md import run_branch
    out,scratch,archive = roots(c)
    plan = read(out/'technical/plan.json')
    deadline = allocation_deadline(reserve_seconds=c['allocation_reserve_seconds'])
    models = {}
    for task in plan['tasks']:
        if time.time()>=deadline:
            write_json(out/f'technical/lane-{lane}-paused.json',dict(reason='allocation deadline',time=time.time()))
            return
        technical = out/'technical/tasks'/task['name'];technical.mkdir(parents=True,exist_ok=True)
        with (technical/'lease.lock').open('w') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (archive/'branches'/task['name']/'complete.json').exists():continue
            state = next(s for s in plan['states'] if s['name']==task['state'])
            event = next(e for e in plan['events'] if e['key']==state['event'])
            with recorded_stage(technical/'stage.json',task=task['name'],lane=lane,allocation=os.environ.get('SLURM_JOB_ID')) as stage:
                finished = run_branch(c,plan,task,state,event,models,deadline)
                if not finished:
                    stage.update(phase='checkpointed',reason='allocation deadline')
                    write_json(out/f'technical/lane-{lane}-paused.json',dict(task=task['name'],reason='allocation deadline',time=time.time()))
                    return
            summarize(c)
    write_json(out/f'technical/lane-{lane}-complete.json',dict(time=time.time()))


def launch(c, config):
    prepare(c)
    out,_,_ = roots(c)
    code = out/'technical/code'
    revision = 1
    while code.exists() and any(sha(code/f)!=sha(REPO/f) for f in
            ('src/research/nucleus_recovery.py','src/research/nucleus_recovery_md.py')):
        revision += 1
        code = out/f'technical/code-v{revision}'
    if not code.exists():
        ExecutionBundle.freeze(REPO,code,c,directories=('src','docs/metrics','configs'),
            files=[(REPO/'machine.local.yaml','machine.local.yaml'),(REPO/'scripts/convert_trajectory.py','scripts/convert_trajectory.py')])
    if read(code/'config.json') != c:raise ValueError('Frozen recipe differs')
    env = os.environ.copy()
    env.update(PCM_PROJECT_ROOT=str(REPO),CUDA_HOME='/usr/local/cuda',MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='12.0',
        TORCHINDUCTOR_COMPILE_THREADS='2',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONUNBUFFERED='1')
    env['PATH'] = '/usr/local/cuda/bin:'+str(Path(sys.executable).parent)+':'+env['PATH']
    receipt = dict(owner='current response/nucleus-recovery chat; only listed worker PIDs are owned',
        allocation=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,started=time.time(),workers=[],config=str(config))
    for lane in range(c['lanes']):
        log = out/f'technical/lane-{lane}.log'
        with log.open('a') as stream:
            p = subprocess.Popen([sys.executable,'-u','-m','src.research.nucleus_recovery','worker','--config',str(code/'config.json'),'--lane',str(lane)],
                cwd=code,env={**env,'CUDA_VISIBLE_DEVICES':str(lane)},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['workers'].append(dict(pid=p.pid,gpu=lane,log=str(log)))
    write_json(out/f'technical/launch-{int(receipt["started"])}.json',receipt)
    print(json.dumps(receipt,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','launch','worker','summarize'])
    parser.add_argument('--config',required=True);parser.add_argument('--lane',type=int)
    args = parser.parse_args();c=read(args.config)
    if args.stage=='launch':launch(c,args.config)
    elif args.stage=='prepare':prepare(c)
    elif args.stage=='worker':worker(c,args.lane)
    else:print(json.dumps(summarize(c),indent=2))


if __name__=='__main__':main()
