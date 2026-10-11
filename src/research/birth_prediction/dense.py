"""Detached native-cadence event search, descriptor preparation and nested readouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path

import numpy as np
import yaml

from src.data.fixed_cohort.protocol import digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.postprocessing.protocols import external_data
from . import dense_data as data
from .dense_data import FAMILY
from .appearance_cv import balance, feature_identity, support
from .data import read
from .merged_fit import fit


def initialize(path):
    global PLAN
    PLAN = data.load(read(path))


def worker_source(item):
    with recorded_stage(resolve_path(PLAN['config']['output'])/f'technical/sources/{item["id"]}/stage.json',
                        job=os.environ.get('SLURM_JOB_ID')):
        return data.source(PLAN, item)


def sources(path, index, progress):
    c = read(path); p = data.load(c)
    selected = [s for s in p['sources'] if s['lane'] == index]
    with ProcessPoolExecutor(max_workers=c['workers_per_lane'],
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(path,)) as pool:
        futures = [pool.submit(worker_source, s) for s in selected]
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            progress.update(completed=number, total=len(selected), last=result)
            print(result, flush=True)


def seal(c):
    p = data.load(c); rows, manifest = data.seal(c,p)
    groups = rows['ancestor']
    # This new native-cadence event assay never imports sparse-grid event labels
    # or modifies the historical fixed-Al release or its role assignments.
    outer = balance(rows, groups, np.ones(len(groups), bool), c['outer_folds'], c['seed'])
    inner = {}
    for fold in range(c['outer_folds']):
        train = np.asarray([outer[g] != fold for g in groups])
        inner[str(fold)] = balance(rows, groups, train, c['folds'], c['seed'])
        assigned = np.asarray([inner[str(fold)][g] for g in groups[train]])
        if set(rows['label'][~train]) != {0,1}:
            raise ValueError(f'Outer fold {fold} lacks a class')
        for k in range(c['folds']):
            for mask in (assigned == k, assigned != k):
                if set(rows['label'][train][mask]) != {0,1}:
                    raise ValueError(f'Inner fold {fold}/{k} lacks a class')
    design = dict(plan=p['identity'], dataset=manifest['identity'], outer=outer, inner=inner,
                  recipe=p['recipe'], context=manifest['input_context'])
    design['identity'] = digest(design)
    path = resolve_path(c['output'])/'technical/design.json'
    if path.exists() and read(path) != design:
        raise ValueError('Nested design changed')
    write_json(path, design)
    write_metric_rows(support(rows,groups,outer,c['outer_folds']),
        resolve_path(c['output'])/'analyses/coverage-v1', family=FAMILY, name='fold-support')
    return design


def packet(frames, observation):
    if frames.ndim != 3 or frames.shape[1:] != (6,442):
        raise ValueError(f'Expected six 442-descriptor observations, got {frames.shape}')
    if observation == 'snapshot': return frames[:,-1].copy()
    if observation == 'early_snapshot': return frames[:,0].copy()
    if observation == 'repeated6': frames = np.repeat(frames[:,-1:],6,axis=1)
    elif observation != 'history6': raise ValueError(observation)
    return np.concatenate((frames.reshape(len(frames),-1), frames.mean(1), frames.std(1),
                           frames[:,-1]-frames[:,0]),axis=1)


def tasks(c):
    for model, arm in (('linear','rich_linear'),('catboost','rich_gbdt')):
        for observation in c['observations']:
            yield dict(model=model, arm=arm, observation=dict(name=observation))


def fit_root(c,spec,fold):
    return (resolve_path(c['output'])/'analyses/fits-v1'/spec['arm']/
            spec['observation']['name']/f'outer-{fold}')


def design_for(c):
    p = data.load(c)
    design = read(resolve_path(c['output'])/'technical/design.json')
    if (design['plan'] != p['identity']
            or digest({k:v for k,v in design.items() if k != 'identity'}) != design['identity']):
        raise ValueError('Changed native-cadence fold design')
    return design


def fit_all(c,progress):
    design = design_for(c)
    rows, frames, manifest = data.dataset(c)
    if manifest['identity'] != design['dataset']: raise ValueError('Changed dense dataset')
    groups = rows['ancestor']
    for spec in tasks(c):
        obs = spec['observation']['name']; x = packet(frames,obs)
        for fold in range(c['outer_folds']):
            train = np.asarray([design['outer'][g] != fold for g in groups])
            local = dict(rows, role=np.where(train,'train','outer_test'))
            assignment = np.asarray([design['inner'][str(fold)][g] for g in groups[train]])
            offsets = (c['offsets_ps'] if obs == 'history6' else
                       [c['offsets_ps'][0]] if obs == 'early_snapshot' else [c['offsets_ps'][-1]])
            predictor = dict(design['context']['predictor'], observation=obs, offsets_ps=offsets,
                repeated_current_copies=6 if obs == 'repeated6' else 0,
                temporal_summary='chronological descriptors, mean, std, last-minus-first' if obs in ('history6','repeated6') else None)
            context = dict(design['context'], predictor=predictor,
                future_selected_sites=True, target='any local crystal appearance, including verified failed embryos',
                selector='inner ancestry-held-out binary NLL', calibration='none',
                features=feature_identity(x), design=design['identity'], outer_fold=fold)
            progress.update(arm=spec['arm'], observation=obs, outer_fold=fold)
            root = fit_root(c,spec,fold)
            fit(c,design['recipe'],local,x,spec,root,assignment,context,family=FAMILY)
            write_json(root/'technical/outer-split.json',dict(design=design['identity'],
                outer_fold=fold, original_role=rows['role'].tolist(),
                train_prior=float(np.average(rows['label'][train],weights=rows['weight'][train]))))
    from .dense_analysis import collect
    collect(c,design,progress)


def submit(path):
    c = read(path); p = data.prepare(c)
    technical = resolve_path(c['output'])/'technical'
    if (technical/'launch.json').exists(): raise ValueError('Already submitted; use the frozen stage to resume')
    bundle = ExecutionBundle.freeze(REPO,technical/'code',c,directories=('src','configs','docs/metrics'))
    machine = yaml.safe_load(Path(os.environ.get('PCM_MACHINE_CONFIG',REPO/'machine.local.yaml')).read_text())
    machine['catalog'] = str(bundle.root/'configs/datasets.json')
    (technical/'machine.yaml').write_text(yaml.safe_dump(machine))
    record = dict(plan=p['identity'], code=str(bundle.root), sources=len(p['sources']),
                  outer_fits=len(list(tasks(c)))*c['outer_folds'], jobs={})
    queue = SlurmQueue(technical,bundle,'src.research.birth_prediction.dense',
        dict(PCM_PROJECT_ROOT=str(bundle.root),PCM_MACHINE_CONFIG=str(technical/'machine.yaml'),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OVITO_THREAD_COUNT='1',QT_QPA_PLATFORM='offscreen'), technical/'launch.json',record,'BIRTH-010')
    with queue.submission():
        ptm = queue.submit('ptm',[f'--array=0-{c["ptm_lanes"]-1}',
            f'--cpus-per-task={c["ptm_workers_per_lane"]}','--mem=16G',f'--time={c["time_limit"]}'])
        search = queue.submit('sources',[f'--array=0-{c["lanes"]-1}',
            f'--cpus-per-task={c["workers_per_lane"]}','--mem=32G',f'--time={c["time_limit"]}'],dependency='afterok:'+ptm)
        sealing = queue.submit('seal',['--cpus-per-task=1','--mem=16G','--time=01:00:00'],dependency='afterok:'+search)
        queue.submit('fit',['--cpus-per-task=8','--mem=24G','--gpus=1','--time=08:00:00',
            '--exclude='+c['excluded_gpu_node']],dependency='afterok:'+sealing,partition=c['gpu_partition'])
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','submit','ptm','sources','seal','fit','collect'))
    parser.add_argument('--config',required=True)
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'prepare': print(data.prepare(c)['identity']); return
    if args.stage == 'submit': print(submit(args.config)); return
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    with recorded_stage(resolve_path(c['output'])/f'technical/{args.stage}-{index}.json',
                        job=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename) as progress:
        if args.stage == 'ptm':
            p = data.load(c)
            # The shared PTM launcher expects one worker count, separate from
            # the four per-source graph/descriptor processes used afterwards.
            selected = [t for t in p['tasks'] if t['lane'] == index]
            with ProcessPoolExecutor(max_workers=c['ptm_workers_per_lane'],
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=external_data.initialize_worker,
                    initargs=(str(resolve_path(c['output'])/'technical/plan.json'),)) as pool:
                futures = [pool.submit(external_data.chunk_worker,t) for t in selected]
                for number,future in enumerate(as_completed(futures),1):
                    result = future.result()
                    progress.update(completed=number,total=len(selected),last=result)
        elif args.stage == 'sources': sources(args.config,index,progress)
        elif args.stage == 'seal': print(seal(c)['identity'])
        elif args.stage == 'fit': fit_all(c,progress)
        else:
            from .dense_analysis import collect
            collect(c,design_for(c),progress)


if __name__ == '__main__': main()
