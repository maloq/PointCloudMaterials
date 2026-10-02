"""Explicitly authorized merging of former selection/calibration/test source roles."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import load, read
from .feature_families import domain_config, variants, FAMILIES
from .features import encode, load_bank
from .merged_fit import FAMILY, fit
from .temporal import design
from .temporal_inputs import context, packet


def base(c, domain):
    b = domain_config(c, domain)
    b['feature_cache'] = c['feature_cache']
    b['output'] = str(resolve_path(c['output']) / 'technical/input-exports' / domain['name'])
    return b


def treatments(c):
    b = domain_config(c, c['domains'][0])
    arms = {a['name']: a for a in b['arms'] if a['model'] != 'prior'}
    specs, seen = [], set()
    def add(arm, observation, group):
        numerical = dict(bank=arm['bank'], model=arm['model'], families=arm['families'],
                         observation={k:v for k,v in observation.items() if k != 'name'})
        key = digest(numerical)
        if key not in seen:
            seen.add(key)
            specs.append(dict(arm=arm['name'], model=arm['model'], bank=arm['bank'],
                families=arm['families'], observation=observation, group=group))
    current = dict(name='current', kind='snapshot', frames=[7])
    # Feature-family comparisons are first in both execution lanes.
    for variant in variants():
        for model, name in [('linear','rich_linear'), ('catboost','rich_gbdt')]:
            arm = arms[name] if variant['name']=='full' else dict(
                name=f'{model}_{variant["name"]}', model=model, bank='descriptors', families=variant['families'])
            add(arm, current, 'feature_families')
    if c['coverage'] == 'all_birth_comparisons':
        for name in c['temporal_arms']:
            for observation in c['observations']:
                add(arms[name], observation, 'temporal')
        for arm in arms.values():
            for length in range(8, 0, -1):
                add(arm, dict(name=f'truncate_to_{length}', kind='history', frames=list(range(length))), 'truncation')
    elif c['coverage'] != 'feature_families':
        raise ValueError(f'Unknown comparison coverage: {c["coverage"]}')
    return specs


def fit_root(c, domain, spec):
    return resolve_path(c['output']) / 'analyses/fits-v1' / domain['name'] / spec['arm'] / spec['observation']['name']


def training_folds(c, rows):
    record = design(c)
    train = np.flatnonzero(rows['role']=='train')
    folds = np.array([record['binding']['source_folds'][str(int(s))] for s in rows['source'][train]])
    for fold in range(c['folds']):
        a = train[folds != fold]; b = train[folds == fold]
        if set(rows['source'][a]) & set(rows['source'][b]) or set(rows['label'][b]) != {0,1}:
            raise ValueError(f'Invalid internal source fold {fold}')
    return folds


def prepare(c):
    check_metric_docs(family=FAMILY)
    specs = treatments(c)
    support, assets, original = [], [], None
    for domain in c['domains']:
        b = base(c, domain)
        _, rows, manifest = load(b)
        if manifest['identity'] != domain['dataset_identity']:
            raise ValueError('Changed input release')
        if original is None:
            original = rows
        elif rows.keys()!=original.keys() or any(not np.array_equal(v,original[k]) for k,v in rows.items()):
            raise ValueError('Original and relaxed domains are not the exact same cohort')
        folds = training_folds(c, rows)
        plan = read(resolve_path(b['cache'])/'plan.json') if domain['name']=='original' else None
        if plan is not None:
            roles = {}
            for source in plan['sources']:
                role = 'train' if source['role']=='train' else 'merged_test'
                lineage = source['lineage']
                if lineage in roles and roles[lineage] != role:
                    raise ValueError(f'Ancestry crosses training/merged test: {lineage}')
                roles[lineage] = role
        for role, take in [('train',rows['role']=='train'),('merged_test',rows['role']!='train')]:
            positive = take & (rows['label']==1)
            support.append(dict(domain=domain['name'], role=role, rows=int(take.sum()),
                positive=int(positive.sum()), negative=int((take & (rows['label']==0)).sum()),
                sources=len(np.unique(rows['source'][take])), births=len(set(zip(rows['source'][positive],rows['event'][positive])))))
        root = resolve_path(b['cache'])
        assets.append(dict(domain=domain['name'], dataset_identity=manifest['identity'],
            rows_sha256=sha(root/'rows.npz'), descriptors_sha256=sha(root/'descriptors.npy'),
            descriptor_manifest_sha256=sha(root/'descriptor-manifest.json')))
    record = dict(config=c, support=support, assets=assets, treatments=specs,
        final_readouts=len(specs)*len(c['domains']), source_fold_identity=c['fold_identity'],
        original_role_mapping={'train':'train','selection':'merged_test','calibration':'merged_test','test':'merged_test'},
        previously_inspected=True, no_probability_calibration=True)
    record['identity'] = digest(record)
    path = resolve_path(c['output'])/'technical/prepared.json'
    if path.exists() and read(path)!=record:
        raise ValueError('Merged protocol changed; create a new version')
    write_json(path,record)
    write_metric_rows(support,resolve_path(c['output'])/'analyses/split-v1',family=FAMILY,name='support')
    return record


@lru_cache(maxsize=6)
def resident(config_text, domain_name, bank_name):
    c = json.loads(config_text)
    domain = next(d for d in c['domains'] if d['name']==domain_name)
    b = base(c, domain)
    _, rows, _ = load(b)
    bank, columns = load_bank(b, bank_name)
    return b, rows, bank, columns


def task(c, domain, spec):
    b, rows, bank, columns = resident(json.dumps(c,sort_keys=True), domain['name'], spec['bank'])
    selected = np.array([i for i,col in enumerate(columns) if spec['bank']!='descriptors'
                         or col.split('/')[0] in spec['families']])
    if not len(selected):raise ValueError(f'Empty input columns: {spec}')
    x = packet(bank,rows,selected,spec['observation'],c['seed'])
    encoder = next((m for m in b['encoders'] if m['name']==spec['bank']), None)
    inputs = dict(encoder=encoder, predictor=dict(model=spec['model'], bank=spec['bank'],
        families=spec['families'], dimensions=x.shape[1], descriptor_columns=[columns[i] for i in selected],
        spatial_support='stored nearest-80 centered patch; rich descriptors resort and clip at 8 A',
        **context(spec['observation'],b['cadence_ps'])),
        domain=domain['name'], input_domain=b.get('input_domain'),
        relaxation=domain['name']=='relaxed', encoder_history_frames=1 if encoder else 0,
        motion='descriptor or embedding differences only; no velocities or tracked displacement',
        conditions=[], species_input=False, probability_calibration='none',
        selector='five source folds entirely within original training; refit on all original training rows',
        role_mapping='former selection/calibration/test are one evaluation set; parent data unchanged',
        encoder_selection_exposure=c['encoder_selection_exposure'].get(spec['bank']),
        tracking='local descriptor controls and frozen readouts', seed=c['seed'])
    fit(c,b,rows,x,spec,fit_root(c,domain,spec),training_folds(c,rows),inputs,family=FAMILY)


def submit(path):
    c=read(path); prepared=prepare(c)
    root=resolve_path(c['output'])/'technical'
    if (root/'launch.json').exists():raise ValueError('Already submitted; resume frozen stages')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,root/'code',c,directories=('src','docs/metrics','configs'))
    record=dict(protocol=c['protocol'],jobs={},code=str(bundle.root),prepared_identity=prepared['identity'],
                final_readouts=prepared['final_readouts'],probability_calibration='none')
    queue=SlurmQueue(root,bundle,'src.research.birth_prediction.merged',
        dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             NUMBA_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),root/'launch.json',record,'BIRTH-MERGED')
    with queue.submission():
        ready=queue.submit('encode',['--gpus=1','--cpus-per-task=4','--mem=12G','--time=01:00:00'],partition=c['gpu_partition'])
        cpu=queue.submit('linear',[f'--array=0-{c["cpu_workers"]-1}','--cpus-per-task=8','--mem=16G','--time=08:00:00'],dependency='afterok:'+ready)
        gpu=queue.submit('boost',[f'--array=0-{c["gpu_workers"]-1}','--gpus=1','--cpus-per-task=8','--mem=16G','--time=08:00:00'],dependency='afterok:'+ready,partition=c['gpu_partition'])
        queue.submit('collect',['--cpus-per-task=4','--mem=12G','--time=02:00:00',
            f'--nodelist={c["collection_node"]}'],dependency='afterok:'+cpu+':'+gpu)
    return record


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('stage',choices=['prepare','submit','encode','linear','boost','collect'])
    p.add_argument('--config',required=True)
    p.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=p.parse_args(); c=read(args.config)
    if args.stage=='submit':print(json.dumps(submit(args.config),indent=2));return
    with recorded_stage(resolve_path(c['output'])/f'technical/{args.stage}-{args.index}.json',job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage=='prepare':progress.update(identity=prepare(c)['identity'])
        elif args.stage=='encode':
            for domain in c['domains']:
                b=base(c,domain)
                for i,m in enumerate(b['encoders']):
                    if any(s['bank']==m['name'] for s in treatments(c)):
                        progress.update(domain=domain['name'],encoder=m['name'])
                        encode(b,i)
        elif args.stage in ('linear','boost'):
            model='linear' if args.stage=='linear' else 'catboost'
            tasks=[(d,s) for d in c['domains'] for s in treatments(c) if s['model']==model]
            workers=c['cpu_workers'] if model=='linear' else c['gpu_workers']
            for i in range(args.index,len(tasks),workers):
                d,s=tasks[i];progress.update(task=i,total=len(tasks),domain=d['name'],arm=s['arm'],observation=s['observation']['name'])
                task(c,d,s)
        else:
            from .merged_analysis import collect, explain
            collect(c,progress)
            explain(c,progress)


if __name__=='__main__':main()
