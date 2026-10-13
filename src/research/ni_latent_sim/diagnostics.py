"""Bounded, restartable diagnostics of frozen Ni stochastic simulators.

Run with python -m src.research.ni_latent_sim.diagnostics COMMAND --config FILE.
This workflow never launches MD or changes existing training jobs/checkpoints.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import write_metric_rows, check_metric_docs
from .data import read, root, cache, sha, digest, write_json, atomic_save, load, parent
from .model import make_model
from .path_scores import (noise_tape, emission_noise, rollout, scramble, joint_scores,
                          endpoint_scores, weighted_summary, check)

FAMILY = 'ni_motion_diagnostics'


def selected(c):
    return read(root(c)/'technical/plan.json')


def model_for(c, spec, device='cuda'):
    checkpoint = root(c)/'technical/checkpoints'/(spec['arm']['name']+'.pt')
    if sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Changed pinned audit checkpoint: '+str(checkpoint))
    model = make_model(spec['arm']).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    return model


def prepare(c):
    out = root(c); path = out/'technical/plan.json'
    if path.exists():
        p = read(path)
        if p['config'] != c:
            raise ValueError('Changed diagnostic protocol; use a new run')
        return p
    source = Path(c['source_run'])/'technical/plan.json'
    if sha(source) != c['source_plan_sha256']:
        raise ValueError('Changed source experiment plan')
    p = read(source); pc = p['config']; data = load(pc, 'data')
    receipt = cache(pc)/'data.json'
    if sha(receipt) != pc['reuse_data']['receipt_sha256']:
        raise ValueError('Changed reused sequence contract')
    rows = data['rows']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sequence identities')
    roles = {}
    for row in rows:
        roles.setdefault(row['ancestor'], set()).add(row['role'])
    if any(len(v) != 1 for v in roles.values()):
        raise ValueError('Ancestry crosses diagnostic fit/selection/test roles')
    groups = {}
    for i, row in enumerate(rows):
        if row['role'] == 'test':
            groups.setdefault((row['ancestor'],row['regime']),[]).append(i)
    indices = []
    for (ancestor, regime), candidates in sorted(groups.items()):
        candidates.sort(key=lambda i: digest([c['seed'], rows[i]['id'], 'panel']))
        if len(candidates) < c['rows_per_ancestor_regime']:
            raise ValueError('Insufficient rows in existing diagnostic stratum: '+ancestor+' '+regime)
        indices.extend(candidates[:c['rows_per_ancestor_regime']])
    absent_strata = [dict(ancestor=a,regime=r) for a in sorted(roles) if roles[a]=={'test'}
                     for r in c['regimes'] if (a,r) not in groups]
    checkpoints = []
    for name in c['models']:
        arm = next(a for a in pc['arms'] if a['name'] == name)
        folder = Path(c['source_run'])/'analyses'/name/'technical'
        complete = read(folder/'complete.json'); expected = complete['checkpoint_sha256']
        if sha(folder/'best.pt') != expected:
            raise ValueError('Selected source checkpoint changed: '+name)
        saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        if saved['identity'] != digest([p['identity'], arm]):
            raise ValueError('Wrong checkpoint training identity')
        dest = out/'technical/checkpoints'/(name+'.pt'); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder/'best.pt', dest)
        if sha(dest) != expected:
            raise ValueError('Checkpoint copy changed while reading')
        checkpoints.append(dict(arm=arm, checkpoint_sha256=expected, training=complete))
    probe_checkpoints = list(checkpoints)
    if 'reference_audit' in c:
        reference = c['reference_audit']; reference_root = Path(reference['path'])
        reference_plan = reference_root/'technical/plan.json'
        if sha(reference_plan) != reference['plan_sha256']:
            raise ValueError('Changed reference audit')
        prior = read(reference_plan)
        if prior['data_files'] != read(receipt)['files'] or prior['panel_rows'] != [rows[i] for i in indices]:
            raise ValueError('Follow-up must preserve the exact data and path panel')
        for spec in prior['checkpoints']:
            source_checkpoint = reference_root/'technical/checkpoints'/(spec['arm']['name']+'.pt')
            if sha(source_checkpoint) != spec['checkpoint_sha256']:
                raise ValueError('Changed reference checkpoint')
            destination = out/'technical/checkpoints'/source_checkpoint.name
            shutil.copy2(source_checkpoint, destination)
            probe_checkpoints.append(spec)
    from .dense_response import bank
    teachers = bank(pc, p)
    anchors = []
    for ancestor in sorted({r['ancestor'] for r in teachers}):
        pool = sorted((r for r in teachers if r['ancestor'] == ancestor),
                      key=lambda r: digest([c['seed'], r['job'], 'response']))
        anchors.append(pool[0]['job'])
    normalizer = out/'technical/normalizer.pt'
    atomic_save(normalizer, dict(mean=data['mean'], scale=data['scale']))
    contract = dict(config=c, source_config=pc, source_identity=p['identity'],
                    data_receipt_sha256=sha(receipt), data_files=read(receipt)['files'],
                    rows=len(rows), roles=dict(Counter(r['role'] for r in rows)),
                    panel_indices=indices, panel_rows=[rows[i] for i in indices],
                    absent_source_regimes=absent_strata,
                    response_jobs=anchors, checkpoints=checkpoints, probe_checkpoints=probe_checkpoints,
                    normalizer_sha256=sha(normalizer),
                    time=dict(origin='t0', native_spacing_ps=.1, history_ps=[-.3,-.2,-.1,0],
                              future_ps=[i/10 for i in range(1,61)],
                              physical_integration_fs=1, teacher_sampling_fs=5, neural_step_fs=5),
                    observation='geometry128 + velocity-secant128 + smooth24; no temperature/age/species inputs',
                    kernels=dict(native='NPT, refreshed nearest80 at each saved time',
                                 response='full-cell fixed-box BAOAB, retained initial80 atom identities'),
                    precision=read(Path(c['source_run'])/'technical/prediction-context.json'),
                    perturbation=dict(settings=parent(pc)['config']['teacher'],
                                      epsilon=parent(pc)['config']['student_difference'],
                                      slots='last history slot only; earlier measured history unchanged'),
                    sampling='future-conditioned regime panel; not a natural prevalence estimate',
                    inference='one autonomous rollout, no teacher reset; float32 recurrence and scores',
                    streams='CPU row/particle/stream SHA keys, nested K prefixes; no scientific W&B run',
                    stages=c.get('stages', ['paths', 'response', 'history']))
    contract['identity'] = digest(contract)
    write_json(path, contract)
    write_json(out/'technical/integration-check.json', check())
    print(json.dumps(dict(rows=len(indices), ancestors=len({rows[i]['ancestor'] for i in indices}),
                         physical_anchors=len(anchors), checkpoints=len(checkpoints))), flush=True)
    return contract


def export(c, name, records):
    write_metric_rows(records, root(c)/'analyses'/name, family=FAMILY, name='metrics')


def summarize(c, name, values, records, context):
    result = []
    for key, columns in values.items():
        for row in weighted_summary(columns, records, c['seed']):
            result.append(dict(context, metric=key, **row))
    export(c, name, result)
    return result


def paths(c, model_index=None):
    p = selected(c); data = load(p['source_config'], 'data')
    ids = p['panel_indices']; records = p['panel_rows']; particles = max(c['particles'])
    x = (data['x']-data['mean'])/data['scale']
    scale = data['scale'][256:].cuda(); mean = data['mean'][256:].cuda()
    checkpoints = p['checkpoints'] if model_index is None else [p['checkpoints'][model_index]]
    for spec in checkpoints:
        name = spec['arm']['name']; model = model_for(c, spec)
        for key in c['outer_keys']:
            folder = root(c)/'analyses'/f'paths-{name}-r{key}'
            all_scores = {}; begin = time.monotonic()
            for start in range(0, len(ids), c['batch_size']):
                block = ids[start:start+c['batch_size']]; rows = records[start:start+len(block)]
                packet = folder/'technical/packets'/f'{start:05d}.pt'
                receipt = packet.with_suffix('.json')
                if receipt.exists():
                    if sha(packet) != read(receipt)['sha256']:
                        raise ValueError('Changed path diagnostic packet')
                    scores = torch.load(packet, weights_only=False, map_location='cpu')['scores']
                else:
                    xx = x[block, :4].cuda(); truth = x[block, 4:].cuda()
                    row_ids = [r['id'] for r in rows]
                    with torch.no_grad():
                        tape = noise_tape(model, row_ids, key, particles, 60, 'cuda')
                        mu, lv = rollout(model, xx, tape)
                        del tape
                        samples = mu+(.5*lv).exp()*emission_noise(row_ids, key, particles, 60, 280, 'cuda')
                        if not all(torch.isfinite(v).all() for v in [mu, lv, samples]):
                            raise FloatingPointError('Nonfinite predictive path: '+name+' '+str(start))
                        scores = {}
                        def retain(prefix, metrics):
                            scores.update({prefix+'/'+m: v.cpu().numpy() for m, v in metrics.items()})
                        for k in c['particles']:
                            for h in [1,5,10,30,60]:
                                for space, sl in [('structure',slice(0,128)),('motion',slice(128,256)),
                                                  ('descriptors',slice(256,280))]:
                                    retain(f'K{k}/endpoint/{space}/{h/10:g}ps', endpoint_scores(
                                        mu[:,:k,h-1,sl], lv[:,:k,h-1,sl], samples[:,:k,h-1,sl], truth[:,h-1,sl]))
                            triplet = (mu[:,:k,:,256:], lv[:,:k,:,256:], samples[:,:k,:,256:])
                            shuffled = scramble(triplet, row_ids, key)
                            for component, index in [('conditional_means',0),('emitted',2)]:
                                original = joint_scores(triplet[index], truth[:,:,256:], xx[:,-1,256:])
                                permuted = joint_scores(shuffled[index], truth[:,:,256:], xx[:,-1,256:])
                                retain(f'K{k}/{component}/connected', original)
                                retain(f'K{k}/{component}/scrambled', permuted)
                                retain(f'K{k}/{component}/scramble_gap', {m:permuted[m]-v for m,v in original.items()})
                            retain(f'K{k}/support', dict(negative_descriptor_rate=(triplet[2]*scale+mean<0).float().mean((1,2,3)),
                                                       nonfinite_rate=(~torch.isfinite(triplet[2])).float().mean((1,2,3))))
                            for horizon in [10,30,60]:
                                peak = triplet[0][:,:,:horizon].abs().flatten(2).amax(2)
                                error = (triplet[0][:,:,:horizon]-truth[:,None,:horizon,256:]).square().mean((2,3))
                                retain(f'K{k}/tails/{horizon/10:g}ps', dict(
                                    particle_peak_gt10=(peak>10).float().mean(1),
                                    particle_peak_gt100=(peak>100).float().mean(1),
                                    particle_path_mse_gt100=(error>100).float().mean(1),
                                    any_particle_path_mse_gt100=(error>100).any(1).float()))
                        retain('persistence', dict(descriptor_path_mse=(xx[:,-1,None,256:]-truth[:,:,256:]).square().mean((1,2)),
                                                  path_energy=(xx[:,-1,None,256:]-truth[:,:,256:]).flatten(1).norm(dim=-1)/(60*24)**.5))
                    atomic_save(packet, dict(rows=rows, key=key, particles=particles, scores=scores,
                                            mu=mu[...,256:].cpu(), logvar=lv[...,256:].cpu(),
                                            samples=samples[...,256:].cpu(), truth=truth[...,256:].cpu(),
                                            initial=xx[:,-1,256:].cpu()))
                    write_json(receipt, dict(sha256=sha(packet), identity=p['identity']))
                for metric, value in scores.items():
                    all_scores.setdefault(metric, []).append(value)
                progress_name = 'path-progress.json' if model_index is None else f'path-progress-{model_index}.json'
                write_json(root(c)/'technical'/progress_name, dict(model=name, key=key,
                           completed_rows=start+len(block), total_rows=len(ids), seconds=time.monotonic()-begin))
                print('paths', name, key, start+len(block), '/', len(ids), flush=True)
            summarize(c, folder.name, {m:np.concatenate(v) for m,v in all_scores.items()}, records,
                      dict(model=name, outer_key=key))


def worker(c, stage, model_index=None):
    torch.set_num_threads(c['cpu_threads'])
    functions = {'paths':paths}
    if stage == 'response':
        from .response_diagnostics import run
        functions[stage] = run
    if stage == 'history':
        from .history_diagnostics import run
        functions[stage] = run
    if stage == 'predictability':
        from .response_predictability import run
        functions[stage] = run
    task = stage if model_index is None else f'{stage}-{model_index}'
    complete = root(c)/'technical'/f'{task}-complete.json'
    if complete.exists():
        return
    for attempt in [1, 2]:
        try:
            with recorded_stage(root(c)/'technical'/f'{task}-attempt-{attempt}.json',
                                stage=stage, job=os.environ.get('SLURM_JOB_ID')):
                if stage == 'paths':
                    paths(c, model_index)
                else:
                    functions[stage](c)
            write_json(complete, dict(state='complete', finished=time.time()))
            return
        except Exception:
            traceback.print_exc()
    raise RuntimeError(stage+' failed both bounded attempts; independent stages remain queued')


def submit(c):
    plan = prepare(c); check_metric_docs(family=FAMILY)
    repository = Path(__file__).resolve().parents[3]
    tech = root(c)/'technical'; receipt = tech/'launch.json'
    if receipt.exists():
        raise FileExistsError('Launch already exists: '+str(receipt))
    bundle = ExecutionBundle.freeze(repository, tech/'code', c, directories=['src','configs','docs/metrics'],
                                    files=[(repository/'machine.local.yaml','machine.local.yaml')])
    launch = dict(started=time.time(), owner='current chat: methods-plan diagnostics 2026-10-11',
                  independent=True, jobs={}, source_plan_sha256=sha(tech/'plan.json'))
    queue = SlurmQueue(tech, bundle, 'src.research.ni_latent_sim.diagnostics',
                       dict(OMP_NUM_THREADS=str(c['cpu_threads']), OPENBLAS_NUM_THREADS=str(c['cpu_threads']),
                            MKL_NUM_THREADS=str(c['cpu_threads']), PYTHONFAULTHANDLER='1',
                            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
                            TORCH_EXTENSIONS_DIR=str(tech/'cuda-build')),
                       receipt, launch, 'ni-audit')
    with queue.submission():
        options = ['--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=06:00:00']
        if 'path_workers' in c:
            options.append(f'--array=0-{len(plan["checkpoints"])-1}%{c["path_workers"]}')
        first = queue.submit('paths', options, partition=c['partitions'])
        for stage in plan['stages']:
            if stage == 'response':
                queue.submit(stage, ['--gres=gpu:1','--cpus-per-task=4','--mem=32G','--time=06:00:00'],
                             partition=c['partitions'], dependency='afterany:'+first)
            elif stage in ['history','predictability']:
                queue.submit(stage, ['--cpus-per-task=4','--mem=32G','--time=04:00:00'])
    print(json.dumps(launch), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','submit','check','paths','response','history','predictability'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--model-index', type=int)
    args = parser.parse_args(); c = read(args.config)
    if args.command == 'check':
        torch.set_num_threads(c['cpu_threads']); results = check()
        if 'probe' in c:
            from .response_predictability import check as probe_check
            results['directional_probe'] = probe_check()
        print(json.dumps(results))
    elif args.command == 'prepare':
        torch.set_num_threads(c['cpu_threads']); prepare(c)
    elif args.command == 'submit':
        submit(c)
    else:
        index = args.model_index
        if args.command == 'paths' and 'SLURM_ARRAY_TASK_ID' in os.environ:
            index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        worker(c, args.command, index)


if __name__ == '__main__':
    main()
