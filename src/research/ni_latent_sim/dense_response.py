"""Whole-100-fs physical paths and their propagated initial-state responses."""
import csv
import fcntl
import time
import numpy as np
import torch
from .artifacts import write_rows as write_metric_rows, table_path, export_root
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.ni_response import simulator
from .data import (read, root, cache, parent, load, sha, digest, write_json,
                   atomic_save, resolve_path, Motion, FAMILY)


def directory(c):
    return resolve_path(c['dense_response']['cache'])


def physical_config(c):
    pc = parent(c)['config']
    return dict(pc, simulation_archive=c['simulation_archive'],
                simulation_scratch=c['simulation_scratch'],
                teacher=dict(pc['teacher'], save_fs=c['dense_response']['save_fs']))


def teacher_plan(c, p):
    spec = c['dense_response']
    if 'reuse_output' not in spec:
        return p
    path = resolve_path(spec['reuse_output'])/'technical/plan.json'
    if sha(path) != spec['reuse_plan_sha256']:
        raise ValueError('Changed reused dense-teacher plan')
    original = read(path)
    for key in ['parent_identity', 'partition', 'jobs', 'sources']:
        if p[key] != original[key]:
            raise ValueError('Reused dense teachers differ in '+key)
    if (original['config']['dense_response']['save_fs'] != spec['save_fs']
            or directory(original['config']) != directory(c)):
        raise ValueError('Reused dense bank has different path or times')
    return original


def prepare(c):
    from .stability import prepare as reuse
    if 'stability' in c or any(a.get('jacobian', False) for a in c['arms']):
        raise ValueError('Dense-path assay separates physical responses from learned-map regularization')
    if c['dense_response']['save_fs'] != list(range(5, 101, 5)):
        raise ValueError('Declared dense physical path must retain all twenty 5-fs observations')
    if any(a['neural_step_fs'] != 5 or a['supervision'] not in ['endpoint', 'path'] for a in c['arms']):
        raise ValueError('All matched arms require the same 5-fs integrator')
    p = reuse(c)
    teacher_plan(c, p)
    pc = physical_config(c)
    if sha(resolve_path(pc['potential_file'])) != pc['potential_sha256']:
        raise ValueError('Changed Ni potential')
    context = read(root(c)/'technical/prediction-context.json')
    context.pop('stability', None)
    context.update(
        predictor='four 16-dimensional codes from native0.1ps inputs; present code advances every5fs; past slots shift only every100fs; no time/temperature/box/species covariates',
        teachers='300 original full-cell initial states, two BAOAB shots each; real values and analytic initial-Q/V responses at5,10,...,100fs; fixed-box separate from native NPT',
        dense_support='teacher follows the original80 atom IDs and central ID for100fs; long native observations recompute nearest80 around the same center',
        inference='autonomous5fs steps; no teacher injection, intermediate resets or physical forces at inference',
        normalization='unchanged original training-only280-feature mean/scale',
        physical_precision='float64 state, momentum, noise, tangents and cropped archive; float32 forces and frozen eSEN/JVP targets')
    if any(a.get('transition_kind') == 'transformer' for a in c['arms']):
        context.update(transition='two-block causal transformer over ordered four latent states; learned slot order, no physical time covariates; same compression/decoder and5fs update',
                       teachers='checksum-bound reuse of completed twenty-time physical target packets; no new physical MD')
    write_json(root(c)/'technical/prediction-context.json', context)
    return p


def encode_path(c, model, physical, original):
    pc = parent(c)['config']; factor = pc['length_factor']
    shots, frames = physical['future'].shape[:2]
    q = (physical['future'][..., :3]*factor).float().flatten(0, 1)
    dq = (physical['tangent'][..., :3, :]*factor).float().flatten(0, 1)
    values, responses = [], []
    for offset in range(0, len(q), c['dense_response']['target_microbatch']):
        xyz = q[offset:offset+c['dense_response']['target_microbatch']].cuda()
        directions = dq[offset:offset+len(xyz)].cuda(); derivatives = []
        for k in range(2):
            with torch.enable_grad(), torch.autocast('cuda', enabled=False):
                value, dv = torch.autograd.functional.jvp(model.geometry, xyz, directions[..., k], strict=True)
            derivatives.append(dv.detach().cpu())
        values.append(value.detach().cpu()); responses.append(torch.stack(derivatives, -1))
    target = torch.cat(values).reshape(shots, frames, 128).mean(0)
    response = torch.cat(responses).reshape(shots, frames, 128, 2).mean(0)
    q0 = (physical['state'][None, ..., :3]*factor).float().cuda()
    d0 = (physical['basis'][None, ..., :3, :]*factor).float().cuda()
    initial = []
    for k in range(2):
        with torch.enable_grad():
            _, derivative = torch.autograd.functional.jvp(model.geometry, q0, d0[..., k], strict=True)
        initial.append(derivative[0].detach().cpu())
    record = dict(original, target=target, response=response,
                  initial_response=torch.stack(initial, -1), horizons_fs=physical['horizons_fs'])
    if not all(torch.isfinite(record[name]).all()
               for name in ['history', 'pairs', 'target', 'response', 'initial_response']):
        raise FloatingPointError('Nonfinite dense physical feature/response: '+physical['job'])
    return record


def replay_errors(physical, previous):
    """Diagnostic only: replay discrepancies never reject an otherwise valid new path."""
    errors = {}
    for name in ['future', 'tangent']:
        for field, sl in [('q', slice(0, 3)), ('v', slice(3, 6))]:
            a, b = physical[name][:, -1, :, sl], previous[name][:, -1, :, sl]
            difference = a-b
            error = float(difference.norm()/b.norm().clamp_min(1e-20))
            errors[name+'_'+field] = dict(relative_l2=error, maximum_absolute=float(difference.abs().max()))
    errors['initial_state_equal'] = torch.equal(physical['state'], previous['state'])
    errors['initial_basis_equal'] = torch.equal(physical['basis'], previous['basis'])
    errors['diagnostic_only'] = True
    return errors


def collect(c, p, lane, deadline):
    launch = read(root(c)/'technical/launch.json')
    execution_started = launch.get('resumed_at', launch['started'])
    if 'reuse_output' in c['dense_response']:
        records = bank(c, p)
        write_json(root(c)/'technical'/f'dense-complete-{lane}.json',
                   dict(identity=p['identity'], groups=len(records), reused=True, execution_started=execution_started))
        return
    pc = physical_config(c); oldpc = parent(c)['config']
    model = Motion(oldpc).cuda().eval().requires_grad_(False)
    originals = {r['job']: r for r in load(c, 'short')}
    folder = directory(c)/'packets'; folder.mkdir(parents=True, exist_ok=True)
    # Both lanes claim individual anchors, avoiding source-level imbalance.
    for job in p['jobs']:
        key = job['key']; packet = folder/(key+'.pt'); receipt = folder/(key+'.json')
        with (folder/(key+'.lock')).open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            identity = digest([p['identity'], job, pc['teacher']])
            if receipt.exists():
                saved = read(receipt)
                if saved['identity'] != identity or sha(packet) != saved['sha256']:
                    raise ValueError('Changed dense response packet: '+key)
                continue
            if time.time() > deadline-600:
                raise TimeoutError('Dense teacher preparation reached deadline')
            begin = time.monotonic(); done = simulator.collect(pc, p, job, deadline-300)
            physical = torch.load(simulator.archive(pc, job)/'response.pt', map_location='cpu', weights_only=False)
            # Keep cross-device replay as an advisory numerical diagnostic.
            previous = simulator.archive(oldpc, job)
            try:
                oldreceipt = read(previous/'complete.json')
                if sha(previous/'response.pt') != oldreceipt['sha256']:
                    raise ValueError('Changed historical replay reference')
                old = torch.load(previous/'response.pt', map_location='cpu', weights_only=False)
                errors = replay_errors(physical, old)
            except Exception as error:
                # The new path has its own identity, finite-value and checksum
                # checks. An unavailable old comparison cannot block its use.
                errors = dict(diagnostic_only=True, unavailable=str(error))
                print('Replay diagnostic unavailable', key, str(error), flush=True)
            record = encode_path(c, model, physical, originals[key]); atomic_save(packet, record)
            write_json(receipt, dict(identity=identity, sha256=sha(packet), physical_sha256=done['sha256'],
                                    seconds=time.monotonic()-begin, endpoint_replay_normwise=errors))
            write_json(root(c)/'technical'/f'worker-{lane}.json', dict(stage='dense_path_teachers', job=key,
                       complete=len(list(folder.glob('*.json'))), total=len(p['jobs']), seconds=time.monotonic()-begin))
            print('dense teacher', key, time.monotonic()-begin, flush=True)
    write_json(root(c)/'technical'/f'dense-complete-{lane}.json',
               dict(identity=p['identity'], execution_started=execution_started))


def bank(c, p):
    p = teacher_plan(c, p)
    records = []
    pc = physical_config(c)
    for job in p['jobs']:
        file = directory(c)/'packets'/(job['key']+'.pt'); receipt = read(file.with_suffix('.json'))
        if receipt['identity'] != digest([p['identity'], job, pc['teacher']]) or sha(file) != receipt['sha256']:
            raise ValueError('Unsealed dense physical path: '+job['key'])
        records.append(torch.load(file, map_location='cpu', weights_only=False))
    return records


def predictions(model, records, mean, scale, seed):
    history = (torch.stack([r['history'] for r in records]).cuda()-mean)/scale
    def predict(h):
        generator = torch.Generator(device='cuda').manual_seed(seed)
        mu, _ = model.dense_rollout(h, 4, generator)
        return mu[..., :128].mean(1)
    return history, predict


def objective(model, records, mean, scale, arm, seed):
    history, predict = predictions(model, records, mean, scale, seed)
    target = (torch.stack([r['target'] for r in records]).cuda()-mean[:128])/scale[:128]
    slots = [-1] if arm['supervision'] == 'endpoint' else list(range(20))
    value = .5*(predict(history)[:, slots]-target[:, slots]).square().mean()
    response = value.new_zeros(())
    if arm['response']:
        pairs = (torch.stack([r['pairs'] for r in records]).cuda()-mean)/scale
        teacher = torch.stack([r['response'] for r in records]).cuda()/scale[:128, None]
        for k in range(2):
            minus, plus = history.clone(), history.clone()
            minus[:, -1] = pairs[:, k, 0]; plus[:, -1] = pairs[:, k, 1]
            derivative = predict(plus)-predict(minus)  # 2*epsilon=1, as in the parent.
            numerator = (derivative[:, slots]-teacher[:, slots, :, k]).square().mean((0, 2))
            denominator = teacher[:, slots, :, k].square().mean((0, 2)).clamp_min(1e-5)
            response = response + .25*(numerator/denominator).mean()
    return value, response


def evaluate(c, p, arm):
    from .evaluate import fitted, aggregate
    out = root(c)/'analyses'/arm['name']; done = out/'technical/dense-evaluation-complete.json'
    if done.exists():
        return
    model = fitted(c, arm); data = load(c, 'data')
    mean, scale = data['mean'].cuda(), data['scale'].cuda()
    held = [r for r in bank(c, p) if r['role'] == 'test']; ancestors = [r['ancestor'] for r in held]
    terms = {name: [] for name in ['value_mse', 'persistence_mse', 'response_q_mse', 'response_v_mse',
             'response_q_power', 'response_v_power', 'persistence_response_q_mse', 'persistence_response_v_mse']}
    with torch.no_grad():
        for offset in range(0, len(held), 16):
            records = held[offset:offset+16]; history, predict = predictions(model, records, mean, scale, c['seed']+offset)
            target = (torch.stack([r['target'] for r in records]).cuda()-mean[:128])/scale[:128]
            terms['value_mse'].append((predict(history)-target).square().mean(-1).cpu().numpy())
            terms['persistence_mse'].append((history[:, -1, None, :128]-target).square().mean(-1).cpu().numpy())
            pairs = (torch.stack([r['pairs'] for r in records]).cuda()-mean)/scale
            response = torch.stack([r['response'] for r in records]).cuda()/scale[:128, None]
            initial = torch.stack([r['initial_response'] for r in records]).cuda()/scale[:128, None]
            for k, name in enumerate(['q', 'v']):
                minus, plus = history.clone(), history.clone(); minus[:, -1] = pairs[:, k, 0]; plus[:, -1] = pairs[:, k, 1]
                derivative = predict(plus)-predict(minus)
                terms[f'response_{name}_mse'].append((derivative-response[..., k]).square().mean(-1).cpu().numpy())
                terms[f'response_{name}_power'].append(response[..., k].square().mean(-1).cpu().numpy())
                terms[f'persistence_response_{name}_mse'].append((initial[:, None, :, k]-response[..., k]).square().mean(-1).cpu().numpy())
    rows = []
    arrays = {k: np.concatenate(v) for k, v in terms.items()}
    for metric, values in arrays.items():
        for j, fs in enumerate(c['dense_response']['save_fs']):
            value, lo, hi = aggregate(values[:, j], ancestors, c['seed'])
            rows.append(dict(model=arm['name'], metric=metric, horizon_fs=fs, value=value, lo=lo, hi=hi,
                             sources=len(set(ancestors)), rows=len(held)))
    # Ratio-of-source-balanced-means; raw powers permit diagnosing weak targets.
    for name in ['q', 'v']:
        for fs in c['dense_response']['save_fs']:
            scores = {r['metric']: r['value'] for r in rows if r['horizon_fs'] == fs}
            power = scores[f'response_{name}_power']
            rows.append(dict(model=arm['name'], metric=f'zero_relative_response_{name}', horizon_fs=fs,
                             value=power/max(power, 1e-5), lo=None, hi=None,
                             sources=len(set(ancestors)), rows=len(held)))
            for prefix in ['', 'persistence_']:
                value = scores[f'{prefix}response_{name}_mse']/max(scores[f'response_{name}_power'], 1e-5)
                rows.append(dict(model=arm['name'], metric=f'{prefix}relative_response_{name}', horizon_fs=fs,
                                 value=value, lo=None, hi=None, sources=len(set(ancestors)), rows=len(held)))
    write_metric_rows(rows, out, family=FAMILY, name='dense_physical_path', columns=list(rows[0]))
    atomic_save(out/'technical/dense-scores.pt', dict(ancestors=ancestors, horizons_fs=c['dense_response']['save_fs'], metrics=arrays))
    write_json(done, dict(state='complete', groups=len(held)))
    update_recorded_summary(out/'technical/wandb.json',
        {f"dense/{r['horizon_fs']}fs/{r['metric']}": r['value'] for r in rows},
        evaluation='dense-physical-path-v1', expected=dict(mode='online'))


def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = export_root(root(c)/'analyses/dense-comparison-v1'); tables = {}
    for name in ['dense_physical_path', 'free_rollout', 'free_rollout_by_regime', 'nucleus_prediction']:
        rows = []
        for arm in c['arms']:
            with table_path(root(c)/'analyses'/arm['name'], name).open() as stream:
                rows.extend(dict(r, treatment=arm['treatment'], seed=arm['seed']) for r in csv.DictReader(stream))
        write_metric_rows(rows, out, family=FAMILY, name=name, columns=list(rows[0])); tables[name] = rows
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ['value_mse', 'relative_response_q', 'relative_response_v']):
        for treatment in sorted({a['treatment'] for a in c['arms']}):
            curves = []
            for seed in c['seeds']:
                selected = [r for r in tables['dense_physical_path'] if r['treatment'] == treatment and r['seed'] == seed and r['metric'] == metric]
                selected.sort(key=lambda r: int(r['horizon_fs'])); curves.append([float(r['value']) for r in selected])
            a = np.array(curves); t = c['dense_response']['save_fs']
            ax.plot(t, a.mean(0), label=treatment); ax.fill_between(t, a.min(0), a.max(0), alpha=.15)
        if metric.startswith('relative_response'):
            reference = [r for r in tables['dense_physical_path'] if r['model'] == c['arms'][0]['name']
                         and r['metric'] == 'zero_'+metric]
            reference.sort(key=lambda r: int(r['horizon_fs']))
            ax.plot(c['dense_response']['save_fs'], [float(r['value']) for r in reference],
                    color='gray', linestyle=':', label='Zero-response baseline')
        ax.set_xlabel('Physical future (fs)'); ax.set_ylabel(metric); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.suptitle('Held-out physical paths; shading is two-seed range')
    plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True); fig.tight_layout()
    fig.savefig(plots/'dense-path.png', dpi=170); fig.savefig(plots/'dense-path.pdf'); plt.close(fig)
    write_json(out/'technical/complete.json', dict(state='complete', models=len(c['arms'])))
    if 'architecture_reference' in c:
        architecture_report(c)


def architecture_report(c):
    """Paired transformer/MLP diagnostics; each model retains its own metrics."""
    spec = c['architecture_reference']; reference = resolve_path(spec['output'])
    if sha(reference/'technical/plan.json') != spec['plan_sha256']:
        raise ValueError('Changed MLP reference plan')
    refplan = read(reference/'technical/plan.json')
    refarms = [a for a in refplan['config']['arms'] if a['name'] in spec['arms']]
    if len(refarms) != len(c['arms']):
        raise ValueError('Missing matched MLP reference arms')
    for arm in c['arms']:
        previous = next(a for a in refarms if a['seed'] == arm['seed'])
        for key in ['dimension', 'history', 'response', 'train_steps', 'neural_step_fs', 'supervision']:
            if arm[key] != previous[key]:raise ValueError('Architecture comparison changes '+key)
    out = export_root(root(c)/'analyses/architecture-comparison-v1'); delta = []; combined = {}; inputs = []
    for table, keys in [('dense_physical_path', ['metric', 'horizon_fs']),
                        ('free_rollout', ['space', 'metric', 'horizon_ps']),
                        ('free_rollout_by_regime', ['regime', 'space', 'metric', 'horizon_ps']),
                        ('nucleus_prediction', ['task', 'lead_ps', 'metric'])]:
        allrows = []
        for arm in c['arms']:
            previous = next(a for a in refarms if a['seed'] == arm['seed']); rows = {}
            for architecture, folder, selected in [('transformer', root(c), arm), ('mlp', reference, previous)]:
                file = table_path(folder/'analyses'/selected['name'], table)
                inputs.append(dict(path=str(file), sha256=sha(file)))
                with file.open() as stream:
                    rr = [dict(r, architecture=architecture, seed=arm['seed']) for r in csv.DictReader(stream)]
                rows[architecture] = {tuple(r[k] for k in keys): r for r in rr}; allrows.extend(rr)
            if rows['mlp'].keys() != rows['transformer'].keys():raise ValueError('Unmatched architecture metric rows')
            for key, current in rows['transformer'].items():
                old = rows['mlp'][key]
                delta.append(dict(table=table, seed=arm['seed'], metric_key='|'.join(key),
                                  transformer=float(current['value']), mlp=float(old['value']),
                                  difference=float(current['value'])-float(old['value'])))
        write_metric_rows(allrows, out, family=FAMILY, name=table, columns=list(allrows[0])); combined[table] = allrows
    write_metric_rows(delta, out, family=FAMILY, name='paired_differences', columns=list(delta[0]))
    write_json(out/'technical/input-tables.json', dict(tables=inputs))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, table, metric, tkey in [(axes[0], 'dense_physical_path', 'relative_response_v', 'horizon_fs'),
                                  (axes[1], 'free_rollout', 'mse', 'horizon_ps')]:
        for architecture in ['mlp', 'transformer']:
            selected = [r for r in combined[table] if r['architecture'] == architecture and r['metric'] == metric
                        and (table != 'free_rollout' or r['space'] == 'structure')]
            times = sorted({float(r[tkey]) for r in selected})
            values = np.array([[float(next(r['value'] for r in selected if r['seed'] == seed and float(r[tkey]) == t))
                                for t in times] for seed in c['seeds']])
            ax.plot(times, values.mean(0), 'o-', label=architecture)
            ax.fill_between(times, values.min(0), values.max(0), alpha=.15)
        ax.set_xlabel(tkey); ax.set_ylabel(metric); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Matched two-seed architecture comparison; shading is seed range'); fig.tight_layout()
    plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots/'architectures.png', dpi=170); fig.savefig(plots/'architectures.pdf'); plt.close(fig)
    write_json(out/'technical/complete.json', dict(state='complete', reference=str(reference), models=4))
