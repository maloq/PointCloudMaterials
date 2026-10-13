"""Shot-resolved teacher power and finite-particle response-loss accounting."""
import numpy as np
import torch
from .data import (root, parent, read, sha, digest, write_json, atomic_save,
                   Motion, observation, resolve_path)
from .dense_response import bank, physical_config
from src.research.ni_response.simulator import archive
from .path_scores import noise_tape, rollout


def encode(c, pc, original, physical, model):
    factor = parent(pc)['config']['length_factor']
    shots, frames = physical['future'].shape[:2]
    q = (physical['future'][..., :3]*factor).float().flatten(0, 1)
    dq = (physical['tangent'][..., :3, :]*factor).float().flatten(0, 1)
    values, responses = [], []
    for offset in range(0, len(q), 8):
        xyz = q[offset:offset+8].cuda(); directions = dq[offset:offset+8].cuda()
        derivatives = []
        for k in range(2):
            with torch.enable_grad(), torch.autocast('cuda', enabled=False):
                value, dv = torch.autograd.functional.jvp(model.geometry, xyz, directions[..., k], strict=True)
            derivatives.append(dv.detach().cpu())
        values.append(value.detach().cpu()); responses.append(torch.stack(derivatives, -1))
    targets = torch.cat(values).reshape(shots, frames, 128)
    rates = torch.cat(responses).reshape(shots, frames, 128, 2)
    errors = {name:float((value.mean(0)-original[name]).square().mean().sqrt())
              for name, value in [('target', targets), ('response', rates)]}
    # Re-encoding the same physical archive on a different GPU is a numerical
    # diagnostic. Record discrepancies rather than relabel old targets.
    pairs = []
    state = physical['state']*factor; basis = physical['basis']*factor
    for epsilon in c['response_epsilons']:
        states = torch.stack([state+sign*epsilon*basis[..., k]
                              for k in range(2) for sign in [-1, 1]])
        pairs.append(observation(model, states, 4).reshape(2, 2, 280))
    return dict(targets=targets, rates=rates, pairs=torch.stack(pairs), replay_rmse=errors,
                role=physical['role'], ancestor=physical['ancestor'], job=physical['job'],
                basis_rms=physical['basis'].square().sum(1).mean(0).sqrt().tolist())


def decomposition(particles, target):
    # particle dimension 1, all others are sample/target coordinates.
    naive = (particles.mean(1)-target).square().mean()
    penalty = particles.var(1, unbiased=True).mean()/particles.shape[1]
    return dict(squared_mean_loss=float(naive), mean_variance_penalty=float(penalty),
                unbiased_mean_loss=float(naive-penalty),
                particle_variance=float(particles.var(1, unbiased=True).mean()))


def run(c):
    from .diagnostics import selected, model_for, export, summarize
    p = selected(c); pc = p['source_config']
    source_plan = read(resolve_path(c['source_run'])/'technical/plan.json')
    teachers = bank(pc, source_plan); originals = {r['job']:r for r in teachers}
    jobs = {j['key']:j for j in source_plan['jobs']}
    physical_c = physical_config(pc)
    normalizer = torch.load(root(c)/'technical/normalizer.pt', map_location='cpu', weights_only=False)
    mean, scale = normalizer['mean'].cuda(), normalizer['scale'].cuda()
    encoder = Motion(parent(pc)['config']).cuda().eval().requires_grad_(False)
    packets = []
    for i, job_key in enumerate(p['response_jobs']):
        packet = root(c)/'analyses/response-teachers/technical/packets'/(job_key+'.pt')
        receipt = packet.with_suffix('.json')
        physical_dir = archive(physical_c, jobs[job_key])
        physical_receipt = read(physical_dir/'complete.json')
        if physical_receipt['identity'] != digest([source_plan['identity'], jobs[job_key], physical_c['teacher']]):
            raise ValueError('Changed physical teacher identity')
        if receipt.exists():
            saved = read(receipt)
            if sha(packet) != saved['sha256'] or saved['physical_sha256'] != physical_receipt['sha256']:
                raise ValueError('Changed per-shot diagnostic packet')
            record = torch.load(packet, map_location='cpu', weights_only=False)
        else:
            if sha(physical_dir/'response.pt') != physical_receipt['sha256']:
                raise ValueError('Changed saved physical path')
            physical = torch.load(physical_dir/'response.pt', map_location='cpu', weights_only=False)
            record = encode(c, pc, originals[job_key], physical, encoder)
            if not all(torch.isfinite(record[k]).all() for k in ['targets','rates','pairs']):
                raise FloatingPointError('Nonfinite per-shot re-encoding: '+job_key)
            atomic_save(packet, record)
            write_json(receipt, dict(sha256=sha(packet), physical_sha256=physical_receipt['sha256']))
        packets.append(record)
        write_json(root(c)/'technical/response-progress.json', dict(stage='per_shot_encoding', completed=i+1,total=len(p['response_jobs'])))
        print('per-shot teacher', i+1, '/', len(p['response_jobs']), flush=True)
    del encoder
    teacher_rows = []
    for record in packets:
        rates = record['rates']/normalizer['scale'][:128,None]
        targets = (record['targets']-normalizer['mean'][:128])/normalizer['scale'][:128]
        initial = originals[record['job']]['initial_response']/normalizer['scale'][:128,None]
        for direction in range(2):
            for t in [0,3,9,19]:
                rr = rates[:,t,:,direction]; avg = rr.mean(0)
                variance_response = ((targets[0,t]-targets[1,t])*(rr[0]-rr[1]))
                metrics = dict(teacher_mean_power=float(avg.square().mean()),
                               raw_teacher_mean_power=float(record['rates'][:,t,:,direction].mean(0).square().mean()),
                               cross_shot_power=float((rr[0]*rr[1]).mean()),
                               shot_variance=float(rr.var(0).mean()),
                               two_shot_mean_noise=float((rr[0]-rr[1]).square().mean()/4),
                               zero_response_mse=float(avg.square().mean()),
                               persistence_response_mse=float((initial[:,direction]-avg).square().mean()),
                               variance_response_rms=float(variance_response.square().mean().sqrt()))
                for metric, value in metrics.items():
                    teacher_rows.append(dict(job=record['job'],ancestor=record['ancestor'],role=record['role'],
                                             direction=['Q','V'][direction], horizon_fs=(t+1)*5, metric=metric,value=value))
    export(c, 'response-teachers', teacher_rows)
    # Reproduce the training-only random batch denominator, including its floor.
    train = [r for r in teachers if r['role']=='train']
    rng = np.random.default_rng(c['seed']); denominators = []
    for draw in range(64):
        batch = [train[i] for i in rng.integers(len(train), size=16)]
        response = torch.stack([r['response'] for r in batch])/normalizer['scale'][:128,None]
        power = response.square().mean((0,2))
        for direction in range(2):
            denominators.append(dict(batch=draw, direction=['Q','V'][direction],
                minimum_raw_power=float(power[:,direction].min()),
                median_raw_power=float(power[:,direction].median()),
                maximum_raw_power=float(power[:,direction].max()),
                floor_fraction=float((power[:,direction]<1e-5).float().mean()),
                maximum_inverse_denominator=float(power[:,direction].clamp_min(1e-5).reciprocal().max())))
    export(c, 'response-normalizers', denominators)
    for spec in p['checkpoints']:
        model = model_for(c, spec); rows = []
        for record in packets:
            if record['role'] != 'test':
                continue
            original = originals[record['job']]
            history = ((original['history'].cuda()-mean)/scale)[None]
            target = (record['targets'].mean(0).cuda()-mean[:128])/scale[:128]
            teacher = record['rates'].mean(0).cuda()/scale[:128,None]
            for key in c['outer_keys']:
                tape = noise_tape(model, [record['job']], key, max(c['particles']), 1, 'cuda')
                with torch.no_grad():
                    mu, lv = rollout(model, history, tape, dense=True)
                    for k in c['particles']:
                        rows.append(dict(job=record['job'],ancestor=record['ancestor'],outer_key=key,
                            particles=k, quantity='value', epsilon=0.,
                            **decomposition(mu[:,:k,:,:128], target[None])))
                    base_derivatives = None
                    for e, epsilon in enumerate(c['response_epsilons']):
                        pairs = (record['pairs'][e].cuda()-mean)/scale
                        derivatives = []
                        for direction in range(2):
                            minus, plus = history.clone(), history.clone()
                            minus[:,-1] = pairs[direction,0]; plus[:,-1] = pairs[direction,1]
                            a, _ = rollout(model, minus, tape, dense=True)
                            b, _ = rollout(model, plus, tape, dense=True)
                            derivatives.append((b[...,:128]-a[...,:128])/(2*epsilon))
                        derivative = torch.stack(derivatives,-1)
                        if base_derivatives is None:
                            base_derivatives = derivative
                        for direction in range(2):
                            for k in c['particles']:
                                values = decomposition(derivative[:,:k,...,direction], teacher[None,...,direction])
                                values['epsilon_change_mse'] = float((derivative[:,:k,...,direction]-base_derivatives[:,:k,...,direction]).square().mean())
                                rows.append(dict(job=record['job'],ancestor=record['ancestor'],outer_key=key,
                                    particles=k,quantity='response_'+['Q','V'][direction],epsilon=epsilon,**values))
                write_json(root(c)/'technical/response-progress.json', dict(stage='student_loss',model=spec['arm']['name'],job=record['job'],outer_key=key))
            print('response audit',spec['arm']['name'],record['job'],flush=True)
        # Value rows have no epsilon comparison; an empty field is undefined.
        for row in rows:
            row.setdefault('epsilon_change_mse', None)
        export(c, 'response-loss-'+spec['arm']['name'], rows)
