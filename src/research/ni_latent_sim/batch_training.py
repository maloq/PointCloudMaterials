"""Explicit equal-example continuation of the frozen dense-path training run.

Optimizer update numbers remain actual updates. Scheduling, validation and
checkpoint cadence use processed examples, including the batch-256 prefix.
"""
import math
import time
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from .artifacts import write_rows as write_metric_rows
from .data import load, root, digest, sha, write_json, atomic_save, FAMILY
from .model import make_model, nll
from .dense_response import bank as dense_bank, objective


def run(c, p, arm, deadline, policy):
    folder = root(c)/'analyses'/arm['name']/'technical'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return True
    identity = digest([p['identity'], arm]); revision = digest(policy)
    torch.manual_seed(arm['seed'])
    model = make_model(arm).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=1e-5)
    data = load(c, 'data'); bank = dense_bank(c, p)
    mean = data['mean'].cuda(); scale = data['scale'].cuda()
    x = (data['x'].cuda()-mean)/scale
    ids = np.flatnonzero([r['role'] == 'train' for r in data['rows']])
    selection = np.flatnonzero([r['role'] == 'selection' for r in data['rows']])
    selection = np.random.default_rng(c['seed']).choice(selection, min(256, len(selection)), replace=False)
    shortids = np.flatnonzero([r['role'] == 'train' for r in bank])
    rng = np.random.default_rng(arm['seed'])
    start = 0; seen = 0; best = float('inf'); selected = 0; history = []
    if (folder/'last.pt').exists():
        old = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if old['identity'] != identity:
            raise ValueError('Changed parent model resume')
        if 'batch_revision' in old and old['batch_revision'] != revision:
            raise ValueError('Changed batch continuation policy')
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        start = old['update']; best = old['best']; selected = old['selected']; history = old['history']
        rng.bit_generator.state = old['rng']
        seen = old['examples_seen'] if 'batch_revision' in old else start*policy['original_batch_size']
    initial_seen = seen
    budget = policy['example_budget']; batch = policy['batch_size']
    total = start+math.ceil((budget-seen)/batch)
    write_json(folder/'batch-continuation.json', dict(policy=policy, revision=revision,
               parent_identity=identity, resume_update=start, resume_examples=seen,
               requested_optimizer_updates=total, remaining_examples=budget-seen))

    def save(update):
        atomic_save(folder/'last.pt', dict(identity=identity, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, best=best, selected=selected,
                    history=history, rng=rng.bit_generator.state, arm=arm,
                    batch_revision=revision, examples_seen=seen, batch_policy=policy))

    with online_training(c['wandb'], run_id='nils-'+identity[:16], name=arm['name'],
                         config=dict(c, arm=arm), folder=folder, receipt_path=folder/'wandb.json',
                         job_type='encoder', group=c['protocol'],
                         receipt_fields=dict(identity=identity, batch_revision=revision)) as tracking:
        tracking.config.update(dict(batch_continuation=policy), allow_val_change=True)
        for update in range(start+1, total+1):
            if time.time() > deadline-600:
                save(update-1); return False
            begin = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            count = min(batch, budget-seen); previous_seen = seen; next_seen = seen+count
            equivalent = next_seen/policy['original_batch_size']
            frac = min(equivalent/200, 1)*.5*(1+np.cos(np.pi*max(equivalent-200, 0)/3800))
            for group in optimizer.param_groups:
                group['lr'] = c['training']['lr']*max(frac, .01)
            rows = torch.as_tensor(rng.choice(ids, count), device='cuda')
            h = x[rows, :4]; target = x[rows][:, [4, 8, 13]]
            g = torch.Generator(device='cuda').manual_seed(int(rng.integers(2**31)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mu, lv = model.rollout(h, 10, 4, g)
                now, _ = model.emit(model.encode(h[:, -1]))
            likelihood = nll(mu[:, :, [0, 4, 9]].float(), lv[:, :, [0, 4, 9]].float(), target).mean()
            reconstruction = .5*(now.float()-h[:, -1]).square().mean()
            (likelihood+.2*reconstruction).backward()
            teacher_count = count*policy['original_teacher_batch']//policy['original_batch_size']
            records = [bank[i] for i in rng.choice(shortids, teacher_count)]
            v, d = objective(model, records, mean, scale, arm, int(rng.integers(2**31)))
            (.1*v+.1*d).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize(); seen = next_seen
            row = dict(update=update, nll=float(likelihood.detach()),
                       reconstruction_half_mse=float(reconstruction.detach()),
                       teacher_half_mse=float(v.detach()), relative_response=float(d.detach()), gradient_norm=float(norm))
            if seen//51200 > previous_seen//51200 or seen == budget:
                model.eval()
                with torch.no_grad():
                    g = torch.Generator(device='cuda').manual_seed(c['seed']+17)
                    mu, lv = model.rollout(x[selection, :4], 10, 8, g)
                    val = float(nll(mu[:, :, [0, 4, 9]], lv[:, :, [0, 4, 9]], x[selection][:, [4, 8, 13]]).mean())
                row['selection_nll'] = val
                if val < best:
                    best = val; selected = update
                    atomic_save(folder/'best.pt', dict(model=model.state_dict(), arm=arm, identity=identity,
                                update=update, examples_seen=seen, batch_revision=revision))
            history.append(row); tracking.log(row, step=update)
            if seen//25600 > previous_seen//25600 or update == start+1 or seen == budget:
                save(update)
            write_json(folder/'progress.json', dict(state='training', **row, total=total,
                       seconds_per_update=time.monotonic()-begin, examples_seen=seen,
                       example_budget=budget, batch_size=count, teacher_batch_size=teacher_count,
                       batch_revision=revision))
            if update == start+1 or 'selection_nll' in row:
                print(arm['name'], row, 'examples', seen, '/', budget, flush=True)
        tracking.summary.update(dict(selected_update=selected, selection_nll=best,
                                     actual_optimizer_updates=total, examples_seen=seen,
                                     teacher_examples_seen=seen*policy['original_teacher_batch']//policy['original_batch_size'],
                                     batch_continuation=policy, batch_continuation_start_examples=initial_seen))
    write_metric_rows(history, folder.parent, family=FAMILY, name='learning',
                      columns=list(dict.fromkeys(k for row in history for k in row)))
    write_json(folder/'complete.json', dict(identity=identity, updates=total, requested_optimizer_updates=total,
               selected_update=selected, selection_nll=best, checkpoint_sha256=sha(folder/'best.pt'),
               batch_revision=revision, examples_seen=seen, example_budget=budget, batch_policy=policy))
    return True
