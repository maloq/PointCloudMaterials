"""Matched present/future/observable-response/multitransition objectives."""
import time
import numpy as np
import torch
from .data import read, root, normalization, write_json, sha
from .history_data import load
from .history_learning import fit as history_fit
from .learning import batch, objective
from .measurement_model import MeasuredHistoryProcess
from .measurement_data import targets, GROUPS, FAMILY


def group_loss(error, active, groups=GROUPS):
    return torch.stack([error[..., lo:hi][..., active[lo:hi]].square().mean()
                        for _, lo, hi in groups]).mean()


def loss_function(data):
    def loss(model, state, tangent, history, ids, length, k, norm, arm, c):
        initial, target, direction, response = batch(state, tangent, ids, np.zeros(len(ids), dtype=int), length, k)
        bound = model.condition(history[ids])
        prediction, codes = bound(initial, length, latents=True)
        base_arm = dict(arm, latent_response=not arm['multi_transport'])
        _, scores = objective(bound, initial, target, direction, response, norm, base_arm, c, prediction=prediction)
        present = model.measurement(bound.encode(initial))
        truth = data['value'][ids, :length+1]
        scores['measurement_present'] = group_loss(present-truth[:, 0], data['active'])
        if arm['future_measurements']:
            future = model.measurement(codes.flatten(0, 1)).reshape(len(ids), length, 244)
            scores['measurement_future'] = group_loss(future-truth[:, 1:], data['active'])
            joined = torch.cat((present[:, None], future), 1)
            scores['measurement_increments'] = group_loss(torch.diff(joined, dim=1)-torch.diff(truth, dim=1), data['active'])
        if arm['observable_response']:
            _, derivative = torch.func.jvp(lambda h: model.measured_path(bound, h, length), (initial,), (direction,))
            error = (derivative[..., 153:]-data['response'][ids, 1:length+1, :, k])
            error = error/data['response_power'][None, 1:length+1, :, k].sqrt()
            scores['measurement_response'] = group_loss(error, data['active'][153:], [('density', 0, 74), ('velocity', 74, 91)])
        if arm['multi_transport']:
            # Four separated physical transitions; rotate offsets so all twenty receive equal coverage.
            slots = (torch.arange(4, device=state.device)*5+model.fit_update) % 20
            h = state[ids][:, slots].flatten(0, 1); following = state[ids][:, slots+1].flatten(0, 1)
            u = tangent[ids][:, slots, ..., k].flatten(0, 1); du = tangent[ids][:, slots+1, ..., k].flatten(0, 1)
            _, left = torch.func.jvp(lambda y: bound.step(bound.encode(y)), (h,), (u,))
            _, right = torch.func.jvp(bound.encode, (following,), (du,))
            scores['latent_response'] = (left-right).square().mean()/right.detach().square().mean().clamp_min(1e-6)
        return sum(c['training']['loss_weights'][key]*value for key, value in scores.items()), scores
    return loss


def fit(c, arm, deadline):
    return history_fit(c, arm, deadline, model_factory=MeasuredHistoryProcess,
        loss_function=loss_function(targets(c, 'train')), family=FAMILY, run_prefix='phasemeas')


def benchmark(c):
    _, records, state, tangent, history = load(c, 'train')
    data = targets(c, 'train'); norm = normalization(state, tangent); loss_fn = loss_function(data); report = []
    for arm in [a for a in c['arms'] if a['seed'] == c['seed']]:
        torch.manual_seed(arm['seed']); model = MeasuredHistoryProcess(c, arm).cuda(); model.fit_update = 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for length in [4, 20]:
            seconds = []
            for repeat in range(2):
                norm['selected_power'] = norm['response_power'][:length, :, 0]
                optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.monotonic()
                n = c['training']['batch_size']; micro = c['training']['microbatch']
                for start in range(0, n, micro):
                    ids = np.arange(start, min(start+micro, n)) % len(state)
                    loss, scores = loss_fn(model, state, tangent, history, ids, length, 0, norm, arm, c)
                    (loss*len(ids)/n).backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize(); seconds.append(time.monotonic()-begin)
            row = dict(arm=arm['name'], steps=length, seconds=seconds, memory_GiB=torch.cuda.max_memory_allocated()/2**30,
                parameters=sum(x.numel() for x in model.parameters()), loss=float(loss.detach()),
                scores={k: float(v.detach()) for k, v in scores.items()}, gradient_norm=float(grad))
            report.append(row); print(row, flush=True)
        with torch.no_grad():
            h = history[:2]; x = state[:2, 0]; u = tangent[:2, 0, ..., 0]; bound = model.condition(h)
            # Called before the captured objects are explicitly released below.
            query = lambda y: model.measured_path(bound, y, 4)  # noqa: F821
            _, jvp = torch.func.jvp(query, (x,), (u,))
            secant = (query(x+.1*u)-query(x-.1*u))/.2
            error = float((jvp-secant).norm()/jvp.norm().clamp_min(1e-8))
            rotation = torch.linalg.qr(torch.randn(3, 3, device='cuda'))[0]
            rotated = (h.reshape(2, 4, 80, 2, 3) @ rotation).flatten(-2)
            actual = model.measured_path(model.condition(rotated), rotated[:, -1], 4)
            invariant_error = float((actual-query(x)).abs().max())
        if error > .03 or invariant_error > 1e-4:
            raise ValueError(f'Measurement decoder JVP/rotation check: {error}, {invariant_error}')
        report.append(dict(arm=arm['name'], jvp_secant_error=error, rotation_error=invariant_error))
        del model, optimizer, bound; torch.cuda.empty_cache()
    write_json(root(c)/'technical/benchmark.json', dict(gpu=torch.cuda.get_device_name(), rows=report,
        scope='Real training inputs, local integration/benchmark only; no online run'))
