"""Structural transfer with explicitly changed training-target normalization."""
import numpy as np
import torch
from src.models.encoders.structural import ARCHITECTURE_REVISION
from src.models.encoders.mixed_gatr import MIXED_ARCHITECTURE_REVISION


def continue_mixed_objective(model,objective,optimizer,saved,identity,total):
    """Explicit v7 -> temporal-only v8 fork; preserve optimization progress."""
    old=saved['identity']
    if old['protocol']!='shared_pretraining_mixed_v7' or identity['protocol']!='shared_pretraining_mixed_v8':
        raise ValueError('Objective continuation requires the declared mixed v7 -> v8 transition')
    if old['architecture_revision']!=MIXED_ARCHITECTURE_REVISION or old['data']!=identity['data']:
        raise ValueError('Objective continuation changed architecture or training data')
    allowed={'backtracking_weight','continue_from'}
    before={k:v for k,v in old['config'].items() if k not in allowed}
    after={k:v for k,v in identity['config'].items() if k not in allowed}
    if before!=after:raise ValueError('Objective continuation may change only backtracking weight and implementation')
    if saved['scheduler']!=dict(next_update=saved['step']+1,total_updates=total,**identity['config']['schedule']):
        raise ValueError('Objective continuation changed update budget or learning-rate schedule')
    if not 0<saved['step']<total:raise ValueError('Continuation checkpoint must be inside the training budget')
    # Normalizers must match bitwise; this transition does not reinterpret targets.
    for key,value in objective.state_dict().items():
        if not torch.equal(value.cpu(),saved['objective'][key].cpu()):
            raise ValueError(f'Objective continuation changed target/statistical buffer {key}')
    model.load_state_dict(saved['model'],strict=True)
    objective.load_state_dict(saved['objective'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'].cpu())
    if saved['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    return dict(parent_step=saved['step'],parent_best_selection_score=saved['best'],
        old_backtracking_weight=old['config']['backtracking_weight'],
        new_backtracking_weight=identity['config']['backtracking_weight'],
        backtracking_updates='temporal_only',optimizer='preserved',scheduler='preserved',
        selection='new baseline and best candidate under unchanged physical/TDA score')


def require_complete_tda(release):
    if release.manifest.get('tda_coverage') != 'all_supervised_views':
        raise ValueError('Training requires the completed full-TDA release')
    for shard in release.manifest['shards']:
        a = release.arrays[shard['task']['id']]
        slots = [2, 4] if shard['static'] else [2, 3, 4]
        endpoints = a['views'][:, slots].ravel()
        if np.any(endpoints < 0) or not np.all(a['tda_valid'][endpoints]):
            raise ValueError(f'Missing supervised TDA in shard {shard["task"]["id"]}')
        if not np.isfinite(a['tda'][endpoints]).all():
            raise FloatingPointError(f'Nonfinite TDA in shard {shard["task"]["id"]}')


@torch.no_grad()
def initialize_structural(model, objective, saved, config):
    previous = saved['identity']
    if previous['architecture_revision'] != ARCHITECTURE_REVISION:
        raise ValueError('Structural parent architecture revision differs')
    for name in ('architecture', 'history_frames', 'method', 'phase'):
        if previous['config'][name] != config[name]:
            raise ValueError(f'Structural parent differs in {name}')
    model.load_state_dict(saved['model'], strict=True)
    # A decoder emits standardized targets. Preserve its physical-unit output
    # when the new release refits moments, rather than silently reinterpreting it.
    for name in ('physical', 'tda'):
        last = getattr(model, name)[-1]
        old_mean = saved['objective'][f'{name}_mean'].to(last.weight)
        old_std = saved['objective'][f'{name}_std'].to(last.weight)
        new_mean = getattr(objective, f'{name}_mean').to(last.weight)
        new_std = getattr(objective, f'{name}_std').to(last.weight)
        ratio = old_std/new_std
        last.weight.mul_(ratio[:, None])
        last.bias.copy_((old_std*last.bias + old_mean-new_mean)/new_std)
    return dict(parent_step=saved['step'], parent_selection_score=saved['best'],
                decoder_transfer='preserve_physical_units', optimizer='fresh', scheduler='fresh')
