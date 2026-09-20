"""Structural transfer with explicitly changed training-target normalization."""
import numpy as np
import torch
from src.models.encoders.structural import ARCHITECTURE_REVISION


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
