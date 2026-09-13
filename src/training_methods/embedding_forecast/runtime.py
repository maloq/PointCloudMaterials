"""Explicit execution options and reviewed implementation transitions on resume."""

import json
from pathlib import Path

from src.experiment_runner.registry import sha256, write_json
from .data import window_loader
from .resident import ResidentLoader


def implementation_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in
            ('data.py', 'model.py', 'metrics.py', 'run.py', 'augmentation.py', 'resident.py', 'runtime.py',
             'context_mixture.py', 'spatial.py')}


def execution_settings(runtime):
    if runtime is None:
        return dict(loader='mmap', log_every_steps=0)
    if runtime['loader'] not in ('mmap', 'resident') or runtime['log_every_steps'] < 0:
        raise ValueError(f'Invalid forecast execution settings: {runtime}')
    if runtime.get('validation_residency', 'device') not in ('device', 'host'):
        raise ValueError(f'Validation residency must be device or host: {runtime}')
    return runtime


def forecast_loader(dataset, settings, shuffle, seed, device, runtime):
    from .spatial import SpatialWindowDataset, SpatialResidentLoader
    if isinstance(dataset, SpatialWindowDataset):
        if runtime['loader'] != 'resident':
            raise ValueError('Spatial forecasts require the explicit resident loader to pool same-frame neighbors.')
        return SpatialResidentLoader(dataset, settings['batch_size'], shuffle, seed, device)
    if runtime['loader'] == 'resident':
        return ResidentLoader(dataset, settings['batch_size'], shuffle, seed, device)
    return window_loader(dataset, settings['batch_size'], settings['workers'], shuffle, seed)


def check_resume_implementation(checkpoint, current, transition_path, directory):
    previous = checkpoint['implementation_sha256']
    if previous == current:
        return
    if transition_path is None:
        raise ValueError(f'Resume implementation changed; use its frozen source or a reviewed transition: {directory}')
    transition = json.loads(Path(transition_path).read_text())
    if transition['previous'] != previous or transition['replacement'] != current:
        raise ValueError(f'Resume transition does not match the exact previous/replacement source: {transition_path}')
    protected = ('model.py', 'augmentation.py') if transition.get('kind') == 'storage-relocation' else ('data.py', 'model.py', 'augmentation.py')
    protected += tuple(name for name in ('context_mixture.py', 'spatial.py') if name in previous)
    if transition.get('kind') == 'storage-relocation' and checkpoint['cache_manifest_sha256'] not in transition['cache_manifest_sha256']:
        raise ValueError('Storage transition does not name this exact verified cache manifest.')
    if any(previous[name] != current[name] for name in protected):
        raise ValueError('Execution-only resume must preserve the cache producer, model, losses and augmentation implementation.')
    write_json(directory / 'implementation_transition.json', dict(transition=transition,
        transition_sha256=sha256(Path(transition_path)), resumed_epoch=checkpoint['epoch'] + 1,
        preserved='model, optimizer, scheduler, scaler, sample order, RNG, best score and patience'))
