"""Explicit execution options and reviewed implementation transitions on resume."""

import json
from contextlib import contextmanager
from pathlib import Path
import torch

from src.experiment_runner.registry import sha256, write_json
from .data import window_loader
from .resident import ResidentLoader


def implementation_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in
            ('data.py', 'model.py', 'metrics.py', 'run.py', 'augmentation.py', 'resident.py', 'runtime.py',
             'context_mixture.py', 'spatial.py', 'spatial_attention.py', 'attention_data.py')}


def execution_settings(runtime):
    if runtime is None:
        return dict(loader='mmap', log_every_steps=0)
    if runtime['loader'] not in ('mmap', 'resident') or runtime['log_every_steps'] < 0:
        raise ValueError(f'Invalid forecast execution settings: {runtime}')
    if runtime.get('validation_residency', 'device') not in ('device', 'host', 'staged_device', 'swap_device'):
        raise ValueError(f'Unknown validation residency: {runtime}')
    if runtime.get('validation_residency') in ('staged_device', 'swap_device') and runtime['loader'] != 'resident':
        raise ValueError('Staged validation requires the resident loader.')
    if 'evaluation_batch_size' in runtime and runtime['evaluation_batch_size'] < 1:
        raise ValueError('Evaluation batch size must be positive.')
    return runtime


@contextmanager
def validation_loader_on_device(loader, device, runtime, training_loader=None):
    """Copy a complete validation split once, then release it before backpropagation."""
    if runtime.get('validation_residency') not in ('staged_device','swap_device'):
        yield loader
        return
    from .spatial import SpatialResidentWindows
    from .attention_data import AttentionResidentWindows
    data = loader.dataset
    names = ['embeddings', 'offsets']
    if isinstance(data, SpatialResidentWindows):
        names += ['spatial', 'radii']
    if isinstance(data, AttentionResidentWindows):
        names = list(data.tensor_names)
    original = {name: getattr(data, name) for name in names}
    swap = runtime.get('validation_residency') == 'swap_device'
    if swap:
        if training_loader is None or not isinstance(training_loader.dataset, AttentionResidentWindows):
            raise ValueError('Swap validation requires the explicit attention training loader.')
        training = training_loader.dataset
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    try:
        if swap:
            for name in training.tensor_names:
                setattr(training,name,getattr(training,name).to('cpu'))
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
        for name, value in original.items():
            setattr(data, name, value.to(device))
        yield loader
    finally:
        for name, value in original.items():
            setattr(data, name, value)
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        if swap:
            for name in training.tensor_names:
                setattr(training,name,getattr(training,name).to(device))


def forecast_loader(dataset, settings, shuffle, seed, device, runtime):
    from .spatial import SpatialWindowDataset, SpatialResidentLoader
    from .attention_data import AttentionWindowDataset, AttentionResidentLoader
    batch_size = settings['batch_size'] if shuffle else runtime.get('evaluation_batch_size',settings['batch_size'])
    if isinstance(dataset, AttentionWindowDataset):
        if runtime['loader'] != 'resident':
            raise ValueError('Individual-neighbor attention requires the resident loader.')
        return AttentionResidentLoader(dataset,batch_size,shuffle,seed,device)
    if isinstance(dataset, SpatialWindowDataset):
        if runtime['loader'] != 'resident':
            raise ValueError('Spatial forecasts require the explicit resident loader to pool same-frame neighbors.')
        return SpatialResidentLoader(dataset, batch_size, shuffle, seed, device)
    if runtime['loader'] == 'resident':
        return ResidentLoader(dataset, batch_size, shuffle, seed, device)
    return window_loader(dataset, batch_size, settings['workers'], shuffle, seed)


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
    protected += tuple(name for name in ('context_mixture.py', 'spatial.py','spatial_attention.py','attention_data.py') if name in previous)
    if transition.get('kind') == 'storage-relocation' and checkpoint['cache_manifest_sha256'] not in transition['cache_manifest_sha256']:
        raise ValueError('Storage transition does not name this exact verified cache manifest.')
    if any(previous[name] != current[name] for name in protected):
        raise ValueError('Execution-only resume must preserve the cache producer, model, losses and augmentation implementation.')
    write_json(directory / 'implementation_transition.json', dict(transition=transition,
        transition_sha256=sha256(Path(transition_path)), resumed_epoch=checkpoint['epoch'] + 1,
        preserved='model, optimizer, scheduler, scaler, sample order, RNG, best score and patience'))
