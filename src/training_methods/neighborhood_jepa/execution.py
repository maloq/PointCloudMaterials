"""Packed GPU inputs and partial activation retention for full-batch JEPA losses.

Microbatches change execution only. The objective sees every anchor exactly once,
and both sides of the joint embedding objective receive encoder gradients.
"""
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.structural_pretraining.batches import move
from .regularization.data import worker_init


MACE_FIELDS = ('packed_positions', 'packed_weights', 'packed_species',
               'node_graph', 'edges', 'log_scale')


def select_profile(profiles, total_memory_gib):
    """Use explicit, offline-tested memory tiers; never benchmark during a fit."""
    eligible = [p for p in profiles if total_memory_gib >= p['min_vram_GiB']]
    if not eligible:
        raise ValueError(f'No execution profile for {total_memory_gib:.1f} GiB: {profiles}')
    selected = max(eligible, key=lambda p: p['min_vram_GiB'])
    if selected['microbatch'] < 1 or selected['retain_chunks'] < 0:
        raise ValueError(f'Invalid execution profile: {selected}')
    return dict(selected)


def collate_graphs(views):
    """The exact fields consumed by neighborhood JEPA's MACE atom pathway.

    No padded coordinates, dummy physical targets or padded species arrays are
    constructed. Neighborhood encoders pool atom features, not packed centers.
    """
    counts = np.array([v['positions'].shape[1] for v in views], dtype=np.int64)
    offsets = np.r_[0, counts.cumsum()]
    return dict(
        packed_positions=torch.from_numpy(np.concatenate([v['positions'][0] for v in views])),
        packed_weights=torch.from_numpy(np.concatenate([v['weights'][0] for v in views])),
        packed_species=torch.from_numpy(np.repeat(np.array([v['species'] for v in views], np.int64), counts)),
        node_graph=torch.from_numpy(np.repeat(np.arange(len(views)), counts)),
        edges=torch.from_numpy(np.concatenate([v['edges'] + offset for v, offset in zip(views, offsets)], axis=1)),
        log_scale=torch.tensor([v['log_scale'] for v in views], dtype=torch.float32))


def pack(samples, microbatch):
    views = [view for sample in samples for view in sample['views']]
    batches = [collate_graphs(views[i:i+microbatch]) for i in range(0, len(views), microbatch)]
    names = ['moments', 'position', 'times', 'physical', 'tda', 'query_atom_ids', 'order']
    if 'reservoir' in samples[0]:
        names.append('reservoir')
    if 'future_valid' in samples[0]:
        names.extend(('future_physical', 'future_tda', 'future_valid'))
    target = {name: torch.from_numpy(np.stack([s[name] for s in samples])) for name in names}
    for name in ('frame', 'index', 'group'):
        target[name] = torch.tensor([s[name] for s in samples], dtype=torch.long)
    target['temperature_K'] = torch.tensor([s['temperature_K'] for s in samples], dtype=torch.float32)
    return batches, target


def loader(data, sampler, microbatch, workers=0):
    if microbatch < 1:
        raise ValueError(f'Encoder microbatch must be positive: {microbatch}')
    if workers:
        torch.multiprocessing.set_sharing_strategy('file_system')
    return DataLoader(data, batch_sampler=sampler, collate_fn=partial(pack, microbatch=microbatch),
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(731),
        **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if workers else {}))


def stage_inputs(batches, device):
    """Cache all raw inputs for ONE update on GPU; never cache learned features.

    Pinned loader tensors transfer asynchronously. Replay reuses these tensors,
    so geometry/edges cross PCIe only once per update. Lifetime is explicitly
    bounded by the update; validation/test data cannot populate a training cache.
    """
    return [{name: batch[name].to(device, non_blocking=True) for name in MACE_FIELDS}
            for batch in batches]


@torch.no_grad()
def prime_encoder(encoder, example, precision):
    """Initialize the compiled inference path before retaining training graphs.

    With the pinned compiler/AMP stack, first tracing a retained gradient graph
    can initialize different BF16 forward numerics. The original replay path
    always traces inference first. Preserve that order, including after resume,
    without updating weights or consuming the scientific RNG stream.
    """
    device = next(encoder.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision=='bf16'):
        encoder(stage_inputs([example], device)[0])


def training_step(model, objective, batches, target, precision, diagnose=False, *,
                  retain_chunks=0, gpu_cache=True):
    if retain_chunks < 0:
        raise ValueError(f'retain_chunks must be nonnegative: {retain_chunks}')
    device = next(model.parameters()).device
    batches = stage_inputs(batches, device) if gpu_cache else batches
    exports, retained = [], []
    for index, batch in enumerate(batches):
        keep = index < retain_chunks
        with torch.set_grad_enabled(keep), torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
            value = model.encoder(move(batch, device)).float()
        exports.append(value.detach())
        retained.append(value if keep else None)
    encoded = torch.cat(exports).requires_grad_(True)
    del exports, value
    loss, terms = objective(model, encoded, target)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite JEPA objective: {terms}')
    diagnostics = dict(objective.diagnostics)
    if diagnose:
        for name, value in terms.items():
            gradient = torch.autograd.grad(value, encoded, retain_graph=True)[0]
            diagnostics[f'export_gradient/{name}'] = float(gradient.norm())
    # Heads and statistical regularization run once, at the original batch size.
    loss.backward()
    offset = 0
    for index, batch in enumerate(batches):
        value = retained[index]
        if value is None:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                value = model.encoder(move(batch, device)).float()
        value.backward(encoded.grad[offset:offset+len(value)])
        offset += len(value)
        retained[index] = None
        del value
    if offset != len(encoded):
        raise ValueError(f'Encoder gradient coverage changed: {offset} != {len(encoded)}')
    diagnostics.update(encoder_chunks=len(batches), retained_chunks=min(retain_chunks, len(batches)),
                       replayed_chunks=max(0, len(batches)-retain_chunks), encoded_observations=offset)
    return float(loss.detach()), {k: float(v.detach()) for k, v in terms.items()}, diagnostics
