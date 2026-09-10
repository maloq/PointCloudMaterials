"""Controlled frozen-MACE topology supervision and temporal denoising ablations."""

from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn

from src.data_utils.mace_denoising import prepare, cache_features, signature
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.mace_denoising import ResidualFrameFusion, AtomTemporalFusion
from src.models.encoders.mace_temporal import PretrainedMACETemporalEncoder
from src.training_methods.mace_objective import variance_covariance


BLOCKS = (slice(0, 16), slice(16, 80), slice(80, 144))


def fit_targets(targets, components, floor_fraction):
    pca = PCA(n_components=components, svd_solver='full').fit(targets)
    block_std = np.array([np.sqrt(targets[:, block].var(0).mean()) for block in BLOCKS])
    floor = floor_fraction * block_std.max()
    block_scale = np.sqrt(block_std**2 + floor**2)
    pixel_scale = np.concatenate([np.full(block.stop-block.start, scale) for block, scale in zip(BLOCKS, block_scale)])
    return dict(tda_mean=pca.mean_.astype(np.float32), tda_components=pca.components_.astype(np.float32),
        tda_std=np.maximum(np.sqrt(pca.explained_variance_), 1e-5).astype(np.float32),
        pixel_scale=pixel_scale.astype(np.float32), block_std=block_std, block_scale=block_scale,
        pca_variance_ratio=pca.explained_variance_ratio_)


def transform_target(target, scaling, kind):
    centered = target - scaling['tda_mean']
    if kind == 'pca':
        return (centered @ scaling['tda_components'].T) / scaling['tda_std']
    if kind == 'blocks':
        return centered / scaling['pixel_scale']
    raise ValueError(f'Unknown topology target transform: {kind}')


def raw_prediction(prediction, scaling, kind):
    if kind == 'pca':
        return (prediction * scaling['tda_std']) @ scaling['tda_components'] + scaling['tda_mean']
    if kind == 'blocks':
        return prediction * scaling['pixel_scale'] + scaling['tda_mean']
    raise ValueError(f'Unknown topology target transform: {kind}')


def topology_loss(prediction, target, kind):
    errors = (prediction-target).square()
    if kind == 'pca':
        return errors.mean()
    return torch.stack([errors[:, block].mean() for block in BLOCKS]).mean()


class FrozenData:
    def __init__(self, cfg):
        self.records = json.loads((Path(cfg['feature_cache']) / 'manifest.json').read_text())['shards']
        self.targets_raw = np.concatenate([np.load(Path(r['data_directory']) / 'targets.npy') for r in self.records])
        self.hot_targets = np.concatenate([np.load(Path(r['data_directory']) / 'hot_targets.npy') for r in self.records])
        self.contexts = np.concatenate([np.full(r['count'], i, np.int64) for i, r in enumerate(self.records)])
        self.sources = np.concatenate([np.full(r['count'], r['source_index'], np.int64) for r in self.records])
        self.temperatures = np.concatenate([np.full(r['count'], r['temperature_K']) for r in self.records])
        self.times = torch.tensor(np.concatenate([
            np.tile(r['provenance']['offsets'], (r['count'], 1)) for r in self.records]),
            dtype=torch.float32, device='cuda') / abs(cfg['frame_offsets_ps'][-2])
        self.indices = {split: np.flatnonzero(np.isin(self.contexts, [i for i, r in enumerate(self.records) if r['split']==split]))
                        for split in ('train', 'val', 'test')}
        self.pooled = torch.from_numpy(np.concatenate([np.load(Path(r['directory']) / 'pooled.npy') for r in self.records])).cuda()
        shape = (len(self.targets_raw), len(cfg['frame_offsets_ps']), 80, 256)
        self.nodes = torch.empty(shape, dtype=torch.float16, device='cuda')
        offset = 0
        for r in self.records:
            self.nodes[offset:offset+r['count']].copy_(torch.from_numpy(np.load(Path(r['directory']) / 'nodes.npy')))
            offset += r['count']
        path = Path(cfg['output']) / 'scaling.npz'
        scaling_signature = signature(dict(features=json.loads((Path(cfg['feature_cache']) / 'manifest.json').read_text()),
            components=cfg['tda_components'], floor_fraction=cfg['block_scale_floor_fraction']))
        if path.exists():
            if json.loads(path.with_suffix('.json').read_text())['signature'] != scaling_signature:
                raise ValueError(f'Target scaling configuration or features changed: {path}')
            self.scaling = dict(np.load(path))
        else:
            self.scaling = fit_targets(self.targets_raw[self.indices['train']], cfg['tda_components'], cfg['block_scale_floor_fraction'])
            train_pooled = self.pooled[self.indices['train'], :-1]
            self.scaling.update(pooled_mean=train_pooled.mean((0, 1)).cpu().numpy(),
                                pooled_std=train_pooled.std((0, 1), correction=0).clamp_min(.01).cpu().numpy())
            total = torch.zeros(256, dtype=torch.float64, device='cuda')
            squares = torch.zeros_like(total)
            count = 0
            for chunk in np.array_split(self.indices['train'], 64):
                values = self.nodes[chunk].double().reshape(-1, 256)
                total += values.sum(0)
                squares += values.square().sum(0)
                count += len(values)
            mean = total/count
            std = (squares/count-mean.square()).clamp_min(.0001).sqrt()
            self.scaling.update(node_mean=mean.float().cpu().numpy(), node_std=std.float().cpu().numpy())
            np.savez(path, **self.scaling)
            write_json(path.with_suffix('.json'), dict(signature=scaling_signature, fit_split='train', anchors=len(self.indices['train']),
                block_std=self.scaling['block_std'].tolist(), block_effective_scale=self.scaling['block_scale'].tolist(),
                pca_cumulative_variance=self.scaling['pca_variance_ratio'].cumsum().tolist(),
                atom_features='Cached raw float16, normalized by training node moments; pooling cache is float32.'))
        self.pooled = (self.pooled-torch.as_tensor(self.scaling['pooled_mean'], device='cuda'))/torch.as_tensor(self.scaling['pooled_std'], device='cuda')
        self.node_mean = torch.as_tensor(self.scaling['node_mean'], device='cuda')
        self.node_std = torch.as_tensor(self.scaling['node_std'], device='cuda')
        self.targets = {kind: torch.from_numpy(transform_target(self.targets_raw, self.scaling, kind).astype(np.float32)).cuda()
                        for kind in ('pca', 'blocks')}

    def get(self, rows, architecture, intervention='real'):
        pooled = self.pooled[rows, :-1]
        nodes = (self.nodes[rows].float()-self.node_mean)/self.node_std if architecture.startswith('atom_') else None
        if architecture == 'relaxed':
            pooled = self.pooled[rows, -1:]
        if intervention == 'repeat_anchor':
            pooled = pooled[:, -1:].expand_as(pooled)
            if nodes is not None:
                nodes = nodes[:, -1:].expand_as(nodes)
        elif intervention == 'reverse_past':
            order = [*range(pooled.shape[1]-2, -1, -1), pooled.shape[1]-1]
            pooled = pooled[:, order]
            if nodes is not None:
                nodes = nodes[:, order]
        elif intervention != 'real':
            raise ValueError(f'Unknown history intervention: {intervention}')
        return pooled, nodes, self.times[rows]


class Predictor(nn.Module):
    def __init__(self, cfg, variant):
        super().__init__()
        self.architecture = variant['architecture']
        self.kind = variant['target']
        width = 256
        if self.architecture == 'transformer':
            self.fusion = PretrainedMACETemporalEncoder(cfg['pretrained_checkpoint'], cfg['frame_offsets_ps'],
                time_scale_ps=abs(cfg['frame_offsets_ps'][-2]), performance=cfg['performance'])
            self.fusion.mace.requires_grad_(False)
            width = 128
        elif self.architecture == 'residual':
            self.fusion = ResidualFrameFusion(cfg['frame_offsets_ps'])
        elif self.architecture in ('atom_anchor', 'atom_temporal'):
            self.fusion = AtomTemporalFusion(cfg['frame_offsets_ps'], anchor_only=self.architecture=='atom_anchor')
        elif self.architecture not in ('anchor', 'mean', 'relaxed'):
            raise ValueError(f'Unknown denoising architecture: {self.architecture}')
        self.trunk = nn.Sequential(nn.Linear(width, 256), nn.SiLU())
        sizes = [cfg['tda_components']] if self.kind == 'pca' else [16, 64, 64]
        self.heads = nn.ModuleList([nn.Linear(256, size) for size in sizes])

    def encode(self, pooled, nodes, times=None):
        if self.architecture == 'transformer':
            return self.fusion.forward_features(pooled, times)
        if self.architecture in ('residual', 'atom_anchor', 'atom_temporal'):
            return self.fusion(pooled, nodes, times)
        if self.architecture == 'mean':
            return pooled.mean(1)
        return pooled[:, -1]

    def forward(self, pooled, nodes, times=None):
        z = self.encode(pooled, nodes, times)
        hidden = self.trunk(z)
        return torch.cat([head(hidden) for head in self.heads], -1), z


@torch.no_grad()
def predict(model, data, rows, cfg, intervention='real'):
    model.eval()
    predictions, embeddings = [], []
    size = cfg['atom_batch_size'] if model.architecture.startswith('atom_') else cfg['batch_size']
    for start in range(0, len(rows), size):
        output, z = model(*data.get(rows[start:start+size], model.architecture, intervention))
        predictions.append(output.cpu())
        embeddings.append(z.cpu())
    return torch.cat(predictions).numpy(), torch.cat(embeddings).numpy()


def validation_loss(model, data, cfg):
    rows = data.indices['val']
    prediction, _ = predict(model, data, rows, cfg)
    return float(topology_loss(torch.from_numpy(prediction), data.targets[model.kind][rows].cpu(), model.kind))


def preflight(cfg, data):
    """Check initial identity skip, real temporal gradients and an overfit sanity test."""
    rows = data.indices['train'][:64]
    reports = {}
    for architecture in ('residual', 'atom_temporal'):
        model = Predictor(cfg, dict(architecture=architecture, target='blocks')).cuda()
        pooled, nodes, times = data.get(rows[:4], architecture)
        torch.testing.assert_close(model.encode(pooled, nodes, times), pooled[:, -1], rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        model.train()
        loss = topology_loss(model(pooled, nodes, times)[0], data.targets['blocks'][rows[:4]], 'blocks')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pooled = pooled.detach().requires_grad_()
        if nodes is not None:
            nodes = nodes.detach().requires_grad_()
        topology_loss(model(pooled, nodes, times)[0], data.targets['blocks'][rows[:4]], 'blocks').backward()
        derivative = pooled.grad.abs().sum((0, 2)) if nodes is None else nodes.grad.abs().sum((0, 2, 3))
        if not torch.isfinite(derivative).all() or not (derivative > 0).all():
            raise AssertionError(f'Missing history gradients in {architecture}: {derivative}')
        reports[architecture] = dict(initial_skip_exact=True, gradient_l1_by_frame=derivative.tolist())
    model = Predictor(cfg, dict(architecture='anchor', target='blocks')).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    batch = data.get(rows, 'anchor')
    target = data.targets['blocks'][rows]
    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = topology_loss(model(*batch)[0], target, 'blocks')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses[-1] >= .5*losses[0]:
        raise AssertionError(f'Tiny-set topology fit failed: {losses[0]} -> {losses[-1]}')
    reports['tiny_set'] = dict(examples=len(rows), initial_loss=losses[0], final_loss=losses[-1], retained_updates=0)
    write_json(Path(cfg['output']) / 'preflight.json', dict(state='passed', checks=reports))


def train_one(cfg, variant, seed, data):
    directory = Path(cfg['output']) / 'runs' / f"{variant['name']}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / 'training_summary.json'
    training_signature = signature({key: value for key, value in cfg.items() if key != 'deadline'})
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved['signature'] != training_signature or saved['variant'] != variant or saved['seed'] != seed or saved['state'] != 'complete':
            raise ValueError(f'Completed run does not match requested variant: {directory}')
        return
    if (directory / 'last.pt').exists():
        raise FileExistsError(f'Interrupted ablation in {directory}; explicit restart configuration is required.')
    torch.manual_seed(seed)
    model = Predictor(cfg, variant).cuda()
    # Every residual branch starts from the same selected anchor predictor.
    warmstart = variant['warmstart']
    if warmstart:
        source = Path(cfg['output']) / 'runs' / f'anchor_blocks_seed{seed}' / 'best.pt'
        checkpoint = torch.load(source, map_location='cuda', weights_only=False)
        head = {key: value for key, value in checkpoint['model'].items() if key.startswith(('trunk.', 'heads.'))}
        state = model.state_dict()
        state.update(head)
        model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-5)
    best = validation_loss(model, data, cfg)
    best_epoch = 0
    torch.save(dict(model=model.state_dict(), variant=variant, seed=seed, epoch=0, validation_loss=best), directory / 'best.pt')
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    steps = exposures = 0
    log_path = directory / 'training.jsonl'
    with log_path.open('w', buffering=1) as log:
        for epoch in range(1, cfg['epochs']+1):
            model.train()
            order = rng.permutation(data.indices['train'])
            totals = dict(tda=0., regularization=0.)
            for start in range(0, len(order), cfg['batch_size']):
                rows = order[start:start+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                micro = cfg['atom_batch_size'] if variant['architecture'].startswith('atom_') else len(rows)
                for j in range(0, len(rows), micro):
                    selected = rows[j:j+micro]
                    prediction, z = model(*data.get(selected, variant['architecture']))
                    task = topology_loss(prediction, data.targets[variant['target']][selected], variant['target'])
                    reg = torch.zeros((), device='cuda')
                    if variant['variance'] or variant['covariance']:
                        variance, covariance = variance_covariance(z)
                        reg = variant['variance']*variance + variant['covariance']*covariance
                    loss = task + reg
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss in {directory}, epoch {epoch}, step {steps}')
                    (loss * len(selected)/len(rows)).backward()
                    totals['tda'] += float(task.detach()) * len(selected)
                    totals['regularization'] += float(reg.detach()) * len(selected)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                exposures += len(rows)
            scheduler.step()
            val = validation_loss(model, data, cfg)
            record = dict(epoch=epoch, step=steps, train={key: value/len(order) for key, value in totals.items()},
                          validation_loss=val, seconds=time.monotonic()-started)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), variant=variant,
                           seed=seed, epoch=epoch, validation_loss=val)
            torch.save(payload, directory / 'last.pt')
            if val < best:
                best, best_epoch = val, epoch
                torch.save(payload, directory / 'best.pt')
            write_json(Path(cfg['output']) / 'status.json', dict(state='training', variant=variant['name'],
                seed=seed, epoch=epoch, best_epoch=best_epoch, validation_loss=val, best_loss=best))
            print('DENOISING_TRAIN', variant['name'], seed, json.dumps(record), flush=True)
            if epoch-best_epoch >= cfg['patience']:
                break
    write_json(completion, dict(state='complete', signature=training_signature, variant=variant, seed=seed, epochs_completed=epoch,
        best_epoch=best_epoch, best_validation_loss=best, optimizer_steps=steps, anchor_exposures=exposures,
        warmstart_anchor_checkpoint=warmstart, seconds=time.monotonic()-started,
        selection='Validation topology loss only. Test labels are excluded from fitting and selection.'))


def run(cfg, stage):
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads'])
    remaining = int(datetime.fromisoformat(cfg['deadline']).timestamp()-time.time())
    if remaining <= 0:
        raise TimeoutError(f"Allocation deadline passed: {cfg['deadline']}")
    def interrupted(signum, frame):
        raise InterruptedError(f'Denoising run interrupted by signal {signum}; see retained artifacts.')
    for sig in (signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(remaining)
    from src.experiment_runner.tracking import tracked_run
    try:
        with tracked_run(out, kind='training' if stage in ('train', 'all') else 'analysis',
                         configs=[Path(sys.argv[sys.argv.index('--config')+1])], command=[sys.executable, *sys.argv]):
            if stage in ('prepare', 'all'):
                if cfg['protocol'] == 'denoising80_reuse':
                    from src.data_utils.mace_existing import prepare_existing
                    prepare_existing(cfg)
                else:
                    prepare(cfg)
                cache_features(cfg)
            if stage in ('preflight', 'train', 'analysis', 'potential-audit', 'all'):
                data = FrozenData(cfg)
                if stage in ('preflight', 'train', 'all'):
                    preflight(cfg, data)
                if stage in ('train', 'all'):
                    for seed in cfg['seeds']:
                        for variant in cfg['variants']:
                            train_one(cfg, variant, seed, data)
                if stage in ('analysis', 'all'):
                    from src.analysis.mace_denoising import analyze
                    analyze(cfg, data)
                if stage == 'potential-audit' or (stage == 'all' and cfg['protocol']=='denoising80_reuse'):
                    if cfg['protocol'] != 'denoising80_reuse':
                        raise ValueError('The paired potential audit requires the explicit denoising80_reuse protocol.')
                    from src.analysis.mace_potential_audit import run_audit
                    exports = json.loads((out/'analysis/metrics.json').read_text())['exports']
                    run_audit(cfg, data, exports)
            write_json(out / 'status.json', dict(state='complete', stage=stage,
                completed_at=datetime.now().astimezone().isoformat()))
    except BaseException:
        write_json(out / 'status.json', dict(state='failed', traceback=traceback.format_exc()))
        raise
    finally:
        signal.alarm(0)
