"""Small CPU integration example; synthetic plumbing data, not a research result."""

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from src.data.conversion.native_qv import convert
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import file_hash, write_json
from src.experiment_runner.metric_docs import write_metric_table
from src.models.encoders.mlp import MLPEncoder
from src.project_runtime.paths import REPO

DEFAULT_CONFIG = REPO / 'configs/examples/cpu_pipeline.json'


def _model(config):
    return nn.Sequential(
        MLPEncoder(config['num_points'], config['latent_size']),
        nn.Dropout(config['dropout']),
        nn.Linear(config['latent_size'], 1),
    )


def _prepare(root, config):
    """Exercise the real converter and reader with independent synthetic clouds."""
    rng = np.random.default_rng(config['seed'])
    rows = config['train_rows'] + config['test_rows']
    atoms = config['num_points']
    positions = 10 + rng.normal(size=(rows, atoms, 3)) * rng.uniform(
        0.25, 0.75, size=(rows, 1, 1)
    )
    source = root / 'native.npz'
    np.savez(
        source, positions=positions, velocities=np.zeros_like(positions),
        box_low=np.zeros((rows, 3)), box_high=np.full((rows, 3), 20.0),
        timesteps=np.arange(rows, dtype=np.int64),
        atom_ids=np.arange(1, atoms + 1, dtype=np.int64),
        atom_types=np.ones(atoms, dtype=np.int32),
        provenance=json.dumps({'purpose': 'integration_fixture',
                               'observations': 'independent clouds; not MD'}),
    )
    conversion = convert(source, root / 'binary')
    trajectory = ShootingBinaryTrajectory.load(root / 'binary')
    trajectory.verify_checksums()
    points = torch.from_numpy(np.array(trajectory.positions, dtype=np.float32))
    points = points - points.mean(dim=1, keepdim=True)
    targets = points.square().sum(dim=2).mean(dim=1, keepdim=True)
    train_rows = config['train_rows']
    scale = points[:train_rows].square().mean().sqrt()
    points = points / scale
    dataset = dict(points=points, targets=targets, scale=scale,
                   train_indices=torch.arange(train_rows),
                   test_indices=torch.arange(train_rows, rows))
    torch.save(dataset, root / 'dataset.pt')
    write_json(root / 'inputs.json', dict(
        kind='integration_fixture', source_sha256=file_hash(source),
        prepared_sha256=file_hash(root / 'dataset.pt'),
        conversion=conversion, normalization_fit='train only',
        inputs='centered coordinates; one train-fitted scalar normalization',
        target='mean squared distance from each cloud centroid',
        train_rows=train_rows, test_rows=config['test_rows'],
    ))
    return dataset


def _fit(dataset, config, stop, checkpoint=None):
    """Resume model, optimizer, stochastic batches and dropout from one checkpoint."""
    torch.manual_seed(config['seed'])
    model = _model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    batches = torch.Generator().manual_seed(config['seed'] + 1)
    history, start = [], 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng'])
        batches.set_state(checkpoint['batch_rng'])
        history, start = list(checkpoint['history']), checkpoint['step']
    model.train()
    for step in range(start, stop):
        indices = torch.randperm(config['train_rows'], generator=batches)
        indices = indices[:config['batch_size']]
        prediction = model(dataset['points'][indices])
        loss = nn.functional.mse_loss(prediction, dataset['targets'][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite example loss at update {step + 1}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({'step': step + 1, 'loss': loss.item(),
                        'batch_indices': indices.tolist()})
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(), batch_rng=batches.get_state(),
                step=stop, history=history, config=config,
                normalization_scale=dataset['scale'])


def run(output, config=None):
    """Prepare, fit, resume, evaluate and export in a fresh local output directory."""
    config_path = Path(config) if config is not None else DEFAULT_CONFIG
    config = json.loads(config_path.read_text())
    if not (0 < config['checkpoint_at'] < config['updates']
            and 0 < config['batch_size'] <= config['train_rows']
            and config['test_rows'] > 0 and config['num_points'] > 0):
        raise ValueError('Require train/test rows, a valid batch and an intermediate checkpoint')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    technical = root / 'technical'
    technical.mkdir()
    write_json(technical / 'config.json', config)
    write_json(technical / 'environment.json', dict(
        python=sys.version, torch=torch.__version__,
        numpy=np.__version__, device='cpu', threads=1,
    ))
    torch.set_num_threads(1)
    dataset = _prepare(technical, config)
    uninterrupted = _fit(dataset, config, config['updates'])
    interrupted = _fit(dataset, config, config['checkpoint_at'])
    torch.save(interrupted, technical / 'interrupted.pt')
    restored = torch.load(technical / 'interrupted.pt', weights_only=True,
                          map_location='cpu')
    resumed = _fit(dataset, config, config['updates'], restored)
    torch.testing.assert_close(resumed, uninterrupted, rtol=0, atol=0)
    torch.save(resumed, technical / 'last.pt')
    model = _model(config)
    model.load_state_dict(torch.load(technical / 'last.pt', weights_only=True,
                                     map_location='cpu')['model'], strict=True)
    model.eval()
    # The held-out rows are first evaluated after the fixed budget is complete.
    with torch.inference_mode():
        indices = dataset['test_indices']
        prediction = model(dataset['points'][indices])
        mse = nn.functional.mse_loss(prediction, dataset['targets'][indices]).item()
    np.save(technical / 'test_predictions.npy', prediction.numpy())
    metrics = dict(test_mse=mse, training_updates=config['updates'],
                   train_rows=config['train_rows'], test_rows=config['test_rows'])
    table = write_metric_table(metrics, root, family='research_example')
    report = dict(state='passed', kind='integration_example', device='cpu',
                  resume_exact=True, checkpoint_sha256=file_hash(technical / 'last.pt'),
                  metrics=metrics, table=str(table.relative_to(root)),
                  interpretation='Checks software plumbing; no scientific claim')
    write_json(technical / 'verification.json', report)
    (root / 'README.md').write_text(
        '# CPU integration example\n\n'
        'Synthetic independent clouds; not a material simulation or benchmark.\n\n'
        'The real Q/V converter, binary reader, MLP encoder, optimizer, checkpoint\n'
        'reload and metric exporter were exercised. Interrupted and uninterrupted\n'
        'training agree exactly, including optimizer and random states.\n\n'
        f'Test MSE after {config["updates"]} fixed updates: {mse:.8g}.\n\n'
        'See `technical/config.json`, `technical/verification.json` and `tables/`.\n'
    )
    return report
