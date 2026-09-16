"""Training-only scales and exact existing cohorts for the recovery experiments."""

from pathlib import Path
import shutil

import numpy as np
import torch

from src.data_utils.topology_targets import fit_targets
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json

GROUPS = {'hot': slice(0, 144), 'relaxed': slice(144, 288), 'structural': slice(288, 292)}
OBSERVABLES = ['q4', 'q6', 'nearest_shell_density', 'mean_r12_A']


def inputs(config):
    pilot = load_json(config['pilot_config'])
    paths = [Path(pilot['diagnostics'])/'technical'/f'{name}.npz' for name in ['probes', 'temporal']]
    arrays = []
    for path in paths:
        with np.load(path) as data:
            arrays.append({key: data[key] for key in data.files if key in
                ['z', 'hot', 'relaxed', 'geometry', 'source', 'context', 'split', 'atom_id', 'observables']})
    probes, temporal = arrays
    assert probes['z'].shape == (5760, 256), probes['z'].shape
    assert temporal['z'].shape == (144, 17, 256), temporal['z'].shape
    ids = {s: np.flatnonzero(probes['split'] == s) for s in ['train', 'val', 'test']}
    groups = [set(probes['source'][ids[s]]) for s in ids]
    if any(groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError('Recovery source splits overlap')
    np.testing.assert_array_equal(np.unique(temporal['source']), sorted(groups[2]))
    raw = np.concatenate([probes['hot'], probes['relaxed'], probes['geometry'][:, -4:]], axis=1).astype(np.float64)
    mean = raw[ids['train']].mean(axis=0)
    scales, weights = [], []
    for key in ['hot', 'relaxed']:
        scale = fit_targets(probes[key][ids['train']], 32, .05)['block_scale']
        scales.extend(np.repeat(scale, [16, 64, 64]))
        weights.extend(np.repeat(1/(3*np.array([16, 64, 64])), [16, 64, 64]))
    scales.extend(raw[ids['train'], 288:].std(axis=0))
    weights.extend([.25]*4)
    scales, weights = np.asarray(scales), np.asarray(weights)
    if np.any(scales <= 0):
        raise ValueError('Nonpositive target scale')
    targets = (raw-mean)/scales
    return pilot, probes, temporal, ids, targets, mean, scales, weights, {str(p): sha256(p) for p in paths}


def feature_arrays(pilot, state, mode):
    paths = [Path(pilot['output'])/'technical'/f'{state}-{m}'/'features.npz'
             for m in (['halo_inner', 'halo_center'] if mode == 'fusion' else [mode])]
    pieces = []
    for path in paths:
        status = load_json(path.parent/'status.json')
        if status['state'] != 'complete':
            raise ValueError(f'Incomplete feature extraction: {path}')
        with np.load(path) as data:
            pieces.append({k: data[k] for k in data.files})
    for part in pieces[1:]:
        np.testing.assert_array_equal(part['epsilons'], pieces[0]['epsilons'])
    result = {k: np.concatenate([p[k] for p in pieces], axis=-1)
              for k in ['z', 'temporal_z', 'crossing_z']}
    result['epsilons'] = pieces[0]['epsilons']
    return result, {str(p): sha256(p) for p in paths}


def group_errors(prediction, target, weights):
    return {key: np.sum((prediction[:, section]-target[:, section])**2*weights[section], axis=1)
            for key, section in GROUPS.items()}


class PhysicalHeads(torch.nn.Module):
    """Independent prediction heads; losses and validation selection stay separate."""
    def __init__(self, width, hidden):
        super().__init__()
        self.heads = torch.nn.ModuleDict({name: torch.nn.Sequential(
            torch.nn.Linear(width, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, section.stop-section.start)) for name, section in GROUPS.items()})

    def forward(self, z):
        return torch.cat([head(z) for head in self.heads.values()], dim=1)


def publish(config, names):
    root, local = Path(config['output']), Path(config['local_output'])
    local.mkdir(parents=True, exist_ok=True)
    for relative in names:
        source = root/relative
        target = local/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def status(config, relative, **fields):
    path = Path(config['output'])/'technical'/relative/'status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, fields)
    publish(config, [str(path.relative_to(Path(config['output'])))])
