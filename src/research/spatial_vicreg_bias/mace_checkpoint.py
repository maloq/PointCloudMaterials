"""Frozen rich-MACE inference on the existing interface-view observations."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_config
from .embedding_travel import asset, payload
from .static_md import assign


def read(path):
    c = resolve_config(json.loads(Path(path).read_text()))
    if c['protocol'] != 'rich_mace_interface_v1':
        raise ValueError('Expected frozen rich-MACE interface analysis')
    out = Path(c['output'])
    binding = out/'technical/run-binding.json'
    if binding.exists() and json.loads(binding.read_text()) != c:
        raise ValueError('Analysis inputs/checkpoint changed; use a new output directory')
    if not binding.exists(): write_json(binding, c)
    return c, out


def load_model(c):
    import torch
    saved = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
    if sha(c['checkpoint']) != c['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    # Import the exact training producer. Its dependencies must also match.
    source = 'src/research/liquid_predictability/rich_multimaterial_train.py'
    frozen = Path(c['training_source']) / source
    if sha(frozen) != saved['implementation'][source]:
        raise ValueError('Frozen training source differs from checkpoint')
    for name in ('src/models/encoders/spatial_mace.py', 'src/models/encoders/mace_backend.py',
                 'src/research/supervised_onset/model.py', 'src/research/encoder_context/geometry.py',
                 'src/research/equivariant_context/features.py', 'src/research/crystal_vector/model.py'):
        if sha(name) != saved['implementation'][name]:
            raise ValueError('Changed inference dependency: ' + name)
    spec = importlib.util.spec_from_file_location('src.research.liquid_predictability.frozen_rich_mace', frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = copy.deepcopy(saved['config'])
    config['patch_chunk'] = c['batch_size']
    model = module.RichPatchMACE(config, saved['model']['readout.2.weight'].shape[0]).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    norm = saved['coordinate_normalization']
    factor = norm['reference_scale_A'] / norm['scales_A']['Al']
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    return model, factor


def features(model, patches, rows, scale, batch_size):
    import torch
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(rows), batch_size):
            ids = rows[start:start+batch_size]
            xyz = np.asarray(patches[ids, :80], np.float32) * scale
            x = torch.from_numpy(xyz).cuda()
            z, _ = model.encode(x)
            value = z.float().cpu().numpy()
            if not np.isfinite(value).all() or value.shape != (len(ids), 256):
                raise ValueError('Invalid frozen MACE state')
            yield start, ids, value


def infer(config):
    import torch
    c, out = read(config)
    data = out/'data'; data.mkdir(parents=True, exist_ok=True)
    model, scale = load_model(c)
    assay = Path(c['assay'])
    parents = np.load(assay/'parents.npy', mmap_mode='r')
    roles = np.load(assay/'role.npy'); uniform = np.load(assay/'uniform.npy')
    started = time.monotonic()
    cluster_path = data/'clusters.npz'
    if not cluster_path.exists():
        fit_rows = np.flatnonzero((roles == 'train') & uniform)
        z = np.empty((len(fit_rows), 256), np.float32)
        for start, ids, value in features(model, parents, fit_rows, scale, c['batch_size']):
            z[start:start+len(ids)] = value
            if start % 8192 == 0:
                print(f'Cluster fitting observations: {start}/{len(fit_rows)}; {time.monotonic()-started:.1f}s', flush=True)
        km = MiniBatchKMeans(n_clusters=7, batch_size=4096, n_init=3, max_iter=200,
                             random_state=20260929, reassignment_ratio=0).fit(z)
        np.savez_compressed(cluster_path, centers=km.cluster_centers_, fit_rows=fit_rows)
        del z
    with np.load(cluster_path) as z:
        centers = z['centers']
    # Preserve the exact source/frame/atom order already used for descriptor plots.
    src = np.load(assay/'source.npy'); frames = np.load(assay/'frame.npy'); atoms = np.load(assay/'atom.npy')
    lookup = {(int(src[i]), int(frames[i]), int(atoms[i])): i for i in np.flatnonzero((roles == 'test') & uniform)}
    d = payload(Path(c['datasets']['matched']['reference']))
    rows = np.array([lookup[k] for k in zip(d['source'], d['frame'], d['atom'])])
    held = data/'heldout.npz'
    if not held.exists():
        parts = [v for _, _, v in features(model, parents, rows, scale, c['batch_size'])]
        values = np.concatenate(parts)
        np.savez_compressed(held, z=values, labels=assign(values, centers), assay_row=rows,
                            source=src[rows], frame=frames[rows], atom=atoms[rows])
    for kind, ds in c['datasets'].items():
        reference = Path(ds['reference']); page = payload(reference)
        sampled_parts = []
        for snap in page['md']['snapshots']:
            key = snap['key']; folder = Path(ds['source'])/'data'/key
            dest = data/kind/key; dest.mkdir(parents=True, exist_ok=True)
            if (dest/'complete.json').exists():
                if kind == 'static':
                    with np.load(dest/'sampled.npz') as z: sampled_parts.append(dict(z))
                continue
            patches = np.load(folder/'patches.npy', mmap_mode='r')
            with np.load(folder/'physical.npz') as z: physical = dict(z)
            geometry = asset(reference/'travel-data'/f'{key}-geometry.js')
            travel_rows = np.asarray(geometry['rows'])
            sampled = np.asarray(physical['sample']) if kind == 'static' else np.empty(0, int)
            retained = np.unique(np.r_[travel_rows, sampled])
            # Include exactly the rows shown in the held-out plots, then audit replay.
            if kind == 'matched':
                chosen = np.flatnonzero((np.asarray(d['source']) == snap['source']) & (np.asarray(d['frame']) == snap['frame']))
                index = {int(v): i for i, v in enumerate(physical['atom'])}
                observed = np.array([index[d['atom'][i]] for i in chosen], int)
                retained = np.unique(np.r_[retained, observed])
            values = np.empty((len(retained), 256), np.float32)
            labels = np.empty(len(patches), np.uint8)
            for start, ids, value in features(model, patches, np.arange(len(patches)), scale, c['batch_size']):
                labels[ids] = assign(value, centers)
                take = (retained >= start) & (retained < start+len(ids))
                values[take] = value[retained[take]-start]
                if start % 16384 == 0:
                    print(f'{kind}/{key}: {start}/{len(patches)}; {time.monotonic()-started:.1f}s', flush=True)
            if kind == 'matched':
                with np.load(held) as z:
                    expected = z['labels'][chosen]
                if np.any(labels[observed] != expected):
                    raise ValueError(f'MACE labels differ between assay and dense MD: {key}')
            np.savez_compressed(dest/'labels.npz', encoder=labels)
            travel_z = values[np.searchsorted(retained, travel_rows)]
            t = torch.as_tensor(travel_z, device='cuda')
            distances = torch.cdist(t, t); distances.fill_diagonal_(float('inf'))
            np.savez_compressed(dest/'travel.npz', z=travel_z, rows=travel_rows,
                                neighbors=distances.topk(16, largest=False).indices.cpu().numpy(), labels=labels[travel_rows])
            if kind == 'static':
                part = dict(z=values[np.searchsorted(retained, sampled)], labels=labels[sampled],
                            frame=np.full(len(sampled), snap['frame']), atom=physical['atom'][sampled])
                np.savez_compressed(dest/'sampled.npz', **part); sampled_parts.append(part)
            write_json(dest/'complete.json', dict(rows=len(patches), checkpoint_sha256=c['checkpoint_sha256'],
                patch_sha256=sha(folder/'patches.npy'), physical_sha256=sha(folder/'physical.npz'),
                labels_sha256=sha(dest/'labels.npz'), travel_sha256=sha(dest/'travel.npz')))
            print(f'Completed frozen inference: {kind}/{key}', flush=True)
        if kind == 'static':
            np.savez_compressed(data/'static.npz', **{k: np.concatenate([p[k] for p in sampled_parts]) for k in sampled_parts[0]})
    write_json(out/'technical/inference.json', dict(checkpoint=c['checkpoint'], checkpoint_sha256=c['checkpoint_sha256'],
        model=c['model'], neural_training=False, clustering='K7, original train-uniform rows only; raw exported state',
        encoder_inputs='centered nearest-80 coordinates, fixed Al material normalization; radius8/cutoff5; constant atom channel',
        normalization_factor=scale, predictor_inputs=None, history=0, motion=False, conditions=[],
        exported_state='256 scalar coordinates after frozen scalar_mean/scalar_scale; no descriptor predictions',
        precision='training BF16 autocast, float32 exports', inference_chunk=c['batch_size'],
        device=torch.cuda.get_device_name(), seconds=time.monotonic()-started))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['infer', 'publish']); p.add_argument('--config', required=True)
    a = p.parse_args()
    if a.action == 'infer': infer(a.config)
    else:
        from .mace_publication import publish
        publish(a.config)
