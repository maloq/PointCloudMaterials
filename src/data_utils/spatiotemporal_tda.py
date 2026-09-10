"""Cache alpha-complex targets for the existing normalized three-view producer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

from src.analysis.liquid_structure import persistence_image
from src.experiment_runner.registry import sha256, write_json


def prepare(cfg):
    root = Path(cfg.data.cache_dir)
    out = Path(cfg.tda.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((root/'manifest.json').read_text())
    protocol = dict(source_root=str(root.resolve()), source_sha256=sha256(root/'manifest.json'),
                    temporal_lag_steps=cfg.data.temporal_lag_steps, points=80,
                    reference_radius_A=cfg.encoder.kwargs.reference_radius_A,
                    components=cfg.tda.components, fit_anchors=cfg.tda.fit_anchors,
                    seed=cfg.seed_everything)
    if (out/'manifest.json').exists():
        saved = json.loads((out/'manifest.json').read_text())
        if saved['protocol'] != protocol:
            raise ValueError(f'TDA cache protocol differs: {out}/manifest.json')
        for name, digest in saved['checksums'].items():
            if sha256(out/name) != digest:
                raise ValueError(f'TDA cache checksum mismatch: {out/name}')
        print(f'Verified existing TDA cache: {out}', flush=True)
        return
    started = time.monotonic()
    shards = []
    with ProcessPoolExecutor(max_workers=cfg.tda.workers) as pool:
        for record in source['shards']:
            views_path = root/record['views']
            pairs_path = root/record['pairs']
            rows = np.flatnonzero(np.load(pairs_path, mmap_mode='r')[:, 3] == cfg.data.temporal_lag_steps)
            clouds = np.load(views_path, mmap_mode='r')[rows].astype(np.float32)
            if clouds.shape[1:] != (3, 80, 3):
                raise ValueError(f'Expected three 80-point views in {views_path}, got {clouds.shape}')
            prefix = Path(record['views']).name.removesuffix('.views.npy')
            raw_path = out/f'{prefix}.images.npy'
            part_manifest = out/f'{prefix}.json'
            provenance = dict(source_views=record['views'], views_sha256=sha256(views_path),
                              pairs_sha256=sha256(pairs_path), protocol=protocol)
            if part_manifest.exists():
                part = json.loads(part_manifest.read_text())
                if part['provenance'] != provenance or part['sha256'] != sha256(raw_path):
                    raise ValueError(f'Invalid completed TDA shard: {part_manifest}')
            else:
                # Use exactly the decoded coordinates and shared length seen by MACE.
                # Training augmentation is intentionally absent from these fixed targets.
                clouds *= cfg.encoder.kwargs.reference_radius_A
                images = np.stack(list(pool.map(persistence_image, clouds.reshape(-1, 80, 3), chunksize=32)))
                np.save(raw_path, images.reshape(len(rows), 3, 144))
                write_json(part_manifest, dict(provenance=provenance, sha256=sha256(raw_path)))
            np.save(out/f'{prefix}.rows.npy', rows)
            shards.append(dict(source_views=record['views'], split=record['split'], rows=f'{prefix}.rows.npy',
                               images=raw_path.name, targets=f'{prefix}.targets.npy', samples=len(rows)))
            write_json(out/'status.json', dict(state='preparing', completed_shards=len(shards),
                       total_shards=len(source['shards']), seconds=time.monotonic()-started))
            print(json.dumps(shards[-1]), flush=True)
    # The same uniform training-only PCA32 target convention as the previous TDA task.
    train = [s for s in shards if s['split'] == 'train']
    ends = np.cumsum([s['samples'] for s in train])
    selected = np.random.default_rng(cfg.seed_everything+100).choice(int(ends[-1]), cfg.tda.fit_anchors, replace=False)
    fit = []
    for index, shard in enumerate(train):
        start = int(ends[index-1]) if index else 0
        local = selected[(selected >= start) & (selected < ends[index])] - start
        fit.append(np.load(out/shard['images'], mmap_mode='r')[local])
    fit = np.concatenate(fit).reshape(-1, 144)
    pca = PCA(n_components=cfg.tda.components, svd_solver='full').fit(fit)
    scale = np.maximum(np.sqrt(pca.explained_variance_), 1e-5).astype(np.float32)
    np.savez(out/'scaling.npz', mean=pca.mean_.astype(np.float32),
             components=pca.components_.astype(np.float32), std=scale)
    checksums = {'scaling.npz': sha256(out/'scaling.npz')}
    baseline = {}
    for shard in shards:
        raw = np.load(out/shard['images'], mmap_mode='r')
        target = ((raw-pca.mean_) @ pca.components_.T / scale).astype(np.float32)
        if not np.isfinite(target).all():
            raise FloatingPointError(f'Nonfinite transformed TDA targets: {shard["source_views"]}')
        np.save(out/shard['targets'], target)
        for key in ('targets', 'rows', 'images'):
            checksums[shard[key]] = sha256(out/shard[key])
        split = shard['split']
        total, count = baseline.get(split, (0., 0))
        baseline[split] = (total + float(np.square(target, dtype=np.float64).sum()), count + target.size)
    write_json(out/'manifest.json', dict(state='complete', protocol=protocol, shards=shards, checksums=checksums,
               explained_variance=float(pca.explained_variance_ratio_.sum()),
               constant_training_mean_mse={k:v[0]/v[1] for k,v in baseline.items()},
               target_definition='Full 80-point alpha-complex H0/H1/H2 images in shared normalized reference units; training-only PCA32 whitening; unaugmented views.'))
    write_json(out/'status.json', dict(state='complete', shards=len(shards),
               target_views=sum(3*s['samples'] for s in shards), seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Resolved training YAML with data and tda settings.')
    args = parser.parse_args()
    prepare(OmegaConf.load(args.config))
