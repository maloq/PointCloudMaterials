"""Native checkpoint inference on centered, periodic, tracked observations."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.data.structural_pretraining.support import local_crop, support_weights, EDGE_CUTOFF
from src.data.structural_pretraining.batches import collate, move
from src.data.structural_pretraining.prepare import REFERENCE_RADIUS, file_hash, save_json
from src.models.encoders.structural import StructuralGATr, StructuralMACE, ATOMIC_NUMBERS, ARCHITECTURE_REVISION
from src.models.encoders.mixed_gatr import GATR_BOND_REVISION
from src.models.encoders.mixed_mace import MACE_BOND_REVISION
from src.training_methods.shared_pretraining.compilation import compile_encoder


def observation(positions, center, scale, architecture):
    x, rows = local_crop(positions, scale)
    center = int(np.flatnonzero(rows == center).item())
    if not np.all(x[center] == 0):
        raise ValueError('Tracked center no longer at the origin')
    result = dict(positions=x[None], weights=support_weights(x)[None],
        center=int(center), times=np.zeros(1, np.float32), species=ATOMIC_NUMBERS.index(13),
        log_scale=np.log(scale/REFERENCE_RADIUS), physical=np.zeros(85, np.float32),
        tda=np.zeros(144, np.float32), tda_valid=False)
    if architecture == 'mace':
        pairs = cKDTree(x).query_pairs(EDGE_CUTOFF, output_type='ndarray')
        if np.any(np.linalg.norm(x[pairs[:, 0]]-x[pairs[:, 1]], axis=-1) <= 0):
            raise ValueError('Coincident atoms in a MACE local graph')
        result['edges'] = np.concatenate((pairs, pairs[:, ::-1]), axis=0).T.astype(np.int64)
    return result


@torch.no_grad()
def encode(plan):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = plan['config']['device']
    batch_size = plan['config']['batch_size']
    root = Path(plan['config']['output'])/'technical'
    for name, record in plan['checkpoints'].items():
        if file_hash(record['path']) != record['sha256']:
            raise ValueError(f'Frozen checkpoint changed: {name}')
        saved = torch.load(record['path'], map_location='cpu', weights_only=False)
        architecture = saved['architecture']
        revision = (MACE_BOND_REVISION if architecture=='mace' else GATR_BOND_REVISION
            ) if saved['identity']['config'].get('batch_mode')=='mixed_triplets' else ARCHITECTURE_REVISION
        if saved['identity']['architecture_revision'] != revision:
            raise ValueError('Unsupported checkpoint architecture revision; local support required')
        model = (StructuralMACE() if architecture == 'mace' else StructuralGATr()).to(device).eval()
        model.load_state_dict(saved['encoder'], strict=True)
        precision = saved['identity']['config']['precision']
        initialized = False
        for source in plan['sources']:
            folder = root/'sources'/str(source['id'])
            destination = folder/f'{name}.npy'
            if destination.exists():
                if not (folder/f'{name}-verification.json').exists():
                    raise RuntimeError(f'Unverified partial extraction: {destination}')
                continue
            a = np.load(folder/'observations.npz')
            positions = np.load(folder/'positions.npy', mmap_mode='r')
            values = []; started = time.monotonic()
            for start in range(0, len(a['center_indices']), batch_size):
                samples = [observation(positions[a['offsets'][i]:a['offsets'][i+1]], a['center_indices'][i],
                    plan['scale'], architecture) for i in range(start, min(start+batch_size, len(a['center_indices'])))]
                batch = move(collate(samples, architecture), device)
                if not initialized:
                    compile_encoder(model, batch, precision)
                    initialized = True
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    z = model(batch).float().cpu().numpy()
                if not np.isfinite(z).all():
                    raise FloatingPointError(f'Nonfinite {name} states at source {source["id"]}, row {start}')
                values.append(z)
                if start == 0:
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        repeat = model(batch).float().cpu().numpy()
                        reordered = model(move(collate(samples[::-1], architecture), device)).float().cpu().numpy()[::-1]
                    np.testing.assert_allclose(z, repeat, atol=2e-6, rtol=2e-5)
                    np.testing.assert_allclose(z, reordered, atol=2e-6, rtol=2e-5)
                    verification = dict(repeat_max_abs=float(np.max(np.abs(z-repeat))),
                        reorder_max_abs=float(np.max(np.abs(z-reordered))),
                        repeat_mse=float(np.mean((z.astype(float)-repeat)**2)),
                        reorder_mse=float(np.mean((z.astype(float)-reordered)**2)))
                if start % (batch_size*20) == 0:
                    save_json(root/'inference-progress.json', dict(method=name, source=source['id'],
                        rows_done=start+len(z), rows_total=len(a['center_indices']), seconds=time.monotonic()-started))
            result = np.concatenate(values)
            np.save(destination, result, allow_pickle=False)
            save_json(folder/f'{name}-verification.json', dict(**verification, precision=precision,
                checkpoint_sha256=record['sha256'], features_sha256=file_hash(destination),
                rows=len(result), seconds=time.monotonic()-started))
            print(f'{name} source {source["id"]}: {len(result)} observations, {time.monotonic()-started:.1f}s', flush=True)
        del model
        torch.cuda.empty_cache()
