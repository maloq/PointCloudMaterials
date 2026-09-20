"""Run under each checkpoint's frozen training checkout, never the live model code."""
import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.data.structural_pretraining.batches import collate, move
from src.models.encoders.structural import StructuralGATr, StructuralMACE
from src.research.trajectory_stability.encode import observation
from src.training_methods.shared_pretraining.compilation import compile_encoder


@torch.no_grad()
def extract(config, architecture):
    if socket.gethostname().split('.')[0] != config['required_hostname'] or config['required_gpu'] not in torch.cuda.get_device_name(0):
        raise RuntimeError('Comparison worker is on the wrong requested hardware')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    root = Path(config['output'])/'technical'
    plan = json.loads((root/'plan.json').read_text())
    record = plan['checkpoints'][architecture]
    if file_hash(record['path']) != record['sha256']:
        raise ValueError('Frozen checkpoint export changed')
    producer = Path(record['producer_code'])
    for name, expected in record['identity']['implementation']['files'].items():
        p = Path(name)
        if file_hash(p if p.is_absolute() else producer/p) != expected:
            raise ValueError(f'Training code changed: {name}')
    import src.models.encoders.structural as implementation
    if Path(implementation.__file__).resolve() != producer/'src/models/encoders/structural.py':
        raise ValueError('Did not import the frozen native encoder')
    saved = torch.load(record['path'], map_location='cpu', weights_only=False)
    model = (StructuralMACE() if architecture == 'mace' else StructuralGATr()).cuda().eval()
    model.load_state_dict(saved['encoder'], strict=True)
    precision = saved['identity']['config']['precision']
    initialized = False
    batch_size = config['batch_size']
    def evaluate(samples):
        nonlocal initialized
        batch = move(collate(samples, architecture), 'cuda')
        if not initialized:
            compile_encoder(model, batch, precision)
            initialized = True
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            z = model(batch).float().cpu().numpy()
        if z.shape != (len(samples), 128) or not np.isfinite(z).all():
            raise FloatingPointError('Invalid native exported state')
        return z
    audited = False
    for kind in ('temporal', 'spatial'):
        for path in sorted((root/'inputs'/kind).glob('*.npz')):
            dest = root/'features'/architecture/kind/path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.with_suffix('.json').exists():
                receipt = json.loads(dest.with_suffix('.json').read_text())
                if file_hash(dest) != receipt['sha256'] or receipt['checkpoint_sha256'] != record['sha256']:
                    raise ValueError('Completed extraction is inconsistent')
                continue
            receipt = json.loads(path.with_suffix('.json').read_text())
            if file_hash(path) != receipt['sha256']:
                raise ValueError('Prepared observation changed')
            a = np.load(path)
            started = time.monotonic()
            output = {}
            for mode, key in (('original', 'positions'), ('radial', 'radial_positions')):
                values = []
                for start in range(0, len(a['center_indices']), batch_size):
                    end = min(start+batch_size, len(a['center_indices']))
                    samples = [observation(a[key][a['offsets'][i]:a['offsets'][i+1]],
                        int(a['center_indices'][i]) if mode == 'original' else 0, plan['scale'], architecture) for i in range(start, end)]
                    z = evaluate(samples)
                    if not audited:
                        repeat = evaluate(samples)
                        reordered = evaluate(samples[::-1])[::-1]
                        np.testing.assert_allclose(z, repeat, rtol=2e-5, atol=2e-6)
                        np.testing.assert_allclose(z, reordered, rtol=2e-5, atol=2e-6)
                        rotation = Rotation.random(random_state=config['seed']).as_matrix().astype(np.float32)
                        rotated = [observation(a[key][a['offsets'][i]:a['offsets'][i+1]]@rotation.T,
                            int(a['center_indices'][i]), plan['scale'], architecture) for i in range(start, end)]
                        zr = evaluate(rotated)
                        np.testing.assert_allclose(z, zr, rtol=2e-4, atol=2e-4)
                        save_json(root/f'{architecture}-numerical-audit.json', dict(repeat_max_abs=float(np.max(np.abs(z-repeat))),
                            reorder_max_abs=float(np.max(np.abs(z-reordered))), rotation_max_abs=float(np.max(np.abs(z-zr))),
                            rotation_relative_rms=float(np.linalg.norm(z-zr)/np.linalg.norm(z)),
                            producer_code=str(producer), precision=precision, execution='compiled',
                            host=socket.gethostname(), gpu=torch.cuda.get_device_name(0)))
                        audited = True
                    values.append(z)
                output[mode] = np.concatenate(values)
            np.savez(dest, **output)
            save_json(dest.with_suffix('.json'), dict(sha256=file_hash(dest), checkpoint_sha256=record['sha256'],
                input_sha256=receipt['sha256'], rows=len(output['original']), seconds=time.monotonic()-started))
            print(f'{architecture} {kind}/{path.stem}: {len(output["original"])} rows, {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--architecture', choices=('gatr', 'mace'), required=True)
    args = parser.parse_args()
    extract(json.loads(Path(args.config).read_text()), args.architecture)
