"""Frozen-v6 and descriptor noise exports; launched in their explicit producers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--record', required=True)
    args = parser.parse_args(); task = json.loads(Path(args.record).read_text())
    sys.path.insert(0, task['producer'])
    import numpy as np
    for path, expected in task['producer_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Producer changed: {path}')
    if task['kind'] == 'v6':
        import torch
        from src.models.encoders.structural import StructuralMACE, StructuralGATr
        from src.research.trajectory_stability.encode import observation
        from src.data.structural_pretraining.batches import collate, move
        from src.training_methods.shared_pretraining.compilation import compile_encoder
        torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        if hashlib.sha256(Path(task['checkpoint']).read_bytes()).hexdigest() != task['checkpoint_sha256']:
            raise ValueError('Changed historical checkpoint')
        saved = torch.load(task['checkpoint'], map_location='cpu', weights_only=False)
        model = (StructuralMACE() if task['architecture'] == 'mace' else StructuralGATr()).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['encoder'], strict=True)
        initialized = False

        def encode(data):
            nonlocal initialized
            values = []
            with torch.inference_mode():
                for first in range(0, len(data['centers']), task['batch_size']):
                    ix = range(first, min(first+task['batch_size'], len(data['centers'])))
                    samples = [observation(data['positions'][data['offsets'][i]:data['offsets'][i+1]],
                        int(data['centers'][i]), task['scale'], task['architecture']) for i in ix]
                    batch = move(collate(samples, task['architecture']), 'cuda')
                    if not initialized:
                        compile_encoder(model, batch, saved['identity']['config']['precision']); initialized = True
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=saved['identity']['config']['precision']=='bf16'):
                        values.append(model(batch).float().cpu().numpy())
            return {task['architecture']: np.concatenate(values)}
    elif task['kind'] == 'descriptors':
        from ase import Atoms
        from dscribe.descriptors import SOAP
        from scipy.spatial import cKDTree
        from src.analysis.liquid_structure import persistence_image, bond_order
        from src.data.structural_pretraining.prepare import geometry_packet
        soap = SOAP(species=['Al'], periodic=False, r_cut=7., n_max=8, l_max=6, sigma=.3,
                    sparse=False, dtype='float64')

        def encode(data):
            out = {k: [] for k in ['tda','soap','bond_order','radial','angular']}
            for i, (a, b) in enumerate(zip(data['offsets'][:-1], data['offsets'][1:], strict=True)):
                x = data['positions'][a:b]; center = int(data['centers'][i])
                tree = cKDTree(x)
                _, neighborhood = tree.query(x[center], k=13)
                if neighborhood[0] != center:
                    raise ValueError('Descriptor center/nearest-neighbor identity lost')
                _, near = tree.query(x[neighborhood], k=13)
                vectors = x[near[:,1:]].astype(float)-x[neighborhood,None].astype(float)
                order, _ = bond_order(vectors[None], 3.5)
                geometry = geometry_packet(x)
                out['tda'].append(persistence_image(data['nearest80'][i]))
                out['soap'].append(soap.create(Atoms('Al'*len(x), positions=x), centers=[[0.,0.,0.]])[0])
                out['bond_order'].append(order[0,:6]); out['radial'].append(geometry[:32]); out['angular'].append(geometry[64:80])
            return {k: np.stack(v) for k,v in out.items()}
    else:
        raise ValueError(f'Unknown noise worker kind: {task["kind"]}')
    inputs, output = Path(task['inputs']), Path(task['destination'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inputs/'manifest.json').read_text()); timings = {}
    for frame in manifest['frames']:
        name = f'frame-{frame["frame_index"]:02d}.npz'
        if hashlib.sha256((inputs/name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError(f'Changed noise input: {name}')
        started = time.monotonic()
        with np.load(inputs/name) as data:
            result = encode(data)
        if any(not np.isfinite(v).all() for v in result.values()):
            raise FloatingPointError(f'Nonfinite noise features: {name}')
        np.savez(output/name, **result); timings[name] = time.monotonic()-started
        print(task['kind'], name, timings[name], flush=True)
    (output/'extraction.json').write_text(json.dumps(dict(state='complete', timings=timings,
        task_sha256=hashlib.sha256(Path(args.record).read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
