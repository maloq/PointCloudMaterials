"""Frozen-tree inference for local transfer of response MACE and matched GeoFormer."""
import json
import hashlib
from pathlib import Path
import sys


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    spec = request['model']
    sys.path[:] = [spec['producer']] + [p for p in sys.path if p != str(Path(__file__).parent)]
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(request['seed'])
    if file_hash(spec['checkpoint']) != spec['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    for relative, expected in spec['inference_dependencies'].items():
        if file_hash(Path(spec['producer'])/relative) != expected:
            raise ValueError(f'Frozen inference source changed: {relative}')
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    positions = np.load(request['positions'], mmap_mode='r')
    batch = request['batch_size']
    checks = {}
    if spec['kind'] == 'response':
        from src.research.response_training.model import Predictor
        config = json.loads((Path(spec['producer'])/'config.json').read_text())
        model = Predictor(config, saved['model']['box']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        encoder = model.encoder

        def export(x, box=None):
            count, atoms, _ = x.shape
            delta = x[:, None] - x[:, :, None]
            shifts = torch.zeros_like(delta) if box is None else box*torch.round(delta/box)
            allowed = (delta-shifts).norm(dim=-1) < encoder.cutoff
            allowed &= ~torch.eye(atoms, device=x.device, dtype=torch.bool)[None]
            b, i, j = allowed.nonzero(as_tuple=True)
            edge = torch.stack((b*atoms+i, b*atoms+j))
            xyz = x.flatten(0, 1)
            vectors = xyz[edge[1]]-xyz[edge[0]]-shifts[b, i, j]
            attrs = xyz.new_ones(len(xyz), 1)
            radial, cutoff = encoder.radial_embedding(vectors.norm(dim=-1, keepdim=True),
                attrs, edge, encoder.atomic_numbers)
            if cutoff is not None:
                raise ValueError('Unexpected radial cutoff contract')
            graph = dict(attrs=attrs, center=torch.zeros_like(attrs), weight=xyz.new_ones(len(xyz)),
                edge=edge, angular=encoder.spherical_harmonics(vectors), radial=radial,
                group=torch.arange(count, device=x.device).repeat_interleave(atoms),
                centers=torch.arange(count, device=x.device)*atoms, size=count)
            scalar = encoder.atom_features(graph)[:, :encoder.channels].reshape(count, atoms, encoder.channels)
            return encoder.export_pooled(torch.cat((scalar.mean(1), scalar.var(1, unbiased=False)), -1))

        parents = torch.load(spec['native_parents'], map_location='cpu', weights_only=False)['parents']
        native = torch.stack([p['q'] for p in parents[:2]]).float().cuda()
        with torch.no_grad():
            reference, adapted = model.encode(native), export(native, model.box)
            checks['native_adapter_max_abs_error'] = float((reference-adapted).abs().max())
            checks['native_adapter_relative_l2_error'] = float((reference-adapted).norm()/reference.norm().clamp_min(1e-12))
        if checks['native_adapter_relative_l2_error'] > 1e-5:
            raise ValueError(f'Adapter does not reproduce native full-cell export: {checks}')
        dimensions = 128
        infer = export
    elif spec['kind'] == 'geoformer':
        from omegaconf import OmegaConf
        from src.research.spatial_vicreg_bias.train import PairEncoder
        if saved['epoch'] != 24 or saved['offset'] != 0:
            raise ValueError('Expected completed GeoFormer epoch24')
        model = PairEncoder(OmegaConf.create(saved['recipe'])).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        dimensions = 128
        def infer(x):
            return model.encoder.forward_features(x/spec['length_scale_A'])
    else:
        raise ValueError(spec['kind'])
    output = np.lib.format.open_memmap(request['destination'], mode='w+', dtype=np.float32,
                                      shape=(len(positions), dimensions))
    with torch.no_grad():
        for first in range(0, len(positions), batch):
            part = np.array(positions[first:first+batch], copy=True)
            if part.shape[1:] != (80, 3) or np.any(part[:, 0]) or np.max(np.linalg.norm(part, axis=-1)) >= 8:
                raise ValueError('Expected centered nearest80 Al neighborhoods entirely within8A')
            x = torch.as_tensor(part, device='cuda')
            z = infer(x).float().cpu().numpy()
            if not np.isfinite(z).all() or z.shape != (len(x), dimensions):
                raise ValueError('Invalid embedding')
            output[first:first+len(x)] = z
            if first == 0:
                repeated = infer(x).float().cpu().numpy()
                checks['same_batch_repeat_max_abs_error'] = float(np.max(np.abs(z-repeated)))
                checks['same_batch_repeat_relative_l2_error'] = float(np.linalg.norm(z-repeated)/max(np.linalg.norm(z),1e-12))
                # Native CUDA scatter reductions differ at float32 roundoff;
                # local-transfer output magnitude need not match native cells.
                if checks['same_batch_repeat_relative_l2_error'] > 1e-5:
                    raise ValueError(f'Unstable deterministic inference: {checks}')
            if first % (batch*16) == 0:
                print(dict(model=spec['name'], completed=first+len(x), total=len(positions)), flush=True)
                output.flush()
    output.flush()
    write_json(Path(request['destination']).with_suffix('.checks.json'), checks)


if __name__ == '__main__':
    main(sys.argv[1])
