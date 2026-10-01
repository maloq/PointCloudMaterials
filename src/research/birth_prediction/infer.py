"""Standalone inference in the checkpoint's recorded source tree; no fitting."""
import json
from pathlib import Path
import sys


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    spec = request['model']
    producer = Path(spec['producer'])
    # Frozen code owns every model import. This standalone file does not import
    # the current repository package before switching to that recorded producer.
    # The sibling queue.py must not shadow Python's standard-library queue.
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(producer)] + [p for p in sys.path if p != script_dir]
    import numpy as np
    import torch
    from src.data.fixed_cohort.protocol import sha
    from src.research.supervised_onset.model import CapacityEncoder
    from src.research.encoder_context.geometry import graph

    if sha(Path(spec['checkpoint'])) != spec['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint checksum changed')
    for name, expected in spec['inference_dependencies'].items():
        if sha(producer / name) != expected:
            raise ValueError(f'Frozen native inference dependency changed: {name}')
    saved = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    factor = saved['coordinate_normalization']['reference_scale_A'] / saved['coordinate_normalization']['scales_A']['Al']
    if spec['kind'] == 'rich':
        from src.research.liquid_predictability.rich_multimaterial_train import RichPatchMACE
        config = saved['config']
        config['patch_chunk'] = request['batch_size']
        model = RichPatchMACE(config, len(saved['model']['output_mask'])).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        dimensions = model.latent_dim
    elif spec['kind'] == 'vicreg':
        model = CapacityEncoder(**saved['encoder_config']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['encoder'], strict=True)
        dimensions = model.projection.out_features
    else:
        raise ValueError(spec['kind'])
    positions = np.load(request['positions'], mmap_mode='r')
    batch = request['batch_size']
    output = np.lib.format.open_memmap(request['destination'], mode='w+', dtype=np.float32,
                                      shape=(len(positions), dimensions))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for begin in range(0, len(positions), batch):
            part = np.array(positions[begin:begin + batch], dtype=np.float32, copy=True) * factor
            if part.shape[1:] != (80, 3) or np.any(part[:, 0]):
                raise ValueError('Expected original centered nearest-80 observations')
            count = len(part)
            if count < batch:
                pad = np.full((batch - count, 80, 3), 100., np.float32)
                pad[:, 0] = 0
                part = np.concatenate((part, pad))
            xyz = torch.as_tensor(part, device='cuda')
            if spec['kind'] == 'rich':
                z, _ = model.encode(xyz)
            else:
                z = model(graph(xyz, model)).float()
            z = z[:count].cpu().numpy()
            if not np.isfinite(z).all():
                raise FloatingPointError('Nonfinite frozen MACE embedding')
            output[begin:begin + count] = z
            if begin % (batch * 16) == 0:
                output.flush()
                print(json.dumps(dict(model=spec['name'], patches=begin + count, total=len(positions))), flush=True)
    output.flush()


if __name__ == '__main__':
    main(sys.argv[1])
