"""Inference of birth observations using the exact frozen response-training source."""
import json
from pathlib import Path
import sys


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    producer = Path(request['producer']).resolve()
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(producer)] + [p for p in sys.path if p != here]
    import numpy as np
    import torch
    from src.data.fixed_cohort.protocol import sha, write_json
    from src.research.response_velocity.model import Predictor
    import src.research.response_velocity.model as implementation

    if Path(implementation.__file__).resolve() != producer/'src/research/response_velocity/model.py':
        raise ValueError('Wrong frozen response producer')
    spec = request['model']
    if sha(Path(spec['checkpoint'])) != spec['sha256']:
        raise ValueError('Changed response checkpoint')
    for filename, expected in request['dependencies'].items():
        if sha(producer/filename) != expected:
            raise ValueError('Changed frozen inference dependency: '+filename)
    cfg = json.loads((producer/'config.json').read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Predictor(cfg, spec['velocities']).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)['model'], strict=True)
    q = np.load(request['positions'], mmap_mode='r')
    v = np.load(request['velocities'], mmap_mode='r')
    if q.shape != v.shape or q.shape[1:] != (80, 3):
        raise ValueError('Coordinate and actual-velocity observations differ')
    # Replay previously saved held-out response inputs before transferring domains.
    audit = np.load(request['audit'])
    old = np.load(request['audit_reference'])
    with torch.inference_mode():
        audit_z = model.encode(torch.from_numpy(audit['state']).cuda())
        audit_p = model.head(audit_z).cpu().numpy()
    error = float(np.max(np.abs(audit_p-old['prediction'][:len(audit_p)])))
    if error > 2e-4:
        raise ValueError(f'Frozen checkpoint replay differs: {error}')
    out = Path(request['destination'])
    z = np.lib.format.open_memmap(out/'embedding.npy', mode='w+', dtype=np.float32, shape=(len(q),128))
    p = np.lib.format.open_memmap(out/'prediction_head.npy', mode='w+', dtype=np.float32, shape=(len(q),256))
    with torch.inference_mode():
        for start in range(0, len(q), request['batch_size']):
            stop = min(start+request['batch_size'], len(q))
            state = torch.from_numpy(np.concatenate((q[start:stop],v[start:stop]),-1)).cuda()
            encoded = model.encode(state)
            zz, pp = encoded.cpu().numpy(), model.head(encoded).cpu().numpy()
            if not np.isfinite(zz).all() or not np.isfinite(pp).all():
                raise FloatingPointError('Nonfinite birth representation')
            z[start:stop], p[start:stop] = zz, pp
            if start % (request['batch_size']*16) == 0:
                print(spec['name'], stop, '/',len(q), flush=True)
    z.flush(); p.flush()
    write_json(out/'complete.json', dict(reproduction_max_abs=error,
        files={f:sha(out/f) for f in ['embedding.npy','prediction_head.npy']}))


if __name__ == '__main__':
    main(sys.argv[1])
