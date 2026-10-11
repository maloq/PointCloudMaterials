"""Frozen continuous-motion inference on exact pre-birth Q,V observations."""
import json
from pathlib import Path
import sys


def main(request_path):
    r = json.loads(Path(request_path).read_text())
    producer = Path(r['producer']).resolve()
    sys.path.insert(0, str(producer))
    import numpy as np
    import torch
    from src.data.fixed_cohort.protocol import sha, write_json
    from src.research.continuous_motion.model import MotionModel
    import src.research.continuous_motion.model as implementation
    if Path(implementation.__file__).resolve() != producer/'src/research/continuous_motion/model.py':
        raise ValueError('Wrong frozen motion implementation')
    for name, expected in r['dependencies'].items():
        if sha(producer/name) != expected:
            raise ValueError('Changed frozen dependency: '+name)
    spec = r['model']
    if sha(spec['checkpoint']) != spec['sha256']:
        raise ValueError('Changed selected checkpoint')
    cfg = json.loads((producer/'config.json').read_text())
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = MotionModel(cfg, spec['arm']).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(spec['checkpoint'], map_location='cuda', weights_only=False)['model'])
    # Preserve the original256-sample noise launch. CUDA normal generation uses
    # a hardware-dependent grid: e.g. A100 reuses rand.y after sample216 while
    # RTX uses rand.x through256. The first64 use the same Philox subsequences
    # on the campaign GPUs. Compare that common prefix, and retain the full
    # block difference as an audit diagnostic (not a portable exact replay).
    old = np.load(r['audit_predictions'])
    data = torch.load(r['audit_dataset'], map_location='cpu', weights_only=False)
    ids = old['row'][:cfg['training']['microbatch']]
    model.inference_microbatch = r['batch_size']
    with torch.no_grad():
        z = model.encode(data['state'][ids].cuda(), create_graph=False)
        mu, _ = model.prediction(model.path(z, cfg['horizons_ps'], particles=8, seed=cfg['seed']+111))
    difference = np.abs(mu.mean(1).cpu().numpy()-old['mean'][:len(ids)])
    audit_rows = min(64,len(ids))
    error = float(difference[:audit_rows].max())
    full_block_error = float(difference.max())
    if error > 2e-4:
        raise ValueError(f'Frozen future-prediction replay differs: {error}')
    del data, z, mu
    q = np.load(r['positions'], mmap_mode='r')
    v = np.load(r['velocities'], mmap_mode='r')
    if q.shape != v.shape or q.shape[1:] != (80, 3):
        raise ValueError('Wrong local Q,V shape')
    if r['stored_velocity_units'] != 'angstrom_per_fs':
        raise ValueError('Birth extraction velocity units changed')
    out = Path(r['destination'])
    arrays = {name:np.lib.format.open_memmap(out/(name+'.npy'), mode='w+', dtype=np.float32,
              shape=shape) for name, shape in [('structural',(len(q),128)),
              ('structural_motion',(len(q),256)),('future_law',(len(q),len(r['leads_ps']),48))]}
    for start in range(0, len(q), r['batch_size']):
        stop = min(start+r['batch_size'],len(q))
        # The established birth extractor stores A/fs; MotionModel requires A/ps.
        state = torch.from_numpy(np.concatenate((q[start:stop],v[start:stop]*np.float32(1000)), -1)).cuda()
        with torch.no_grad():
            z = model.encode(state,create_graph=False)
            mu, lv = model.prediction(model.path(z,r['leads_ps'],particles=8,seed=r['seed']))
            mean = mu.mean(1)
            variance = (lv.exp()+mu.square()).mean(1)-mean.square()
            law = torch.cat((mean,variance.clamp_min(1e-8).log()),-1)
        values = dict(structural=z[:,:128],structural_motion=z,future_law=law)
        for name, value in values.items():
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite '+name)
            arrays[name][start:stop] = value.cpu().numpy()
        write_json(out/'progress.json',dict(model=spec['name'],observations=stop,total=len(q)))
    for array in arrays.values():array.flush()
    write_json(out/'complete.json',dict(reproduction_max_abs=error,reproduction_rows=audit_rows,
        full_stochastic_block_max_abs=full_block_error,
        replay_protocol='first64 common Philox subsequences of original256-sized block; CUDA grid depends on GPU',
        velocity_conversion='A/fs *1000 -> A/ps',
        checkpoint_sha256=spec['sha256'],files={name+'.npy':sha(out/(name+'.npy')) for name in arrays}))


if __name__ == '__main__':
    main(sys.argv[1])
