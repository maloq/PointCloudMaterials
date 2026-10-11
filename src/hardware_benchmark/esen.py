"""Real-data eSEN execution benchmark and saved-gradient numerical comparison."""
import argparse
from contextlib import nullcontext
import time
from pathlib import Path

import numpy as np
import torch

from src.models.encoders.esen_runtime import compile_blocks, direct_backward, index_permutations
from src.research.spatial_architecture.model import Spatial, teacher
from src.research.structural_response.model import spatial_backward
from src.research.structural_response.data import spatial_data, read, write_json, sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['eager','compiled','direct','compiled-direct','indexed','compiled-indexed','indexed-direct','compiled-indexed-direct'], required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--microbatch', type=int, default=256)
    p.add_argument('--architecture', choices=['esen','mace'], default='esen')
    a = p.parse_args();c = read(a.config);a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = Spatial(c, a.architecture).cuda();model.load_state_dict(saved['model'])
    ema = teacher(c, a.architecture, model.geometry);ema.load_state_dict(saved['ema'])
    opt = torch.optim.AdamW(model.parameters(), lr=c['spatial_training']['learning_rate'],
                           weight_decay=c['spatial_training']['weight_decay'])
    opt.load_state_dict(saved['optimizer'])
    if a.architecture == 'mace' and a.variant not in ('eager','direct'):
        raise ValueError('MACE benchmark retains its existing cuEq backend')
    if 'indexed' in a.variant:index_permutations(model.geometry)
    if a.variant.startswith('compiled'):compile_blocks(model.geometry)
    data, columns, weights = spatial_data(c)
    weights = torch.from_numpy(weights).cuda()
    ids = np.random.default_rng(57201).permutation(len(data))[:8*256]
    times = [];startup=time.monotonic()
    for step in range(8):
        torch.cuda.synchronize();begin=time.monotonic()
        b = data.batch(ids[step*256:(step+1)*256])
        q = torch.from_numpy(b['positions']).cuda()
        y = torch.from_numpy(b['target'][:,columns]).cuda()
        opt.zero_grad(set_to_none=True)
        profiling = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA]) if a.profile and step == 0 else nullcontext()
        with profiling:
            if a.variant.endswith('direct'):
                loss, stats = direct_backward(model, q, y, weights, 7391+step)
            else:
                loss, stats = spatial_backward(model, ema, q, y, weights, 'vicreg_descriptors', 7391+step, a.microbatch)
        if a.profile and step == 0:
            (a.output/(a.variant+'-profile.txt')).write_text(profiling.key_averages().table(
                sort_by='self_cuda_time_total',row_limit=35))
        if step == 0:
            gradients = torch.cat([v.grad.flatten() for v in model.parameters() if v.grad is not None]).detach().cpu().numpy()
            np.save(a.output/(a.variant+'-gradient.npy'), gradients)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):embedding=model.geometry(q[:8]).float().cpu().numpy()
            np.save(a.output/(a.variant+'-embedding.npy'),embedding)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            for x,z in zip(ema.parameters(),model.geometry.parameters(),strict=True):x.lerp_(z,.01)
        torch.cuda.synchronize();times.append(time.monotonic()-begin)
        print(a.variant,step,loss,times[-1],float(norm),flush=True)
    write_json(a.output/(a.variant+'.json'),dict(variant=a.variant,architecture=a.architecture,microbatch=a.microbatch,checkpoint_sha256=sha(a.checkpoint),
        checkpoint_update=saved['update'],update_seconds=times,warmed_median_seconds=float(np.median(times[3:])),
        benchmark_elapsed_seconds=time.monotonic()-startup,peak_memory_GB=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(),loss=loss,online_runs_created=0))
    data.close()


if __name__ == '__main__':main()
