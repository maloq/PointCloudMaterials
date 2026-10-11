"""Local full-step batch benchmark of a frozen Ni latent-simulator producer.

Run this file directly: imports must come from --bundle, not the live checkout.
Scientific checkpoints and W&B are never written by this benchmark.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle))
    import numpy as np
    import torch
    from src.research.ni_latent_sim.data import load, read, root, sha, write_json
    from src.research.ni_latent_sim.model import make_model, nll
    from src.research.ni_latent_sim.dense_response import bank as dense_bank, objective

    torch.set_num_threads(2)
    c = read(args.bundle/'config.json')
    p = read(root(c)/'technical/plan.json')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data = load(c, 'data'); bank = dense_bank(c, p)
    mean = data['mean'].cuda(); scale = data['scale'].cuda()
    x = (data['x'].cuda()-mean)/scale
    ids = np.flatnonzero([r['role'] == 'train' for r in data['rows']])
    shortids = np.flatnonzero([r['role'] == 'train' for r in bank])
    receipt = dict(bundle=str(args.bundle), checkpoint=str(args.checkpoint),
                   checkpoint_sha256=sha(args.checkpoint), checkpoint_update=saved['update'],
                   gpu=torch.cuda.get_device_name(), online_runs_created=0,
                   source_hashes={name: sha(args.bundle/'src/research/ni_latent_sim'/name)
                                  for name in ['train.py', 'model.py', 'fine_model.py', 'transformer.py', 'dense_response.py']},
                   note='Clones one 1ps values checkpoint; response on/off is a hardware-only benchmark.', results=[])
    for response, batch, teacher in [(False, 256, 16), (False, 1024, 64),
                                    (True, 256, 16), (True, 1024, 64),
                                    (False, 256, 16)]:
        arm = dict(saved['arm'], response=response)
        torch.manual_seed(91824)
        model = make_model(arm).cuda(); model.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['training']['lr'], weight_decay=1e-5)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        rng = np.random.default_rng(91824)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(8):
            torch.cuda.synchronize(); begin = time.monotonic()
            model.train(); optimizer.zero_grad(set_to_none=True)
            rows = torch.as_tensor(rng.choice(ids, batch), device='cuda')
            h = x[rows, :4]; target = x[rows][:, [4, 8, 13]]
            g = torch.Generator(device='cuda').manual_seed(int(rng.integers(2**31)))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mu, lv = model.rollout(h, 10, 4, g)
                now, _ = model.emit(model.encode(h[:, -1]))
            likelihood = nll(mu[:, :, [0, 4, 9]].float(), lv[:, :, [0, 4, 9]].float(), target).mean()
            reconstruction = .5*(now.float()-h[:, -1]).square().mean()
            (likelihood+.2*reconstruction).backward()
            records = [bank[i] for i in rng.choice(shortids, teacher)]
            v, d = objective(model, records, mean, scale, arm, int(rng.integers(2**31)))
            (.1*v+.1*d).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            times.append(time.monotonic()-begin)
        median = float(np.median(times[3:]))
        row = dict(response=response, batch_size=batch, teacher_batch=teacher,
                   update_seconds=times, warmed_median_seconds=median,
                   examples_per_second=batch/median, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                   final_nll=float(likelihood.detach()), final_gradient_norm=float(norm))
        receipt['results'].append(row); write_json(args.output, receipt)
        print(json.dumps(row), flush=True)
        del model, optimizer, mu, lv, now, likelihood, reconstruction, v, d
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
