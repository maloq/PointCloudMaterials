"""Measured FP32 execution comparisons on verified physical-history examples."""
import copy
import gc
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_table
from src.experiment_runner.registry import write_json
from .data import load
from .evaluate import encode, extract
from .objective import target_normalization, targets, task_objective
from .runtime import CausalRuntime
from .train import initialize


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()


def run(config, args):
    if not str(args.device).startswith('cuda'):
        raise ValueError('The causal runtime throughput benchmark requires a CUDA device')
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = config['benchmark']
    root = Path(config['output'])
    if root.exists():
        raise FileExistsError(f'Preserve existing benchmark: {root}')
    (root/'technical').mkdir(parents=True)
    samples, identity = load(config)
    norm = target_normalization(samples)
    training = [s for s in samples if s['split'] == 'train']
    rng = np.random.default_rng(config['seed'])
    order = rng.permutation(len(training))
    validation = [s for s in samples if s['split'] == 'val'][:spec['evaluation_samples']]
    before = gpu_processes()
    metadata = dict(pid=os.getpid(), device=torch.cuda.get_device_name(args.device),
                    total_vram_bytes=torch.cuda.get_device_properties(args.device).total_memory,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    precision='float32; TF32 disabled; no autocast', cache_identity=identity,
                    processes_before=before, config=config)
    write_json(root/'technical/metadata.json', metadata)
    rows = []
    for width in spec['widths']:
        recipe = copy.deepcopy(config); recipe['encoder']['channels'] = width
        model, heads = initialize(recipe, 'D', args.device)
        initial_model = copy.deepcopy(model.state_dict())
        initial_heads = copy.deepcopy(heads.state_dict())
        parameters = list(model.parameters())+list(heads.parameters())
        for batch_size in spec['batch_sizes']:
            batch = [training[int(i)] for i in order[:batch_size]]
            if len(batch) != batch_size:
                raise ValueError('Benchmark batch exceeds available training examples')
            reference_z, reference_grads = None, None
            for mode in ('reference', 'packed_host', 'packed_device'):
                runtime = None if mode == 'reference' else CausalRuntime(
                    samples, norm, model, args.device,
                    dict(batch_size=batch_size, residency='device' if mode == 'packed_device' else 'host'))
                model.load_state_dict(initial_model); heads.load_state_dict(initial_heads)
                model.train(); heads.train()
                optimizer = torch.optim.AdamW(parameters, lr=config['training']['learning_rate'],
                                              weight_decay=config['training']['weight_decay'])

                def step(update=True):
                    optimizer.zero_grad(set_to_none=True)
                    z = encode(model, batch, args.device) if runtime is None else runtime.encode(model, batch)
                    y = targets(batch, norm, args.device) if runtime is None else runtime.target(batch)
                    loss, _ = task_objective(heads(z, y['temperature']), y, **config['loss'])
                    loss.backward()
                    if update:
                        torch.nn.utils.clip_grad_norm_(parameters, config['training']['gradient_clip'],
                                                       error_if_nonfinite=True)
                        optimizer.step()
                    return z.detach()

                z = step(update=False)
                grads = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
                if mode == 'reference':
                    reference_z, reference_grads = z.clone(), grads
                    agreement = dict(output_max_abs=0., gradient_relative_l2=0.)
                else:
                    torch.testing.assert_close(z, reference_z, atol=2e-5, rtol=2e-4)
                    if [g is None for g in grads] != [g is None for g in reference_grads]:
                        raise AssertionError('Packing changed which parameters receive a gradient')
                    a = torch.cat([g.flatten() for g in grads if g is not None])
                    b = torch.cat([g.flatten() for g in reference_grads if g is not None])
                    relative = float((a-b).norm()/b.norm().clamp_min(1e-12))
                    if relative > 1e-3:
                        raise AssertionError(f'Packing changed gradient: relative L2={relative}')
                    agreement = dict(output_max_abs=float((z-reference_z).abs().max()), gradient_relative_l2=relative)
                    del a, b
                del grads, z
                for _ in range(spec['warmup_steps']):
                    step()
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                seconds = []
                for repeat in range(spec['repeats']):
                    start = time.perf_counter()
                    for _ in range(spec['timed_steps']):
                        step()
                    torch.cuda.synchronize(args.device)
                    seconds.append(time.perf_counter()-start)
                allocated = torch.cuda.max_memory_allocated(args.device)
                reserved = torch.cuda.max_memory_reserved(args.device)
                start = time.perf_counter()
                result = (extract(model, heads, validation, norm, args.device) if runtime is None
                          else runtime.extract(model, heads, validation))
                torch.cuda.synchronize(args.device)
                evaluation_seconds = time.perf_counter()-start
                if any(v is not None and not np.isfinite(v).all() for v in result.values()):
                    raise FloatingPointError('Nonfinite physical benchmark outputs')
                row = dict(width=width, batch_size=batch_size, mode=mode,
                           step_seconds=float(np.median(seconds))/spec['timed_steps'],
                           examples_per_second=batch_size*spec['timed_steps']/float(np.median(seconds)),
                           evaluation_examples_per_second=len(validation)/evaluation_seconds,
                           peak_allocated_GiB=allocated/2**30, peak_reserved_GiB=reserved/2**30,
                           repeat_seconds=seconds, **agreement)
                rows.append(row)
                write_json(root/'technical/results.json', rows)
                print('CAUSAL RUNTIME BENCHMARK', row, flush=True)
                del runtime, optimizer, result
                for p in parameters:
                    p.grad = None
                gc.collect(); torch.cuda.empty_cache()
        del model, heads, parameters, initial_model, initial_heads, reference_grads, reference_z
        gc.collect(); torch.cuda.empty_cache()
    metrics = {f'width{r["width"]}/batch{r["batch_size"]}/{r["mode"]}':
               {k: v for k, v in r.items() if k not in ('mode', 'repeat_seconds')} for r in rows}
    write_metric_table(metrics, root, family='mace_causal_runtime', name='throughput')
    metadata['processes_after'] = gpu_processes()
    write_json(root/'technical/metadata.json', metadata)
    write_json(root/'technical/status.json', dict(state='complete', cases=len(rows)))
