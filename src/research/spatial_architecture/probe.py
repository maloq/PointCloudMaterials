"""Local real-data numerical checks and matched GPU timing; no online runs."""
import gc
import math
import time

import numpy as np
import torch

from src.research.structural_response.data import root, spatial_data, write_json, digest
from src.research.structural_response.model import spatial_backward
from .model import Spatial, teacher
from src.models.encoders.esen_runtime import configure, direct_backward
from . import parallel


def derivative_probe(geometry, q):
    """Local FP32 AD diagnostic on two real patches before block compilation.

    This measures encoder derivatives only, not a full simulator-response update.
    The random directions are numerical probes, not observed atomic velocities.
    """
    q = q[:2].detach().clone().requires_grad_(True)
    with torch.random.fork_rng(devices=[q.device.index]):
        torch.manual_seed(20261008)
        direction = torch.randn_like(q)
        direction = direction - direction[:, :1]
    torch.cuda.synchronize()
    start = time.monotonic()
    value, jvp = torch.autograd.functional.jvp(geometry, q, direction,
                                              create_graph=True, strict=True)
    torch.cuda.synchronize()
    jvp_seconds = time.monotonic() - start
    with torch.no_grad():
        finite = (geometry(q + .001 * direction) - geometry(q - .001 * direction)) / .002
    error = float((jvp.detach() - finite).norm() / jvp.detach().norm().clamp_min(1e-12))
    if not np.isfinite(error) or error > .03:
        raise FloatingPointError(f'TensorNet input JVP disagrees with finite differences: {error}')
    # Hessian-vector product of a scalar projection; then differentiate its
    # squared norm with respect to parameters (mixed third derivatives).
    scalar = value.square().mean()
    gradient = torch.autograd.grad(scalar, q, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * direction).sum(), q, create_graph=True)[0]
    torch.cuda.synchronize()
    start = time.monotonic()
    loss = jvp.square().mean() + hvp.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - start
    gradients = [p.grad for p in geometry.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('TensorNet nested derivative parameter gradient is missing or nonfinite')
    report = dict(rows=len(q), precision='float32', jvp_relative_fd_error=error,
        jvp_seconds=jvp_seconds, nested_parameter_backward_seconds=backward_seconds,
        jvp_norm=float(jvp.detach().norm()), hvp_norm=float(hvp.detach().norm()),
        all_parameter_gradients_finite=True,
        scope='eager encoder only, fixed local atom list, random centered directions; not full motion-training timing')
    geometry.zero_grad(set_to_none=True)
    return report


def run(c):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu_devices = parallel.devices(c)
    data, columns, weights = spatial_data(c)
    weights = torch.from_numpy(weights).cuda()
    batch = c['spatial_training']['batch_size']
    micro = c['spatial_training']['microbatch']
    ids = np.random.default_rng(np.random.SeedSequence([c['seed'], 0])).permutation(len(data))[:14*batch]
    q = torch.from_numpy(data.batch(ids[:8])['positions']).cuda()
    reports = []
    for architecture in c['architectures']:
        torch.manual_seed(c['seed'])
        model = Spatial(c, architecture).cuda()
        ema = teacher(c, architecture, model.geometry)
        variant = c.get('execution_variants', {}).get(architecture, 'eager')
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            reference = model.geometry(q)
            matrix, _ = torch.linalg.qr(torch.randn(3, 3, device='cuda'))
            matrix[:, 0] *= torch.linalg.det(matrix)
            # Transform geometry in FP32 for both execution paths. A matmul
            # inside BF16 autocast would perturb positions as well as rotate
            # them, confounding symmetry with sensitivity to atomic motion.
            rotated_q = q @ matrix
            rotated = model.geometry(rotated_q)
            translated = model.geometry(q + q.new_tensor([1., 2., 3.]))
            perm = torch.cat((torch.zeros(1, device='cuda', dtype=torch.long), torch.randperm(79, device='cuda')+1))
            permuted = model.geometry(q[:, perm])
            repeated = model.geometry(q)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            amp = model.geometry(q)
        audit = dict(rotation_max_abs=float((reference-rotated).abs().max()),
            translation_max_abs=float((reference-translated).abs().max()),
            permutation_max_abs=float((reference-permuted).abs().max()),
            repeat_max_abs=float((reference-repeated).abs().max()),
            amp_relative_rms=float((reference-amp).square().mean().sqrt()/reference.square().mean().sqrt()))
        if max(audit[k] for k in ('rotation_max_abs', 'translation_max_abs', 'permutation_max_abs')) > 5e-4:
            raise FloatingPointError(f'{architecture} symmetry check failed: {audit}')
        if audit['amp_relative_rms'] > .03 or audit['repeat_max_abs'] > 1e-5:
            raise FloatingPointError(f'{architecture} precision/replay check failed: {audit}')
        if architecture == 'tensornet':
            audit['derivatives'] = derivative_probe(model.geometry, q)
        configure(model.geometry, architecture, variant)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            accelerated = model.geometry(q)
            accelerated_rotation = model.geometry(rotated_q)
        audit['execution_relative_rms'] = float((accelerated-amp).square().mean().sqrt()/amp.square().mean().sqrt())
        audit['execution_rotation_relative_rms'] = float((accelerated-accelerated_rotation).square().mean().sqrt()/accelerated.square().mean().sqrt())
        if max(audit['execution_relative_rms'], audit['execution_rotation_relative_rms']) > .02:
            raise FloatingPointError(f'{architecture} accelerated precision/symmetry failed: {audit}')
        parallel_step = parallel.create(c, architecture, model)
        if parallel_step is not None:
            b = data.batch(ids[:batch])
            audit['parallel_global_batch'] = parallel_step.audit(
                torch.from_numpy(b['positions']).cuda(), torch.from_numpy(b['target'][:, columns]).cuda(),
                weights, c['seed']+1)
            print('two-GPU objective/gradient audit', audit['parallel_global_batch'], flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=c['spatial_training']['weight_decay'])
        times = []
        for device in gpu_devices:
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for step in range(14):
            parallel.synchronize(c)
            begin = time.monotonic()
            b = data.batch(ids[step*batch:(step+1)*batch])
            x = torch.from_numpy(b['positions']).cuda()
            y = torch.from_numpy(b['target'][:, columns]).cuda()
            opt.zero_grad(set_to_none=True)
            if parallel_step is not None:
                loss, stats = parallel_step.backward(x, y, weights, c['seed']+step+1)
            elif variant.endswith('direct'):
                loss, stats = direct_backward(model, x, y, weights, c['seed']+step+1)
            else:
                loss, stats = spatial_backward(model, ema, x, y, weights, 'vicreg_descriptors', c['seed']+step+1, micro)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.geometry.parameters(), strict=True):
                    a.lerp_(b, c['spatial_training']['ema'])
            parallel.synchronize(c)
            times.append(time.monotonic()-begin)
            print(architecture, 'timing update', step+1, times[-1], flush=True)
        median = float(np.median(times[4:]))
        reports.append(dict(architecture=architecture, execution_variant=variant, **audit,
            encoder_parameters=sum(p.numel() for p in model.geometry.parameters()),
            total_parameters=sum(p.numel() for p in model.parameters()),
            update_seconds=times, warmed_median_seconds=median,
            estimated_epoch_seconds=median*math.ceil(len(data)/batch),
            estimated_12epoch_training_seconds=median*c['spatial_training']['updates'],
            peak_memory_GB=torch.cuda.max_memory_allocated()/2**30,
            per_gpu_peak_memory_GiB=[torch.cuda.max_memory_allocated(d)/2**30 for d in gpu_devices],
            final_probe_loss=loss, gradient_norm=float(norm), **stats))
        if parallel_step is not None:
            parallel_step.close()
            del parallel_step
        del model, ema, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()
    record = dict(passed=True, config_identity=digest(c), gpu=torch.cuda.get_device_name(),
        gpu_count=len(gpu_devices), gpu_names=[torch.cuda.get_device_name(d) for d in gpu_devices],
        per_device_batch=math.ceil(batch/len(gpu_devices)), global_vicreg_batch=batch,
        torch=torch.__version__, batch_size=batch, microbatch=micro, warmup_updates=4,
        measured_updates=10, identical_rows=True, online_runs_created=0, architectures=reports,
        timing_scope='Synchronized data batch/transfer + two-view full-batch VICReg/descriptor backward + clip + AdamW + EMA. Validation, checkpoints, tracking, initialization and queue time excluded.')
    write_json(root(c)/'technical/probe.json', record)
    data.close()
    return record
