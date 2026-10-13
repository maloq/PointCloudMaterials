"""Local GPU throughput and numerical audit of fixed Ni Q,V measurements.

Reuses recorded training-role Ni patches, without simulator launches, model
fitting, scientific-run modifications or W&B runs. Times exclude disk I/O.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.models.encoders.smooth_density import SmoothDensity
from src.research.ni_latent_sim.measurements import VelocityMeasurements


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261011)
    plan_path = args.parent/'technical/plan.json'
    plan = json.loads(plan_path.read_text()); config = plan['config']
    sources = {s['key']: s for s in plan['sources']}
    chunks, inputs = [], []
    for job in plan['jobs']:
        if sources[job['source_key']]['role'] != 'train':
            continue
        file = resolve_path(config['cache'])/'observations'/job['key']/'data.pt'
        receipt = json.loads((file.parent/'complete.json').read_text())
        checksum = sha(file)
        if checksum != receipt['sha256']:
            raise ValueError('Changed recorded Ni observations: '+str(file))
        data = torch.load(file, map_location='cpu', weights_only=False)
        chunks.append(data['state']/config['length_factor'])
        inputs.append(dict(path=str(file), sha256=checksum, rows=len(data['state']),
                           source=job['source_key'], frame=job['frame']))
        if sum(len(x) for x in chunks) >= 1024:
            break
    state = torch.cat(chunks)[:1024].cuda()
    measured = VelocityMeasurements().cuda().eval()
    q = state[:, 1:, :3]-state[:, :1, :3]
    # Exact support is a scientific requirement for smooth neighborhood targets.
    outer_radius = q.norm(dim=-1).max(-1).values
    if bool((outer_radius <= measured.cutoff_A).any()):
        raise ValueError('nearest80 may omit atoms inside the descriptor cutoff')
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output/'technical'; output.mkdir(exist_ok=True)
    module_path = Path(__file__).parents[1]/'research/ni_latent_sim/measurements.py'
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, dtype='float32',
        online_runs_created=0, scientific_training=False, independent_MD_continuations=0,
        input_plan=str(plan_path), input_plan_sha256=sha(plan_path), inputs=inputs,
        units='native Angstrom, Angstrom/ps; undo recorded material length factor',
        rows=len(state), atoms=state.shape[1], min_outer_radius_A=float(outer_radius.min()),
        module_sha256=sha(module_path), benchmark_sha256=sha(Path(__file__)),
        smooth_density_sha256=sha(Path(__file__).parents[1]/'models/encoders/smooth_density.py'),
        velocity_settings=dict(cutoff_A=5.5, shell_centers_A=[2.5, 3.5, 4.5],
                               shell_width_A=.45, affine_ridge_A2=1e-4),
        columns=measured.columns, timings=[])
    def save():
        (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')

    # Numerical audit on real patches, with double-precision reference arithmetic.
    small = state[:32].double(); reference = VelocityMeasurements().double().cuda()
    with torch.no_grad():
        exact = reference(small)
        fp32 = measured(state[:32]).double()
        relative = (fp32-exact).square().mean(0).sqrt()/exact.square().mean(0).sqrt().clamp_min(1e-10)
        a, _ = torch.linalg.qr(torch.randn(3, 3, device='cuda', dtype=torch.float64))
        rotated = torch.cat((small[..., :3] @ a, small[..., 3:] @ a), -1)
        rotation_error = float((reference(rotated)-exact).abs().max())
        shifted = small+torch.tensor([3., -2., 1., .4, -.7, .2], device='cuda')
        shift_error = float((reference(shifted)-exact).abs().max())
        relq = small[:, 1:, :3]-small[:, :1, :3]
        relv = small[:, 1:, 3:]-small[:, :1, 3:]
        eps = 1e-5  # ps; a ballistic directional derivative audit, not MD.
        finite = (reference.bond_power(relq+eps*relv)-reference.bond_power(relq-eps*relv))/(2*eps)
        jvp_error = float((finite-exact[:, 9:13]).norm()/exact[:, 9:13].norm())
    report['audit'] = dict(max_channel_fp32_relative_rms=float(relative.max()),
        rotation_max_absolute=rotation_error, translation_boost_max_absolute=shift_error,
        bond_rate_finite_difference_relative_l2=jvp_error, finite_difference_step_ps=eps)
    if not torch.isfinite(fp32).all() or max(rotation_error, shift_error) > 1e-8 or jvp_error > 1e-5 or float(relative.max()) > 1e-3:
        save(); raise ValueError('Fixed-measurement numerical audit failed: '+str(report['audit']))
    save(); print('audit', report['audit'], flush=True)

    density = SmoothDensity(radial_channels=4, max_ell=6).cuda().eval()
    def density74(x):
        q = (x[:, 1:, :3]-x[:, :1, :3])/measured.cutoff_A
        return density.power(density(q))

    for name, function in [('velocity17', measured), ('density74', density74)]:
        for mode in (['eager', 'compiled'] if args.compile else ['eager']):
            fn = torch.compile(function, fullgraph=True) if mode == 'compiled' else function
            for batch in [256, 1024]:
                x = state[:batch]
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                with torch.no_grad():
                    y = fn(x); torch.cuda.synchronize()
                    setup = time.monotonic()-started
                    reference_y = function(x)
                    if not torch.allclose(y, reference_y, atol=1e-5, rtol=1e-4):
                        raise ValueError('Compiled measurement differs from eager producer')
                    for _ in range(10):
                        fn(x)
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(5):
                        begin = time.monotonic()
                        for _ in range(50):
                            fn(x)
                        torch.cuda.synchronize()
                        samples.append((time.monotonic()-begin)/50)
                median = float(np.median(samples))
                row = dict(measurements=name, mode=mode, batch=batch,
                    setup_seconds=setup, repeat_seconds=samples, median_ms=median*1000,
                    patches_per_second=batch/median,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
                report['timings'].append(row); save(); print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
