"""Opt-in real-frame CUDA feasibility benchmark; never publishes birth labels.

Invocation: python -m src.postprocessing.gpu_benchmark --config CONFIG
This local hardware/numerical audit creates no W&B run or production cache.
"""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import torch
import freud
from src.experiment_runner.artifacts import file_hash, write_json
from src.postprocessing.io import geometry
from src.postprocessing.descriptors.ptm import full_labels
from src.postprocessing.descriptors.structure import structure, components, bond_correlations
from src.postprocessing.descriptors.gpu import prepare, finish, extension, reference_neighbors
from src.postprocessing.descriptors.gpu import structure as gpu_structure, finish_joint
from src.postprocessing.descriptors.gpu.ptm import labels as gpu_ptm, implementation_files as ptm_files

GPU_MODES = {'freud': ('freud', 'ovito'), 'cuda': ('cuda', 'ovito'),
             'cuda-voronoi': ('cuda', 'voronoi'), 'cuda-distance': ('cuda', 'distance'),
             'resident-voronoi': ('resident', 'voronoi'), 'joint-voronoi': ('joint', 'voronoi')}


def cpu_after_ptm(points, box, c, labels, *, diagnostics=False):
    query, neighbors, packed = reference_neighbors(points, box, c['neighbor_cutoff_A'], pack=diagnostics)
    pair = np.column_stack((neighbors.query_point_indices, neighbors.point_indices))
    crystal = np.isin(labels, [1, 2, 3])
    result = [components(crystal, pair, c['minimum_tracked_atoms'])]
    counts = [int(crystal.sum())]
    order = freud.order.Steinhardt(l=6).compute(query, neighbors=neighbors)
    corr = bond_correlations(order.particle_harmonics, order.ql, pair)
    solid = []
    for q, n in c['bond_order']['thresholds']:
        accepted = corr > q
        mask = np.bincount(pair[accepted, 0], minlength=len(points)) >= n
        result.append(components(mask, pair[accepted], c['minimum_tracked_atoms']))
        counts.append(int(mask.sum())); solid.append(mask)
    output = (np.stack(result), np.asarray(counts, np.int64), np.histogram(corr, bins=np.linspace(-1, 1, 41))[0])
    if diagnostics:
        packed_corr = np.full(packed.shape, -2, np.float32)
        packed_corr[packed >= 0] = corr
        return output, dict(neighbors=packed, q6=order.ql, correlations=packed_corr, solid=np.stack(solid))
    return output


def disagreement(ref, candidate, rd, cd):
    same_neighbors = np.array_equal(rd['neighbors'], cd['neighbors'])
    result = dict(component_atom_differences=np.count_nonzero(ref[0] != candidate[0], axis=1).tolist(),
        counts_equal=bool(np.array_equal(ref[1], candidate[1])),
        histogram_equal=bool(np.array_equal(ref[2], candidate[2])),
        histogram_l1=int(np.abs(ref[2]-candidate[2]).sum()),
        solid_atom_differences=np.count_nonzero(rd['solid'] != cd['solid'], axis=1).tolist(),
        neighbors_equal=same_neighbors, q6_max_abs=float(np.max(np.abs(rd['q6']-cd['q6']))))
    if same_neighbors:
        mask = rd['neighbors'] >= 0
        rc, cc = rd['correlations'][mask], cd['correlations'][mask]
        result['correlation_max_abs'] = float(np.max(np.abs(rc-cc)))
        result['bond_decision_differences'] = [int(np.count_nonzero((rc > q) != (cc > q))) for q in (.65, .7, .75)]
    result['scientific_arrays_exact'] = all(np.array_equal(a, b) for a, b in zip(ref, candidate))
    return result


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter(); value = call(); torch.cuda.synchronize()
    return value, time.perf_counter()-start


def ptm_disagreement(reference, candidate):
    """Do not hide errors in rare crystal atoms behind liquid true negatives."""
    r, c = reference > 0, candidate > 0
    missed, added = int((r & ~c).sum()), int((~r & c).sum())
    total = int(r.sum())
    return dict(confusion=np.bincount(reference.astype(int)*4+candidate, minlength=16).reshape(4,4).tolist(),
                atom_disagreement_fraction=float(np.mean(reference != candidate)),
                reference_crystalline=total, candidate_crystalline=int(c.sum()),
                missed_crystalline=missed, added_crystalline=added,
                missed_fraction=None if not total else missed/total,
                symmetric_difference_over_reference=None if not total else (missed+added)/total,
                crystalline_count_relative_error=None if not total else abs(int(c.sum())-total)/total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config); c = json.loads(path.read_text()); dest = Path(c['output'])
    dest.mkdir(parents=True, exist_ok=True)
    if len(c['threads']) != 1:
        raise ValueError('Use a fresh process per CPU thread count: OVITO reads its limit at import')
    os.environ['OVITO_THREAD_COUNT'] = str(c['threads'][0])
    backends = c.get('backends', ['cpu', 'freud', 'cuda'])
    if not backends or len(backends) != len(set(backends)) or set(backends)-({'cpu'} | GPU_MODES.keys()):
        raise ValueError(f'Invalid benchmark backends: {backends}')
    torch.set_num_threads(1)
    started = time.perf_counter(); extension(); torch.cuda.synchronize()
    if any(mode in GPU_MODES and GPU_MODES[mode][1] != 'ovito' for mode in backends):
        from src.postprocessing.descriptors.gpu.ptm import initialize
        initialize(torch.cuda.current_device())
    report = dict(config_sha256=file_hash(path), host=platform.node(), gpu=torch.cuda.get_device_name(),
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_threads=c['threads'][0],
        torch=torch.__version__, freud=freud.__version__, cuda=torch.version.cuda,
        extension_startup_seconds=time.perf_counter()-started,
        timing_policy='Synchronized wall time, warm filesystem/JIT, alternating order, allocation and transfers included. Full-frame totals separately include read, CPU PTM and NPZ compression. No production checksum scan, tracking or shared-filesystem writes included.',
        implementation={str(p):file_hash(p) for p in [Path(__file__), Path(__file__).parent/'descriptors/structure.py', Path(__file__).parent/'descriptors/ptm.py', Path(__file__).parent/'descriptors/gpu/__init__.py', Path(__file__).parent/'descriptors/gpu/kernels.cu', *ptm_files()]}, cases=[])
    report['timing_policy'] = 'Synchronized wall time, warm filesystem/JIT, alternating order, allocation and transfers included. Full-frame totals include read, the named PTM backend, all descriptors/components, and NPZ compression. No production checksum scan, tracking or shared-filesystem writes included.'
    report['backends'] = backends
    write_json(dest/'results.json', report)
    for case in c['cases']:
        torch.cuda.reset_peak_memory_stats()
        source, frame, config = case['source'], case['frame'], case['descriptor_config']
        (points, box), read_time = timed(lambda: geometry(source, frame))
        labels, ptm_time = timed(lambda: full_labels(points, box, config['ptm_rmsd_cutoff']))
        original = structure(points, box, config)
        reference, rd = cpu_after_ptm(points, box, config, labels, diagnostics=True)
        if not all(np.array_equal(a,b) for a,b in zip(original, reference)):
            raise RuntimeError('Instrumented CPU reference differs from production')
        row = dict(name=case['name'], source_id=source['id'], input_identity=source['input_identity'],
            frame=frame, atoms=len(points), read_seconds=read_time, initial_ptm_seconds=ptm_time,
            solid_counts=reference[1].tolist(), agreement={}, timings=[])
        ptm_methods = dict(cpu=lambda: full_labels(points, box, config['ptm_rmsd_cutoff']))
        for mode in backends:
            if mode == 'cpu':
                continue
            spatial, ordering = GPU_MODES[mode]
            if ordering != 'ovito':
                ptm_methods[mode] = lambda ordering=ordering: gpu_ptm(points, box, config['ptm_rmsd_cutoff'], neighbor_radius=2*config['neighbor_cutoff_A'], ordering=ordering)
            full, cd = gpu_structure(points, box, config, backend=spatial, ptm_backend=ordering, include_ptm=True, diagnostics=True)
            candidate, types = full[:3], full[3]
            row['agreement'][mode] = disagreement(reference, candidate, rd, cd)
            row['agreement'][mode]['ptm'] = ptm_disagreement(labels, types)
            print(json.dumps(dict(case=case['name'], mode=mode, agreement=row['agreement'][mode])), flush=True)
            del cd
        del rd
        report['cases'].append(row); write_json(dest/'results.json', report)
        for threads in c.get('threads', [1,4]):
            os.environ['OVITO_THREAD_COUNT'] = str(threads)
            methods = dict(cpu=lambda: cpu_after_ptm(points, box, config, labels),
                freud=lambda: finish(prepare(points, box, config['neighbor_cutoff_A'], 'freud'), labels, config),
                cuda=lambda: finish(prepare(points, box, config['neighbor_cutoff_A'], 'cuda'), labels, config))
            for mode in backends:
                if mode.startswith('cuda-'):
                    methods[mode] = methods['cuda']
                elif mode in ('resident-voronoi', 'joint-voronoi'):
                    # Isolated post-PTM work uses separate prepared neighbors;
                    # shared-search fusion is measured only in full_frame.
                    methods[mode] = lambda: finish_joint(prepare(points, box, config['neighbor_cutoff_A'], 'cuda'), torch.as_tensor(labels, device='cuda'), config)
            for repeat in range(c.get('repeats', 3)):
                order = backends if repeat%2 == 0 else list(reversed(backends))
                for mode in order:
                    if mode in ptm_methods:
                        _, elapsed = timed(ptm_methods[mode])
                        row['timings'].append(dict(threads=threads, repeat=repeat, stage='ptm', backend=mode, seconds=elapsed))
                for mode in order:
                    _, elapsed = timed(methods[mode])
                    row['timings'].append(dict(threads=threads, repeat=repeat, stage='after_ptm', backend=mode, seconds=elapsed))
                # Each full-frame method uses its explicitly named PTM backend.
                for mode in order:
                    def full_frame():
                        q, b = geometry(source, frame)
                        if mode == 'cpu':
                            result = structure(q, b, config, include_ptm=True)
                        else:
                            spatial, ordering = GPU_MODES[mode]
                            result = gpu_structure(q, b, config, backend=spatial, ptm_backend=ordering, include_ptm=True)
                        stream = io.BytesIO()
                        np.savez_compressed(stream, **{f'clusters_{d}':result[0][i:i+1] for i,d in enumerate(config['detectors'])},
                            solid_counts=result[1][None], bond_histogram=result[2][None], ptm=result[3][None], frames=[frame])
                        return len(stream.getbuffer())
                    size, elapsed = timed(full_frame)
                    row['timings'].append(dict(threads=threads, repeat=repeat, stage='full_frame', backend=mode, seconds=elapsed, compressed_bytes=size))
                write_json(dest/'results.json', report)
                print(json.dumps(dict(case=case['name'], threads=threads, repeat=repeat, recent=row['timings'][-6:])), flush=True)
        groups = sorted({(t['stage'], t['threads'], t['backend']) for t in row['timings']})
        row['median_seconds'] = {f'{stage}/{threads}/{mode}':statistics.median(t['seconds'] for t in row['timings'] if t['stage']==stage and t['threads']==threads and t['backend']==mode)
            for stage, threads, mode in groups}
        row['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(dest/'results.json', report)
    report['completed'] = True
    write_json(dest/'results.json', report)


if __name__ == '__main__':
    main()
