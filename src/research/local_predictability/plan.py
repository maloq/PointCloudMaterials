"""Validate and materialize a planning queue; never launch training or simulations."""
import argparse
import hashlib
import json
from pathlib import Path


def build_queue(config):
    if config['status'] != 'planning_only_requires_new_adapters':
        raise ValueError('This command only supports planning, not execution')
    if config['wall_hours'] != 16 or set(config['workers']) != {'local_h100', 'remote_h200'}:
        raise ValueError('Require one shared 16-hour window on the two declared workers')
    previous = 0
    for phase in config['phases']:
        if phase['start_hour'] != previous or phase['end_hour'] <= previous:
            raise ValueError(f"Overlapping or missing phase interval: {phase['id']}")
        previous = phase['end_hour']
        for worker in config['workers']:
            if not phase[worker]:
                raise ValueError(f'Missing assignment for {worker}')
    if previous != config['wall_hours'] or config['phases'][-1]['id'] != 'finalize' or \
            config['phases'][-1]['start_hour'] > config['wall_hours'] - 1:
        raise ValueError('Reserve the final hour for evaluation/artifact preservation')
    sampling, assay = config['sampling'], config['assay']
    dt = sampling['cadence_ps']
    for value in [*sampling['horizons_ps'], sampling['maximum_history_ps'], sampling['population_origin_stride_ps']]:
        if abs(value / dt - round(value / dt)) > 1e-9:
            raise ValueError(f'Physical time {value} is not on the source timeline')
    pad = (max(assay['persistence_frames'], *assay['sensitivity_persistence_frames']) - 1) * dt
    first = max(sampling['maximum_history_ps'], (assay['negative_history_frames'] - 1) * dt)
    last = sampling['end_ps'] - max(sampling['horizons_ps']) - pad
    stride = sampling['population_origin_stride_ps']
    origins = [first + i * stride for i in range(int((last - first) // stride) + 1)]
    if len(origins) < sampling['native_anchors_per_center']:
        raise ValueError('Insufficient fully confirmed origins for native sampling')
    native = config['native']
    variants = {v['name']: v for v in native['variants']}
    if len(native['variants']) != 3 or set(variants) != {'snapshot', 'history12', 'repeat12'}:
        raise ValueError('A core group requires snapshot, real history and separately trained repeated frames')
    if variants['snapshot']['history_ps'] != 0 or variants['snapshot']['repeat_anchor']:
        raise ValueError('Snapshot must use only the current frame')
    if variants['history12']['history_ps'] != 12 or \
            variants['history12']['history_ps'] != variants['repeat12']['history_ps'] or \
            variants['history12']['repeat_anchor'] or not variants['repeat12']['repeat_anchor']:
        raise ValueError('Real/repeated histories must match duration and differ only in observations')
    if native['seeds'] != [20260919] or config['descriptors']['mlp_hazard']['seeds'] != native['seeds']:
        raise ValueError('This fast queue uses exactly one shared training seed: 20260919')
    for gate in native['core_requires'] + native['history_expansion_requires']:
        if gate not in config['gates']:
            raise ValueError(f'Undefined gate: {gate}')

    jobs = []
    for observation in config['descriptors']['observations']:
        jobs.append(dict(id=f'linear-{observation}', worker='local_h100', kind='descriptor_hazard',
                         model='linear', observation=observation))
        for seed in config['descriptors']['mlp_hazard']['seeds']:
            jobs.append(dict(id=f'mlp-{observation}-{seed}', worker='local_h100', kind='descriptor_hazard',
                             model='mlp', observation=observation, seed=seed))
    for observation in config['descriptors']['physical_ridge_inputs']:
        jobs.append(dict(id=f'ridge-{observation}', worker='local_h100', kind='physical_ridge', observation=observation))
    for objective, worker in native['workers_by_objective'].items():
        for seed in native['seeds']:
            parent = f'{objective}-{seed}-parent'
            jobs.append(dict(id=parent, worker=worker, kind='native_parent', objective=objective, seed=seed,
                             gates=native['core_requires'], updates='K; frozen after profiling'))
            for variant in native['variants']:
                jobs.append(dict(id=f"{objective}-{seed}-{variant['name']}", worker=worker,
                                 kind='native_continuation', objective=objective, seed=seed, **variant,
                                 depends_on=[parent], updates='K additional; snapshot receives the same continuation'))
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate queue job identity')
    return dict(protocol=config['protocol'], execution_status=config['status'], wall_hours=16,
                primary_common_origins_ps=origins, confirmation_padding_ps=pad,
                population_candidate_windows=sum(sampling['source_counts'].values()) *
                    sampling['descriptor_centers_per_source'] * len(origins),
                native_candidate_windows=sum(sampling['source_counts'].values()) *
                    sampling['native_centers_per_source'] * sampling['native_anchors_per_center'],
                jobs=jobs, optional_enabled=config['optional_extensions_enabled'],
                optional=config['optional_priority'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Planning configuration has repository-relative identities and no machine paths.
    raw = args.config.read_bytes()
    queue = build_queue(json.loads(raw))
    queue['plan_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(queue, stream, indent=2)
        stream.write('\n')
    print(f"Planned {len(queue['jobs'])} model jobs; no experiments launched. Output: {args.output}")


if __name__ == '__main__':
    main()
