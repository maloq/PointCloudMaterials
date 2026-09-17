"""Audit and derive observations from existing independent Al trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from src.data.trajectories.shooting import ShootingBinaryTrajectory, _array_sha256
from src.project_runtime.paths import load_json, portable_config
from .observations import frame_observation
from .targets import physical_packet


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare_source(record, config):
    torch.set_num_threads(1)
    s = record['source']
    root = Path(s['path'])
    if file_hash(root/'manifest.json') != s['manifest_sha256']:
        raise RuntimeError(f"Source {s['id']}: manifest changed")
    raw = ShootingBinaryTrajectory.load(root)
    for name, description in raw.manifest['arrays'].items():
        if _array_sha256(getattr(raw, name)) != description['sha256']:
            raise RuntimeError(f"Source {s['id']}: array checksum mismatch for {name}")
    if not np.all(raw.atom_types == 1) or raw.manifest['velocity_units'] != 'angstrom_per_ps':
        raise ValueError(f"Source {s['id']}: expected single-species Al with A/ps velocities")
    source_input = (root.parent/'source.in.lammps').read_text()
    log = (root.parent/'measurement.lammps.log').read_text()
    if 'INDEPENDENT_SOURCE_MEASUREMENT_COMPLETE' not in log:
        raise RuntimeError(f"Source {s['id']}: missing measurement completion evidence")
    temp = s['temperature_K']
    expected = f'fix ensemble all npt temp {temp:g} {temp:g} 0.3 iso 0 0 3'
    if expected not in source_input or 'timestep 0.003' not in source_input or 'units metal' not in source_input:
        raise ValueError(f"Source {s['id']}: unsupported transition kernel; inspect source.in.lammps")
    measurement = source_input.split('reset_timestep 0', 1)[1]
    if re.search(r'^\s*(velocity|read_restart|read_data)\s', measurement, re.M):
        raise ValueError(f"Source {s['id']}: intervention in measurement interval")
    times = raw.timesteps.astype(np.float64)*s['timestep_fs']/1000
    if not np.array_equal(times, np.arange(801)*.75):
        raise ValueError(f"Source {s['id']}: pilot requires complete 0--600 ps at 0.75 ps cadence")
    center = record['center_atom_ids'][0]
    anchors = config['anchor_frames']
    past_steps = int(config['maximum_history_ps']/.75)
    lag_steps = [int(lag/.75) for lag in config['future_lags_ps']]
    if any(lag != step*.75 for lag, step in zip(config['future_lags_ps'], lag_steps)):
        raise ValueError('Future offsets must exist exactly on the source timeline')
    observed_indices = list(range(min(anchors)-past_steps, max(anchors)+1))
    target_indices = sorted({a+d for a in anchors for d in [0, *lag_steps]})
    frames, targets = {}, {}
    for k in sorted(set(observed_indices+target_indices)):
        box = raw.box_high[k].astype(np.float64)-raw.box_low[k].astype(np.float64)
        f = frame_observation(raw.positions[k], raw.velocities[k], box, raw.atom_ids,
                              center, config['radius_A'], config['cutoff_A'])
        if k in observed_indices:
            frames[k] = f
        if k in target_indices:
            targets[k] = torch.from_numpy(physical_packet(f['positions'].numpy(), f['velocities'].numpy()))
    shard = dict(source_id=s['id'], lineage=s['lineage'], split=s['split'], temperature_K=temp,
        center_id=center, frames=frames, times_ps=times.tolist(),
        samples=[dict(anchor=a, present=targets[a], future=torch.stack([targets[a+d] for d in lag_steps]))
                 for a in anchors])
    destination = Path(config['cache'])/f"source-{s['id']:04d}.pt"
    torch.save(shard, destination.with_suffix('.building'))
    destination.with_suffix('.building').replace(destination)
    potentials = re.search(r'^pair_coeff \* \* (\S+) Al (\S+) Al$', source_input, re.M)
    if potentials is None:
        raise ValueError('Expected repository Al MEAM producer potential declaration')
    potential_hashes = {Path(p).name: file_hash(root.parent/p) for p in potentials.groups()}
    melt = (root.parent/'melt.in.lammps').read_text()
    melt_seed = int(re.search(r'velocity all create 1325 (\d+)', melt)[1])
    initial_path = re.search(r'^read_data (\S+)', melt, re.M)[1]
    metadata = dict(dataset_id=s['dataset'], trajectory_id=s['id'], root_lineage=s['lineage'],
        parent_trajectory_id=None, split=s['split'], center_id=center, material='Al', atomic_numbers=[13],
        temperature_K=temp, ensemble='NPT', thermostat='Nose-Hoover', thermostat_damping_ps=.3,
        barostat='isotropic Nose-Hoover', barostat_damping_ps=3., pressure_bar=0.,
        momentum_removal_interval_ps=.3, units='metal', timestep_ps=.003,
        melt_seed=melt_seed, melt_duration_ps=300., equilibration_duration_ps=15.,
        initial_fcc_sha256=file_hash(root.parent/initial_path),
        prepared_liquid_sha256=file_hash(root.parent/'prepared_liquid.lammps.data'),
        simulator_version=(root.parent/'log.lammps').read_text().splitlines()[0],
        potential_hashes=potential_hashes, source_input_sha256=file_hash(root.parent/'source.in.lammps'),
        source_manifest_sha256=s['manifest_sha256'], arrays=raw.manifest['arrays'],
        conversion=raw.manifest['source'], frame_times_ps=times.tolist(),
        coordinate_convention='wrapped relative to box_low; orthorhombic box_low/high arrays hashed above',
        velocity_reset_before_equilibration=True, measurement_interventions=[],
        measurement_complete_marker=True, stopping_rule='fixed 600 ps', missing_segments=[],
        precision='full-box float16 positions and velocities; no matched full precision audit yet',
        data_access='inherited exploratory train/validation/test; test sources previously examined',
        shard_sha256=file_hash(destination), max_frame_atoms=max(len(f['ids']) for f in frames.values()))
    return metadata


def prepare(config):
    cache = Path(config['cache'])
    cache.mkdir(parents=True, exist_ok=True)
    if (cache/'complete.json').exists():
        raise FileExistsError(f'Release already exists: {cache}')
    inherited = load_json(config['source_plan'])
    records = inherited['records']
    lineages = [r['source']['lineage'] for r in records]
    if len(set(lineages)) != len(lineages):
        raise ValueError('Pilot expects distinct melt lineages, not branches or duplicates')
    write_json(cache/'preparation.json', dict(config=portable_config(config), source_plan_sha256=file_hash(config['source_plan'])))
    metadata = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=config['prepare_workers']) as pool:
        futures = {pool.submit(prepare_source, record, config): record['source']['id'] for record in records}
        for f in as_completed(futures):
            metadata.append(f.result())
            write_json(cache/'progress.json', dict(sources=len(metadata), total=len(records), seconds=time.monotonic()-start))
            print(f'Prepared {len(metadata)}/{len(records)} source={futures[f]} seconds={time.monotonic()-start:.1f}', flush=True)
    metadata.sort(key=lambda x: x['trajectory_id'])
    for key in ('prepared_liquid_sha256', 'melt_seed'):
        if len({m[key] for m in metadata}) != len(metadata):
            raise ValueError(f'Duplicate independent initial state detected: {key}')
    for name in ('positions', 'velocities'):
        if len({m['arrays'][name]['sha256'] for m in metadata}) != len(metadata):
            raise ValueError(f'Duplicate source array detected: {name}')
    write_json(cache/'dataset_release.json', dict(schema_version=1, protocol=config['protocol'], sources=metadata,
        source_plan_sha256=file_hash(config['source_plan']), config=portable_config(config),
        label_policy='no PTM, crystallization, onset, topology, or basin labels loaded',
        independent_claim='distinct 300 ps melts from shared FCC preparation; not proof of statistical independence'))
    write_json(cache/'splits.json', {split: [m['trajectory_id'] for m in metadata if m['split'] == split]
                                   for split in ('train', 'val', 'test')})
    pd.DataFrame([{k: m[k] for k in ('trajectory_id', 'root_lineage', 'parent_trajectory_id', 'split', 'melt_seed',
                  'initial_fcc_sha256', 'prepared_liquid_sha256')} for m in metadata]).to_parquet(cache/'lineages.parquet', index=False)
    write_json(cache/'precision_audit.json', dict(state='limited_existing_float16',
        measured_quantization_errors=None, matched_full_precision_comparison='not performed: unavailable in this pilot',
        position_dtype='float16', velocity_dtype='float16', box_dtype='float32',
        local_calculation_dtype='float64', cache_dtype='float32',
        limitation='history gains may include quantization-noise averaging; casting does not recover precision'))
    (cache/'DATA_CARD.md').write_text('# Predictive memory exploratory release\n\n'
        f'{len(metadata)} distinct melt sources; first center from each inherited sorted four-center sample and three anchors per source. '
        'Whole-source splits inherited from the previous exploratory study. All manifest array content checksums verified. '
        'Distinct prepared melts and trajectory checksums verified; common initial FCC configurations are recorded. '
        'NPT Al MEAM, 0.75 ps stored cadence, 600 ps measurement, periodic momentum removal. '
        'Targets use continuous 5--7 A geometry and motion; no crystallization or topology labels are loaded.\n\n'
        'Precision audit is incomplete: existing full-box float16 data cannot establish unquantized memory gains. '
        'This release supports a pipeline pilot, not confirmatory physical-memory claims.\n')
    write_json(cache/'complete.json', dict(state='complete', sources=len(metadata), samples=3*len(metadata),
        release_sha256=file_hash(cache/'dataset_release.json'), seconds=time.monotonic()-start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    from src.experiment_runner.tracking import tracked_run
    import sys
    config = load_json(args.config)
    with tracked_run(Path(config['output'])/'technical'/'preparation', kind='analysis',
                     configs=[Path(args.config)], command=sys.argv, question='predictive_memory'):
        prepare(config)


if __name__ == '__main__':
    main()
