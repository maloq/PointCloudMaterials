"""Verify paired float32/float16 exports from the memory-source producer."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data.temporal import TemporalLAMMPSDumpDataset
from src.data.trajectories.shooting import (ShootingBinaryTrajectory,
    convert_shooting_trajectory, compose_shooting_binary_trajectories)
from src.project_runtime.transfer import write_json
from src.experiment_runner.registry import sha256


PROTOCOL = 'predictive_memory_precision_sources_v1'


def paired_errors(reference, rounded):
    """Check every rounded value and unchanged identity/timeline/box array."""
    for name in ('atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high'):
        if not np.array_equal(getattr(reference, name), getattr(rounded, name)):
            raise ValueError(f'Paired precision export changed {name}')
    result = {}
    for field in ('positions', 'velocities'):
        maximum, square_sum, count = 0., 0., 0
        original, stored = getattr(reference, field), getattr(rounded, field)
        if original.dtype != np.float32 or stored.dtype != np.float16 or original.shape != stored.shape:
            raise ValueError(f'Wrong paired {field} shape/dtype')
        for i, (a, b) in enumerate(zip(original, stored, strict=True)):
            if not np.isfinite(b).all() or not np.array_equal(a.astype(np.float16), b):
                raise ValueError(f'Incorrect/nonfinite float16 rounding: {field}, frame {i}')
            error = b.astype(np.float64)-a
            if field == 'positions':
                lengths = reference.box_high[i]-reference.box_low[i]
                error -= lengths*np.rint(error/lengths)
            maximum = max(maximum, float(np.abs(error).max()))
            square_sum += float(np.square(error).sum())
            count += error.size
        result[field] = dict(max_abs=maximum, rms=(square_sum/count)**.5, coordinate_count=count,
            units='angstrom' if field == 'positions' else 'angstrom_per_ps',
            periodic_minimum_image=field == 'positions')
    return result


def convert(directory, *, delete_source=False):
    root = Path(directory)
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['protocol'] != PROTOCOL or metadata['state'] != 'dynamics_complete':
        raise ValueError(f'Paired conversion requires completed memory-source dynamics: {root}')
    source = root/'trajectory.lammpstrj'
    if sha256(source) != metadata['source_sha256']:
        raise ValueError(f'Memory-source dump differs from completed dynamics: {source}')
    steps = tuple(range(0, metadata['measurement_steps']+1, metadata['sample_interval_steps']))
    scan = TemporalLAMMPSDumpDataset.scan_dump_file(source)
    if scan.num_atoms != metadata['atom_count'] or tuple(scan.timesteps) != steps or tuple(scan.atom_columns) != (
            'id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz'):
        raise ValueError(f'Dump atom/timeline/column contract mismatch: {source}')
    provenance = dict(protocol=PROTOCOL, root_lineage=metadata['root_lineage'], split=metadata['split'],
                      source_sha256=metadata['source_sha256'], timestep_ps=metadata['timestep_ps'])
    full_path, half_path = root/'trajectory_binary_float32', root/'trajectory_binary_float16'
    if full_path.exists():
        full = ShootingBinaryTrajectory.load(full_path)
        if full.manifest['provenance'] != provenance:
            raise ValueError(f'Existing float32 export has different provenance: {full_path}')
    else:
        full = convert_shooting_trajectory(source, full_path, timesteps=steps,
            atom_count=metadata['atom_count'], storage_dtype='float32', provenance=provenance)
    full.verify_checksums()
    if half_path.exists():
        half = ShootingBinaryTrajectory.load(half_path)
    else:
        half = compose_shooting_binary_trajectories([full], half_path, timesteps=steps,
            storage_dtype='float16', provenance=provenance)
    half.verify_checksums()
    errors = paired_errors(full, half)
    report = dict(state='complete', protocol=PROTOCOL, source_sha256=metadata['source_sha256'],
        source_deleted=False, frame_count=full.frame_count, atom_count=metadata['atom_count'],
        float32_manifest_sha256=sha256(full_path/'manifest.json'),
        float16_manifest_sha256=sha256(half_path/'manifest.json'), quantization=errors,
        precision_reference='float32 consumer coordinates; native restart precision is unchanged')
    write_json(root/'paired_conversion.json', report)
    if delete_source:
        # Both complete, checksum-verified precision variants and native restarts remain.
        source.unlink()
        report['source_deleted'] = True
        write_json(root/'paired_conversion.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--delete-source', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.directory, delete_source=args.delete_source), indent=2))


if __name__ == '__main__':
    main()
