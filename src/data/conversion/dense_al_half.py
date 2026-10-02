"""Verified bounded-memory conversion for variable-duration halfway-stopped Al."""
import argparse
import json
from pathlib import Path

from src.data.conversion.streaming_pair import convert_streaming_pair
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.birth_sources import sha256
from src.simulation.campaigns.dense_al_half_stop import PROTOCOLS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--delete-source', action='store_true')
    args = parser.parse_args(argv)
    metadata = json.loads((args.directory / 'metadata.json').read_text())
    interval = metadata['sample_interval_steps']
    if metadata['protocol'] != PROTOCOLS[interval] or metadata['state'] != 'dynamics_complete':
        raise ValueError('Variable Al conversion requires completed declared halfway dynamics')
    termination = json.loads((args.directory / 'technical/stop-monitor/outcome.json').read_text())
    last = metadata['measurement_steps']
    if metadata['timestep_ps'] != .002 or last != termination['final_step'] or last % interval:
        raise ValueError('Actual halfway timeline differs from its endpoint certificate')
    if last > metadata['stopping']['maximum_measurement_steps']:
        raise ValueError('Actual endpoint exceeds the declared peer cap')
    result = convert_streaming_pair(args.directory, metadata, range(0, last + 1, interval),
                                    chunk_frames=32, delete_source=False)
    # Add explicit termination/censoring to the ordinary shooting-binary provenance.
    path = args.directory / 'trajectory_binary_float16/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['provenance'].update(termination=termination,
        termination_certificate_sha256=sha256(args.directory / 'technical/stop-monitor/outcome.json'),
        source_run_id=metadata['run_id'], actual_sampling_ps=metadata['sampling_ps'])
    write_json(path, manifest)
    from src.data.trajectories.shooting import ShootingBinaryTrajectory
    loaded = ShootingBinaryTrajectory.load(path.parent)
    if loaded.frame_count != last // interval + 1 or int(loaded.timesteps[-1]) != last:
        raise ValueError('Final variable-duration consumer timeline differs')
    result.update(float16_manifest_sha256=sha256(path), termination=termination)
    write_json(args.directory / 'paired_conversion.json', result)
    if args.delete_source:
        (args.directory / 'trajectory.lammpstrj').unlink()
        result['source_deleted'] = True
        write_json(args.directory / 'paired_conversion.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
