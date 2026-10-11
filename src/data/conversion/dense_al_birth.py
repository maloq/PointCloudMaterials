"""Verified continuous 0.10-ps Q/V conversion for birth-focused Al MD."""
import argparse
import json
from pathlib import Path

from src.data.conversion.streaming_pair import convert_streaming_pair
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.birth_sources import sha256
from src.simulation.campaigns.dense_al_birth_stop import PROTOCOL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--delete-source', action='store_true')
    args = parser.parse_args(argv); directory = args.directory
    metadata = json.loads((directory/'metadata.json').read_text())
    termination = json.loads((directory/'technical/stop-monitor/outcome.json').read_text())
    last = metadata['measurement_steps']
    if (metadata['protocol'], metadata['state'], metadata['timestep_ps'], metadata['sample_interval_steps']) != (PROTOCOL, 'dynamics_complete', .002, 50):
        raise ValueError('Requires completed continuous 0.10-ps birth-focused dynamics')
    if termination['state'] != 'dynamics_complete' or termination['protocol'] != PROTOCOL or last != termination['final_step'] or last % 50:
        raise ValueError('Timeline does not match birth-focused endpoint certificate')
    if last > metadata['stopping']['maximum_measurement_steps'] or metadata['observation_start_step'] != 0:
        raise ValueError('Continuous observation origin/cap differs')
    result = convert_streaming_pair(directory, metadata, range(0, last+1, 50), chunk_frames=32, delete_source=False)
    path = directory/'trajectory_binary_float16/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['provenance'].update(termination=termination,
        termination_certificate_sha256=sha256(directory/'technical/stop-monitor/outcome.json'),
        source_run_id=metadata['run_id'], actual_sampling_ps=.1,
        observation_start_step=0, initial_settling_included_ps=15.,
        replaces_unstarted_run=metadata['replaces_unstarted_run'])
    write_json(path, manifest)
    loaded = ShootingBinaryTrajectory.load(path.parent)
    if loaded.frame_count != last//50+1 or int(loaded.timesteps[0]) != 0 or int(loaded.timesteps[-1]) != last:
        raise ValueError('Consumer timeline differs from continuous observation contract')
    result.update(float16_manifest_sha256=sha256(path), termination=termination)
    write_json(directory/'paired_conversion.json', result)
    if args.delete_source:
        (directory/'trajectory.lammpstrj').unlink()
        result['source_deleted'] = True; write_json(directory/'paired_conversion.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
