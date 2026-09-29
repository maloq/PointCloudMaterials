"""Verified canonical float16 export for the exact-0.1-ps Al rerun protocol."""
import argparse
import json
from pathlib import Path
import shutil

from .memory_pair import convert_completed_pair
from src.project_runtime.transfer import write_json

PROTOCOL = 'al_dense_replay_v1'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--delete-source', action='store_true')
    args = parser.parse_args(argv)
    root = args.directory
    metadata = json.loads((root/'metadata.json').read_text())
    if metadata['protocol'] != PROTOCOL or metadata['state'] != 'dynamics_complete':
        raise ValueError('Dense Al conversion requires completed dense rerun dynamics')
    if (metadata['timestep_ps'], metadata['sample_interval_steps'], metadata['measurement_steps']) != (.002, 50, 300000):
        raise ValueError('Dense Al timeline differs from the declared 600-ps / 0.1-ps protocol')
    steps = tuple(range(0, 300001, 50))
    report = convert_completed_pair(root, metadata, steps, delete_source=args.delete_source)
    # The shared converter checks every rounded coordinate/velocity and all hashes.
    # Float32 is a disposable verification intermediate for this specific protocol.
    report.update(float32_reference_retained=False,
                  precision_reference='Verified transient float32 consumer coordinates; native restarts retained')
    shutil.rmtree(root/'trajectory_binary_float32')
    write_json(root/'paired_conversion.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
