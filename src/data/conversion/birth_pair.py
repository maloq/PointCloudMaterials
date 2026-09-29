"""Paired precision conversion of continuously observed Al birth sources."""
import argparse
import json
from pathlib import Path

from .memory_pair import convert_completed_pair

PROTOCOL = 'al_spontaneous_birth_sources_v1'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--delete-source', action='store_true')
    args = parser.parse_args(argv)
    metadata = json.loads((args.directory/'metadata.json').read_text())
    if metadata['protocol'] != PROTOCOL or metadata['state'] != 'dynamics_complete':
        raise ValueError('Birth conversion requires complete birth-source dynamics')
    if metadata['completed_steps'] % metadata['sample_interval_steps']:
        raise ValueError('Completed hold is not aligned to the declared observation grid')
    steps = tuple(range(0, metadata['completed_steps']+1, metadata['sample_interval_steps']))
    print(json.dumps(convert_completed_pair(args.directory, metadata, steps,
                                          delete_source=args.delete_source), indent=2))


if __name__ == '__main__':
    main()
