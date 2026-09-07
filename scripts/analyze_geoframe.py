#!/usr/bin/env python3
"""Run a maintained geoframe workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.command_line import dispatch

COMMANDS = {
    'stability': 'src.temporal_vamp.commands.geoframe_stability',
    'variability': 'src.temporal_vamp.commands.geoframe_variability',
    'compare-variability': 'src.temporal_vamp.commands.geoframe_compare_variability',
    'compare-representations': 'src.temporal_vamp.commands.geoframe_compare_representations',
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
