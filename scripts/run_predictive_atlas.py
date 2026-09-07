#!/usr/bin/env python3
"""Run a maintained predictive atlas workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.command_line import dispatch

COMMANDS = {
    'frozen': 'src.temporal_vamp.commands.atlas_frozen',
    'history': 'src.temporal_vamp.commands.atlas_history',
    'finetune': 'src.temporal_vamp.commands.atlas_finetune',
    'temporal-encoder': 'src.temporal_vamp.commands.atlas_temporal_encoder',
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
