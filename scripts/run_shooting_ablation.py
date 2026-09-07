#!/usr/bin/env python3
"""Run a maintained shooting ablation workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.command_line import dispatch

COMMANDS = {
    'distributional': 'src.temporal_vamp.commands.ablation_distributional',
    'dynamical': 'src.temporal_vamp.commands.ablation_dynamical',
    'encoder-finetune': 'src.temporal_vamp.commands.ablation_encoder_finetune',
    'geometry': 'src.temporal_vamp.commands.ablation_geometry',
    'multiscale': 'src.temporal_vamp.commands.ablation_multiscale',
    'short-horizon': 'src.temporal_vamp.commands.ablation_short_horizon',
    'spatial': 'src.temporal_vamp.commands.ablation_spatial',
    'temporal-pretraining': 'src.temporal_vamp.commands.ablation_temporal_pretraining',
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
