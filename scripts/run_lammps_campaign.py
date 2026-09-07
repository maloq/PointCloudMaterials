#!/usr/bin/env python3
"""Run a maintained lammps campaign workflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.command_line import dispatch

COMMANDS = {
    'elemental': 'src.simulation.campaigns.elemental',
    'homogeneous': 'src.simulation.campaigns.homogeneous',
    'independent-meam-source': 'src.simulation.campaigns.independent_meam_source',
    'meam-nested-shooting': 'src.simulation.campaigns.meam_nested_shooting',
    'meam-shooting': 'src.simulation.campaigns.meam_shooting',
    'meam-shooting-followup': 'src.simulation.campaigns.meam_shooting_followup',
    'nested-fixed-horizon-compatibility': 'src.simulation.campaigns.nested_fixed_horizon_compatibility',
    'predictive-dynamics-15ps': 'src.simulation.campaigns.predictive_dynamics_15ps',
    'predictive-dynamics': 'src.simulation.campaigns.predictive_dynamics',
    'seeded-crystallization': 'src.simulation.campaigns.seeded_crystallization',
    'unseeded-meam-crystallization': 'src.simulation.campaigns.unseeded_meam_crystallization',
    'unseeded-meam-ensemble': 'src.simulation.campaigns.unseeded_meam_ensemble',
    'unseeded-meam-source-followup': 'src.simulation.campaigns.unseeded_meam_source_followup',
    'unseeded-meam-temperature': 'src.simulation.campaigns.unseeded_meam_temperature',
}


def main(argv=None):
    return dispatch(COMMANDS, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
