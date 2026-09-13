"""Check Lee-2003 Al liquid/crystal PTM separation with the elemental producer."""
import json
from pathlib import Path

import numpy as np

from src.simulation.campaigns.elemental import (
    bind_allocation, execute, melt_input, sha256, structure, write_json,
)

EXP = Path(__file__).resolve().parents[3] / 'docs/simulations/al_crystallization/technical'


def main():
    config = json.loads((EXP / 'al.json').read_text())
    root = Path(config['output_root']) / 'preflight'
    root.mkdir(exist_ok=True)
    for item in config['potential_files']:
        if sha256(item['path']) != item['sha256']:
            raise RuntimeError(f'Changed shooting potential: {item}')
    bind_allocation(config)
    config.update(repetitions_xyz=[10, 10, 10], atom_count=4000)
    reports = {}
    for name, temperature, steps in [('hot_crystal', config['temperature_K'], 10000),
                                      ('liquid', config['melt_temperature_K'], 50000)]:
        case = {**config, 'melt_temperature_K': temperature, 'melt_steps': steps}
        directory = root / name
        execute(case, directory, melt_input(case), 'MELT_COMPLETE')
        result = structure(directory / 'liquid.lammpstrj', config['ptm_rmsd_cutoff'])
        msd = np.loadtxt(directory / 'msd.dat')
        result['msd_growth_last_half_A2'] = float(msd[-1, 1] - msd[len(msd) // 2, 1])
        reports[name] = result
        write_json(root / 'observations.json', reports)
    if reports['hot_crystal']['crystal_fraction'] < config['complete_crystal_fraction']:
        raise RuntimeError(f'Hot Al crystal falls below stopping threshold: {reports}')
    if (reports['liquid']['crystal_fraction'] > config['max_liquid_crystal_fraction']
            or reports['liquid']['msd_growth_last_half_A2'] < config['minimum_liquid_msd_growth_A2']):
        raise RuntimeError(f'Al melt has not passed liquid checks: {reports}')
    write_json(root / 'validation.json', {'state': 'passed', 'potential': config['potential_files'],
                                         'ptm_rmsd_cutoff': config['ptm_rmsd_cutoff'], **reports})
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    main()
