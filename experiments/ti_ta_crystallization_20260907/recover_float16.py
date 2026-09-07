"""2026-09-07 quota recovery: compress Ta, finalize 3.0ns, resume Ta then Ti."""
import json
from pathlib import Path

from src.data_utils.conversion.position_storage import compress
from src.data_utils.temporal_lammps_binary import TemporalLAMMPSBinaryTrajectory
from src.simulation.campaigns.elemental import complete_trajectory, sequence, sha256

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/ids/vmorozov/simulations/ti_ta_crystallization_20260907')
BASELINE = Path('/home/ids/vmorozov/simulations/ta_initial_model_1m_24ps_npt_20260905')


def main():
    directories = [BASELINE / 'branches/Ta/model_1m', BASELINE / 'preflight']
    directories += [ROOT / 'Ta/branches' / name for name in ('2.7ns', '2.8ns', '2.9ns')]
    for directory in directories:
        legacy = directory / 'trajectory_binary_float32'
        if legacy.is_symlink():
            report = json.loads((directory / 'float16_conversion.json').read_text())
            binary = TemporalLAMMPSBinaryTrajectory.load(legacy)
            if binary.manifest['storage_dtype'] != 'float16' or binary.verify_checksums() != report['checksums']:
                raise RuntimeError(f'Changed previously compressed artifact: {legacy}')
            print(f'Verified already compressed: {legacy}', flush=True)
        else:
            compress(legacy, delete_source=True)
    config = json.loads((REPO / 'experiments/ti_ta_crystallization_20260907/ta.json').read_text())
    directory = ROOT / 'Ta/branches/3.0ns'
    metadata = json.loads((directory / 'metadata.json').read_text())
    if sha256(directory / 'final.restart.bin') != metadata['final_restart_sha256']:
        raise RuntimeError(f'Final restart changed since completed dynamics: {directory}')
    complete_trajectory(config, directory, metadata['steps'], metadata['origin'])
    (ROOT / 'float16_recovery_complete.json').write_text(json.dumps({
        'state': 'complete', 'converted_branches': [str(p) for p in directories] + [str(directory)],
        'next': 'Final Ta/3.60ns branch, then 100000-atom Ti source and six 240ps branches',
    }, indent=2) + '\n')
    sequence(REPO / 'experiments/ti_ta_crystallization_20260907/ta.json',
             REPO / 'experiments/ti_ta_crystallization_20260907/ti.json', resume_ta=True)


if __name__ == '__main__':
    main()
