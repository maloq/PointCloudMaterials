"""Diagnose same-rank CSLD restart divergence without changing production data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from src.simulation.campaigns.predictive_dynamics_15ps import _render_uninterrupted_24ps_input
from src.data_utils.synthetic.atomistic.lammps_shooting import _lammps_command, _lammps_environment


def read_probe(path):
    frames = {}
    with path.open() as stream:
        while True:
            marker = stream.readline()
            if not marker:
                break
            assert marker == 'ITEM: TIMESTEP\n', path
            step = int(stream.readline())
            assert stream.readline() == 'ITEM: NUMBER OF ATOMS\n'
            count = int(stream.readline())
            assert count == 70304
            assert stream.readline() == 'ITEM: BOX BOUNDS pp pp pp\n'
            bounds = np.array([[float(v) for v in stream.readline().split()] for _ in range(3)])
            assert stream.readline() == 'ITEM: ATOMS id proc x y z vx vy vz\n'
            table = np.loadtxt(stream, max_rows=count)
            assert table.shape == (70304, 8)
            frames[step] = (table, bounds)
    assert set(frames) == {5000, 5001, 5002}, (path, list(frames))
    return frames


def compare(reference, observed):
    result = {}
    for step in reference:
        a, box = reference[step]
        b, other_box = observed[step]
        ordered_a = a[np.argsort(a[:, 0])]
        ordered_b = b[np.argsort(b[:, 0])]
        np.testing.assert_array_equal(ordered_a[:, 0], ordered_b[:, 0])
        np.testing.assert_array_equal(box, other_box)
        delta = ordered_b[:, 2:5] - ordered_a[:, 2:5]
        lengths = box[:, 1] - box[:, 0]
        minimum_image = delta - lengths * np.rint(delta / lengths)
        result[step] = dict(
            same_local_atom_order=bool(np.array_equal(a[:, :2], b[:, :2])),
            changed_owner_count=int(np.count_nonzero(ordered_a[:, 1] != ordered_b[:, 1])),
            max_position_difference_A=float(np.abs(delta).max()),
            max_minimum_image_difference_A=float(np.abs(minimum_image).max()),
            max_velocity_difference_A_per_ps=float(np.abs(ordered_b[:, 5:] - ordered_a[:, 5:]).max()),
            positions_float32_equal=bool(np.array_equal(ordered_a[:, 2:5].astype(np.float32), ordered_b[:, 2:5].astype(np.float32))),
            velocities_float32_equal=bool(np.array_equal(ordered_a[:, 5:].astype(np.float32), ordered_b[:, 5:].astype(np.float32))),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = dict(state='running', started_at=datetime.now(timezone.utc).isoformat())

    def save():
        temporary = root / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(root / 'status.json')

    save()
    try:
        smoke = args.smoke_root.resolve()
        manifest = json.loads((smoke / 'manifest.json').read_text())
        branch = manifest['branches'][0]
        original_restart = smoke / branch['branch_dir'] / 'final.restart.bin'
        initial = _render_uninterrupted_24ps_input(branch)
        initial = initial.replace('../../parents/', str(smoke / 'parents') + '/')
        initial = initial.replace('../../potential/', str(smoke / 'potential') + '/')
        probe = '''dump probe all custom 1 boundary.lammpstrj id proc x y z vx vy vz
 dump_modify probe sort off format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"
run 2
'''
        initial = initial.replace('run 3000', 'undump trajectory\n' + probe)
        variants = {'uninterrupted': initial}
        header = f'''log lammps.log
units metal
dimension 3
boundary p p p
atom_style atomic
read_restart RESTART_PATH
pair_style meam
pair_coeff * * {smoke}/potential/Lee2003_Al.library.meam Al {smoke}/potential/Lee2003_Al.meam Al
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.003
fix remove_drift all momentum 100 linear 1 1 1
fix integrate all nve
fix thermostat all temp/csld {branch['temperature_K']} {branch['temperature_K']} 0.3 {branch['thermostat_seed']}
thermo 1
thermo_style custom step temp press vol pe ke etotal
thermo_modify format float %.16g flush yes
'''
        variants['restart_existing'] = header.replace('RESTART_PATH', str(original_restart)) + probe
        rerun_restart = root / 'uninterrupted/midpoint_15ps.restart.bin'
        variants['restart_same_job'] = header.replace('RESTART_PATH', str(rerun_restart)) + probe
        variants['restart_no_sort'] = header.replace('RESTART_PATH', str(rerun_restart)) + 'atom_modify sort 0 0.0\n' + probe
        for name, script in variants.items():
            status['stage'] = name
            save()
            directory = root / name
            directory.mkdir()
            (directory / 'in.lammps').write_text(script)
            with (directory / 'stdout.log').open('wb') as log:
                subprocess.run(_lammps_command(mpi_ranks=24, launcher='srun_pmi2'),
                               cwd=directory, env=_lammps_environment(), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        reference = read_probe(root / 'uninterrupted/boundary.lammpstrj')
        comparisons = {name: compare(reference, read_probe(root / name / 'boundary.lammpstrj'))
                       for name in variants if name != 'uninterrupted'}
        result = dict(
            state='diagnostic_complete',
            original_restart_sha256=hashlib.sha256(original_restart.read_bytes()).hexdigest(),
            rerun_restart_sha256=hashlib.sha256(rerun_restart.read_bytes()).hexdigest(),
            comparisons=comparisons,
            scope='Two steps diagnose the restart boundary; this is not a 24 ps exact-continuation acceptance test.',
        )
        (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        status.update(state='complete', finished_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        status.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
