"""Online whole-cell halfway detection for versioned, unstarted Al descendants."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np

from src.analysis.al_replay import structure
from src.data.conversion.shooting_text import iter_lammps_shooting_frames_for_conversion
from src.project_runtime.transfer import write_json
from .birth_sources import now, sha256

PROTOCOLS = {50: 'al_dense_half_stop_010ps_v1', 5: 'al_dense_half_stop_001ps_v1'}
MONITOR_STEPS = 7500
TAIL_STEPS = 3000


def source_input(config, record, directory):
    """Keep one native LAMMPS process and its NPT fix across all observations."""
    from .dense_al import source_input as fixed_input
    interval = record['sample_interval_steps']
    fixed_config = dict(config, sample_interval_steps=interval)
    original = fixed_input(fixed_config, record)
    prefix = original.split('run 0\nrun ', 1)[0] + 'run 0\n'
    cap = record['stopping']['maximum_measurement_steps']
    helper = shlex.join([sys.executable, '-m', 'src.simulation.campaigns.dense_al_half_stop',
                         'monitor', '--directory', str(directory)])
    finalizer = shlex.join([sys.executable, '-m', 'src.simulation.campaigns.dense_al_half_stop',
                            'finalize', '--directory', str(directory)])
    snapshot = ('write_dump all custom technical/stop-monitor/current.lammpstrj '
                'id type x y z vx vy vz modify sort id '
                'format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"')
    callback = '\n'.join([snapshot, 'shell rm -f technical/stop-monitor/decision.lammps',
                           'shell ' + helper, 'include technical/stop-monitor/decision.lammps',
                           'if "${half_stop} == 1" then "jump SELF half_done"'])
    return prefix + f'''{callback}
label half_monitor_loop
run ${{half_chunk}}
{callback}
jump SELF half_monitor_loop
label half_done
if "${{half_tail}} > 0" then "run ${{half_tail}}"
restart 0
undump trajectory
write_restart final.restart.bin
{snapshot}
shell rm -f technical/stop-monitor/final-ok.lammps
shell {finalizer}
include technical/stop-monitor/final-ok.lammps
print "DENSE_AL_COMPLETE {record['run_id']}"
'''


def observation(directory, metadata):
    path = directory / 'technical/stop-monitor/current.lammpstrj'
    with path.open() as handle:
        if handle.readline() != 'ITEM: TIMESTEP\n':
            raise ValueError(f'{path}: missing initial timestep marker')
        step = int(handle.readline())
    frames = list(iter_lammps_shooting_frames_for_conversion(path, timesteps=[step],
                  atom_count=metadata['atom_count'], exact_timeline=True))
    if len(frames) != 1:
        raise ValueError(f'{path}: expected one native observation')
    frame = frames[0]
    if not np.all(frame.atom_types == 1):
        raise ValueError(f'{path}: non-Al atom types')
    os.environ['OVITO_THREAD_COUNT'] = '1'
    _, stats = structure(frame.positions.astype(float),
                         (frame.box_high - frame.box_low).astype(float))
    return dict(step=step, time_ps=step * .002, crystal_fraction=stats['crystal_fraction'],
                largest_crystal_cluster=stats['largest_crystal_cluster'],
                snapshot_sha256=sha256(path), observed_at=now())


def monitor(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'input_metadata.json').read_text())
    if metadata['protocol'] not in PROTOCOLS.values() or metadata['timestep_ps'] != .002:
        raise ValueError('Halfway monitor requires the declared 2-fs Al protocol')
    policy = metadata['stopping']
    if (policy['monitor_interval_steps'], policy['prediction_tail_steps'], policy['crystal_fraction']) != (7500, 3000, .5):
        raise ValueError('Changed halfway criterion, confirmation cadence or prediction tail')
    folder = directory / 'technical/stop-monitor'
    history_path = folder / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    row = observation(directory, metadata)
    step = row['step']; cap = policy['maximum_measurement_steps']
    expected = min(history[-1]['step'] + MONITOR_STEPS, cap) if history else 0
    if step != expected or step > cap:
        raise ValueError(f'{directory}: expected monitor step {expected}, observed {step}, cap {cap}')
    previous = history[-1] if history else None
    confirmed = (previous is not None and previous['crystal_fraction'] >= .5 and
                 row['crystal_fraction'] >= .5 and step - previous['step'] == MONITOR_STEPS)
    row['confirmed_half_crossing'] = bool(confirmed)
    history.append(row)
    write_json(history_path, history)
    if confirmed or step == cap:
        tail = min(TAIL_STEPS, cap - step) if confirmed else 0
        decision = dict(protocol=metadata['protocol'], reason='half_crystal' if confirmed else 'peer_time_cap',
            detection_step=step, confirmation_time_ps=step * .002,
            first_half_observation_ps=previous['time_ps'] if confirmed else None,
            confirmed_half_crossing=bool(confirmed), half_event_right_censored=not bool(confirmed),
            retained_prediction_tail_steps=tail, retained_prediction_tail_ps=tail * .002,
            prediction_tail_truncated=bool(confirmed and tail < TAIL_STEPS),
            final_step=step + tail, measurement_ps=(step + tail) * .002,
            maximum_measurement_steps=cap, maximum_measurement_ps=cap * .002,
            peer90_observed_in_reference=policy['peer90_observed_in_reference'],
            recorded_at=now())
        write_json(folder / 'decision.json', decision)
        command = f'variable half_stop equal 1\nvariable half_tail equal {tail}\n'
    else:
        command = ('variable half_stop equal 0\nvariable half_tail equal 0\n'
                   f'variable half_chunk equal {min(MONITOR_STEPS, cap - step)}\n')
    # LAMMPS removes the previous include before calling us. A failed helper
    # therefore causes a missing-include error rather than reuse of a stale decision.
    temporary = folder / 'decision.lammps.building'
    temporary.write_text(command)
    temporary.replace(folder / 'decision.lammps')
    return row


def finalize(directory):
    directory = Path(directory)
    folder = directory / 'technical/stop-monitor'
    decision = json.loads((folder / 'decision.json').read_text())
    metadata = json.loads((directory / 'input_metadata.json').read_text())
    row = observation(directory, metadata)
    if row['step'] != decision['final_step'] or row['step'] % metadata['sample_interval_steps']:
        raise ValueError(f'{directory}: final observation does not match the declared sampled endpoint')
    decision.update(final_crystal_fraction=row['crystal_fraction'], final_snapshot_sha256=row['snapshot_sha256'],
                    final_largest_crystal_cluster=row['largest_crystal_cluster'], state='dynamics_complete')
    write_json(folder / 'outcome.json', decision)
    (folder / 'final-ok.lammps').write_text('print "DENSE_HALF_ENDPOINT_VERIFIED"\n')
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['monitor', 'finalize'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(monitor(args.directory) if args.action == 'monitor' else finalize(args.directory)))


if __name__ == '__main__':
    main()
