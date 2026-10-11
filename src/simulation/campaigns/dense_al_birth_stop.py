"""Continuous Al Q/V collection with an early cluster-growth stopping proxy.

The online trigger allocates simulation time. It is deliberately not a nucleus
birth label; the full 0.10-ps lineage search supplies origins and fates later.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np

from src.project_runtime.transfer import write_json
from .birth_sources import now, sha256

PROTOCOL = 'al_birth_focused_010ps_v1'
POLICY = dict(monitor_interval_steps=1500, cluster_atoms=128,
              overlap_fraction=.5, followup_steps=6000,
              ptm_rmsd_cutoff=.1, cluster_cutoff_A=3.6)


def source_input(config, record, directory):
    from .dense_al import source_input as fixed_input
    # Reuse the exact native initialization. Its zero-step equilibration and
    # reset occur before any integration; all subsequent integration is saved.
    initialized = fixed_input(dict(config, equilibration_steps=0), record)
    prefix = initialized.split('run 0\nrun ', 1)[0] + 'run 0\n'
    helper = shlex.join([sys.executable, '-m', __name__, 'monitor',
                         '--directory', str(directory)])
    finalizer = shlex.join([sys.executable, '-m', __name__, 'finalize',
                            '--directory', str(directory)])
    snapshot = ('write_dump all custom technical/stop-monitor/current.lammpstrj '
                'id type x y z vx vy vz modify sort id '
                'format line "%d %d %.17g %.17g %.17g %.17g %.17g %.17g"')
    callback = '\n'.join([snapshot,
        'shell rm -f technical/stop-monitor/decision.lammps', 'shell '+helper,
        'include technical/stop-monitor/decision.lammps',
        'if "${birth_stop} == 1" then "jump SELF birth_done"'])
    return prefix + f'''{callback}
label birth_monitor_loop
run ${{birth_chunk}}
{callback}
jump SELF birth_monitor_loop
label birth_done
restart 0
undump trajectory
write_restart final.restart.bin
shell rm -f technical/stop-monitor/final-ok.lammps
shell {finalizer}
include technical/stop-monitor/final-ok.lammps
print "DENSE_AL_COMPLETE {record['run_id']}"
'''


def observation(directory, metadata):
    from src.data.conversion.shooting_text import iter_lammps_shooting_frames_for_conversion
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier, ClusterAnalysisModifier
    path = directory/'technical/stop-monitor/current.lammpstrj'
    with path.open() as stream:
        if stream.readline() != 'ITEM: TIMESTEP\n':
            raise ValueError(f'Missing native step in {path}')
        step = int(stream.readline())
    frames = list(iter_lammps_shooting_frames_for_conversion(path, timesteps=[step],
        atom_count=metadata['atom_count'], exact_timeline=True))
    if len(frames) != 1:
        raise ValueError(f'Expected exactly one monitoring frame: {path}')
    frame, = frames
    if not np.all(frame.atom_types == 1):
        raise ValueError(f'Non-Al atom types: {path}')
    lengths = (frame.box_high-frame.box_low).astype(float)
    data = DataCollection(); cell = SimulationCell(pbc=(True, True, True))
    cell[:, :3] = np.diag(lengths); data.objects.append(cell)
    particles = Particles()
    particles.create_property('Position', data=np.mod(frame.positions.astype(float), lengths))
    data.objects.append(particles)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=POLICY['ptm_rmsd_cutoff']))
    selected = np.isin(np.asarray(data.particles['Structure Type']), [1, 2, 3])
    data.particles_.create_property('Selection', data=selected.astype(np.int32))
    data.apply(ClusterAnalysisModifier(cutoff=POLICY['cluster_cutoff_A'], only_selected=True))
    labels = np.asarray(data.particles['Cluster'], dtype=np.int32).copy()
    sizes = np.bincount(labels)
    row = dict(step=step, time_ps=step*.002, crystal_fraction=float(selected.mean()),
        largest_crystal_cluster=int(sizes[1:].max(initial=0)),
        snapshot_sha256=sha256(path), observed_at=now())
    return row, labels, frame.atom_ids


def monitor(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'input_metadata.json').read_text())
    policy = metadata['stopping']
    if (metadata['protocol'], metadata['timestep_ps'], metadata['sample_interval_steps']) != (PROTOCOL, .002, 50):
        raise ValueError('Birth-focused monitoring requires the declared 2-fs/0.10-ps protocol')
    if any(policy[k] != v for k, v in POLICY.items()):
        raise ValueError('Birth-focused stopping policy changed')
    folder = directory/'technical/stop-monitor'; hp = folder/'history.json'
    history = json.loads(hp.read_text()) if hp.exists() else []
    row, labels, ids = observation(directory, metadata)
    step = row['step']; cap = policy['maximum_measurement_steps']
    expected = min(history[-1]['step']+POLICY['monitor_interval_steps'], cap) if history else 0
    if step != expected or cap % 50 or step > cap:
        raise ValueError(f'Invalid monitoring timeline: expected {expected}, got {step}, cap {cap}')
    if not history and row['largest_crystal_cluster'] >= 8:
        raise ValueError('Initial melt has an >=8-atom PTM embryo; retain as failed QC, do not silently resample')
    trigger = history[-1]['trigger'] if history else None
    if history and trigger is None:
        with np.load(folder/'previous.npz', allow_pickle=False) as previous:
            if not np.array_equal(ids, previous['ids']):
                raise ValueError('Atom identity/order changed between monitoring frames')
            old = previous['labels']; old_sizes = np.bincount(old)
            sizes = np.bincount(labels)
            for cid in np.flatnonzero(sizes >= POLICY['cluster_atoms']):
                if cid == 0:
                    continue
                overlaps = np.bincount(old[labels == cid], minlength=len(old_sizes))
                candidates = np.flatnonzero((old_sizes >= POLICY['cluster_atoms']) &
                    (overlaps >= POLICY['overlap_fraction']*np.minimum(old_sizes, sizes[cid])))
                candidates = candidates[candidates != 0]
                if len(candidates):
                    oid = int(candidates[np.argmax(overlaps[candidates])])
                    trigger = dict(first_observed_step=history[-1]['step'], confirmation_step=step,
                        first_size=int(old_sizes[oid]), confirmation_size=int(sizes[cid]),
                        shared_atoms=int(overlaps[oid]),
                        interpretation='Matched >=128-atom components at consecutive 3-ps checks; not continuous persistence or a birth label')
                    break
    row['trigger'] = trigger
    history.append(row); write_json(hp, history)
    np.savez(folder/'previous.npz', labels=labels, ids=ids)
    followup_done = trigger is not None and step >= trigger['confirmation_step']+POLICY['followup_steps']
    stop = followup_done or step == cap
    if stop:
        tail = step-trigger['confirmation_step'] if trigger else 0
        write_json(folder/'decision.json', dict(protocol=PROTOCOL,
            reason='cluster_followup_complete' if followup_done else 'duration_cap',
            trigger=trigger, final_step=step, measurement_ps=step*.002,
            maximum_measurement_steps=cap, recorded_from_initialization=True,
            retained_followup_ps=tail*.002, followup_truncated=bool(trigger and not followup_done),
            birth_label_status='not_searched', unresolved_fates='right_censored', recorded_at=now()))
    command = f'variable birth_stop equal {int(stop)}\nvariable birth_chunk equal {min(POLICY["monitor_interval_steps"], cap-step)}\n'
    temporary = folder/'decision.lammps.building'; temporary.write_text(command)
    temporary.replace(folder/'decision.lammps')
    return row


def finalize(directory):
    directory = Path(directory); folder = directory/'technical/stop-monitor'
    decision = json.loads((folder/'decision.json').read_text())
    history = json.loads((folder/'history.json').read_text()); row = history[-1]
    if row['step'] != decision['final_step'] or sha256(folder/'current.lammpstrj') != row['snapshot_sha256']:
        raise ValueError('Final state does not match the saved stopping observation')
    decision.update(state='dynamics_complete', final_crystal_fraction=row['crystal_fraction'],
        final_largest_crystal_cluster=row['largest_crystal_cluster'], final_snapshot_sha256=row['snapshot_sha256'])
    write_json(folder/'outcome.json', decision)
    (folder/'final-ok.lammps').write_text('print "DENSE_BIRTH_ENDPOINT_VERIFIED"\n')
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['monitor', 'finalize'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    os.environ['OVITO_THREAD_COUNT'] = '1'
    print(json.dumps(globals()[args.action](args.directory)), flush=True)


if __name__ == '__main__':
    main()
