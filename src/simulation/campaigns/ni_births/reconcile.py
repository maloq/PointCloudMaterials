"""Verify completed Ni observations and reconcile the original NFS cleanup failure."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.project_runtime.paths import resolve_path


def run(c):
    launch=resolve_path(c['launch']);archive=resolve_path(c['archive'])
    plan=json.loads((launch/'plan.json').read_text());launch_receipt=json.loads((launch/'launch.json').read_text())
    if launch_receipt['identity']!=plan['identity']:raise ValueError('Campaign launch identity changed')
    workers={job for name,job in launch_receipt['jobs'].items() if name=='worker' or name.startswith('worker-')}
    attached={};dispatch=launch/'dispatch.json'
    if dispatch.exists():
        routing=json.loads(dispatch.read_text())
        if routing['campaign_identity']!=plan['identity'] or routing['parent_launch_sha256']!=sha(launch/'launch.json'):
            raise ValueError('Changed Ni worker dispatch ancestry')
        for worker in routing['attached_workers']:
            expected=[r['id'] for i,r in enumerate(plan['records']) if i%c['gpu_workers']==worker['lane']]
            receipt=json.loads(Path(worker['receipt']).read_text())
            if worker['assigned_source_ids']!=expected or not set(worker['source_ids'])<=set(expected) or receipt!={k:v for k,v in worker.items() if k!='receipt'}:
                raise ValueError('Changed Ni attached-worker launch receipt')
            attached.update({source:worker['allocation_id'] for source in worker['source_ids']})
    rows=[]
    for record in plan['records']:
        folder=archive/record['id'];done=folder/'complete.json'
        if not done.exists():
            rows.append(dict(source=record['id'],state='not_complete'));continue
        complete=json.loads(done.read_text());root=folder/'trajectory_binary_float16'
        if complete['job'] not in workers and complete['job']!=attached.get(record['id']):
            raise ValueError('Completion belongs to another job')
        if complete['record']!=record or sha(root/'manifest.json')!=complete['trajectory_manifest_sha256']:
            raise ValueError(f'Changed Ni completion: {folder}')
        b=ShootingBinaryTrajectory.load(root);b.verify_checksums()
        interval=round(c['sample_interval_ps']/c['timestep_ps'])
        if not np.array_equal(b.timesteps,np.arange(complete['frames'])*interval):raise ValueError('Changed timeline')
        if b.atom_count!=c['atoms'] or b.positions.dtype!=np.float16 or b.velocities.dtype!=np.float16:
            raise ValueError('Changed Ni observation layout')
        previous=json.loads((folder/'outcome.json').read_text())
        if previous['state']=='failed':
            log=Path(previous['log']).read_text()
            if 'for path in chunk_paths:shutil.rmtree(path)' not in log or 'OSError: [Errno 39] Directory not empty:' not in log:
                raise ValueError(f'Unrecognized post-completion failure: {folder}')
            receipt=folder/'postprocessing-failure.json'
            if not receipt.exists():write_json(receipt,previous)
            write_json(folder/'outcome.json',dict(complete,postprocessing_state='cleanup_failed',
                reconciliation='Completed trajectory payload/checksums/timeline reverified; original failure retained. No MD rerun.'))
        rows.append(dict(source=record['id'],state='complete',frames=b.frame_count,
            trajectory_manifest_sha256=complete['trajectory_manifest_sha256'],
            cleanup_failure=(folder/'postprocessing-failure.json').exists()))
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID'),
        campaign_identity=plan['identity'],producer_sha256=sha(Path(__file__)),sources=rows,
        completed_sources=sum(r['state']=='complete' for r in rows),planned_sources=len(rows))
    write_json(launch/'verified-status.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    run(json.loads(resolve_path(p.parse_args().config).read_text()))
