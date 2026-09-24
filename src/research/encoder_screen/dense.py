"""Eightfold spatial plot density on the same slices; numerical cohorts stay fixed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from .common import load_config,checked,sha,write


def prepare(config):
    from src.data.static_sources import load_points
    from src.research.geoframe_evolution.reference import bond_descriptors,contexts
    root=Path(config['output']);reference=Path(config['reference']);folder=root/config['spatial_plot']['inputs']
    folder.mkdir(exist_ok=True)
    original=json.loads((reference/'manifest.json').read_text());receipt=dict(frames=[],files={},factor=config['spatial_plot']['factor'],original_reference_sha256=sha(reference/'manifest.json'))
    for i in [2,4,7]:
        record=original['frames'][i];a=np.load(reference/f'frame-{i:02d}.npz');n=record['anchor_count'];xyz=a['coords'][:n]
        middle=np.median(xyz[:,2]);half=np.ptp(xyz[:,2])*.12;slab=abs(xyz[:,2]-middle)<half;old=np.flatnonzero(slab)
        checked(record['file'],record['input_sha256']);points=load_points(record['file']);tree=cKDTree(points,balanced_tree=False)
        radius=record['radius_A'];lo=points.min(0);hi=points.max(0)
        eligible=np.flatnonzero(((points>lo+2*radius)&(points<hi-2*radius)).all(1)&(abs(points[:,2]-middle)<half))
        previous=np.load(root/config['spatial_plot']['previous_inputs']/f'labels-{i:02d}.npz')
        np.testing.assert_array_equal(previous['original_indices'],old)
        if len(previous['rows'])!=4*len(old):raise ValueError('Expected the prior fourfold slice')
        eligible=np.setdiff1d(eligible,np.r_[a['rows'][:n],previous['rows'][len(old):]],assume_unique=True)
        added=np.random.default_rng(20260924+i).choice(eligible,4*len(old),replace=False)
        rows=np.r_[previous['rows'],added];new=rows[len(old):]
        _,near=tree.query(points[rows],k=81,workers=1)
        k=record['primary_order_neighbors'];orders=[]
        for start in range(0,len(rows),128):
            core=near[start:start+128,:k+1];_,neighbors=tree.query(points[core],k=k+1,workers=1)
            v=points[neighbors[:,:,1:]].astype(np.float64)-points[core][:,:,None]
            order,_=bond_descriptors(v);orders.append(order)
        order=np.concatenate(orders);fullpath=reference/f'full-reference-{i:02d}.npz';full=np.load(fullpath)
        best=full['best_ptm'];rmsd=full['rmsd'];fault=full['planar_fault']
        ptm=np.where(rmsd[rows]<=.1,best[rows],0);solid=np.isin(best,[1,2,3])&(rmsd<=.1)
        fraction=solid[near[:,1:15]].mean(1)
        labels=contexts(ptm,fraction,fault[rows],order,record['order_threshold_qbar6'],record['material'],-.08)
        np.testing.assert_array_equal(labels[:len(old)],a['context'][:n,1][old],err_msg='Dense reference does not reproduce the original spatial labels')
        np.testing.assert_array_equal(labels[:len(previous['rows'])],previous['context'],err_msg='Prior fourfold plot labels changed')
        groups=tree.query_ball_point(points[new],10.,return_sorted=True,workers=1);patches=[];centers=[]
        for atom,ids in zip(new,groups,strict=True):
            ids=np.asarray(ids);centers.append(int(np.flatnonzero(ids==atom).item()));patches.append((points[ids].astype(np.float64)-points[atom].astype(np.float64)).astype(np.float32))
        nearest80=(points[near[len(old):,:80]].astype(np.float64)-points[new,None].astype(np.float64)).astype(np.float32)
        path=folder/f'frame-{i:02d}.npz'
        np.savez(path,positions=np.concatenate(patches),offsets=np.r_[0,np.cumsum([len(p) for p in patches])],centers=np.array(centers),nearest80=nearest80)
        np.savez(folder/f'labels-{i:02d}.npz',coords=points[rows],rows=rows,context=labels,original_indices=old)
        receipt['files'][path.name]=sha(path);receipt['files'][f'labels-{i:02d}.npz']=sha(folder/f'labels-{i:02d}.npz')
        receipt['frames'].append(dict(record,count=len(new),anchor_count=len(new),original_slab_count=len(old),dense_slab_count=len(rows),slab_middle_A=float(middle),slab_half_width_A=float(half),full_reference_sha256=sha(fullpath)))
        print(record['material'],len(old),'->',len(rows),'same slab',flush=True)
    write(folder/'manifest.json',receipt);return receipt


def evaluate(config,task):
    root=Path(config['output']);folder=root/'technical/evaluations'/task['name'];inputs=root/config['spatial_plot']['inputs'];manifest=json.loads((inputs/'manifest.json').read_text())
    for f,h in manifest['files'].items():checked(inputs/f,h)
    destination=folder/'dense8-embeddings';record=dict(task,inputs=str(inputs),destination=str(destination),inference_driver_sha256=sha(Path(__file__).with_name('native.py')))
    write(folder/'dense8-task.json',record)
    if not (destination/'extraction.json').exists():
        command=[sys.executable,'-u',str(Path(__file__).with_name('native.py')),'--record',str(folder/'dense8-task.json'),'--static-only']
        with (folder/'dense8-inference.log').open('w') as log:subprocess.run(command,cwd=task['producer'],stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=json.loads((destination/'extraction.json').read_text())
    if receipt['task_sha256']!=sha(folder/'dense8-task.json') or receipt['inputs_sha256']!=sha(inputs/'manifest.json'):raise ValueError('Dense plot extraction identity mismatch')
    # Render reuses the exact old fitting population and checks cluster identities.
    from .figures import render
    previous=folder/'figures.json'
    if previous.exists():
        history=root/'technical/fourfold-spatial-panels';history.mkdir(exist_ok=True)
        for name in json.loads(previous.read_text())['plots']:
            target=history/Path(name).name
            if not target.exists():shutil.copy2(root/name,target)
    paths=render(config,task,dense=True)
    write(folder/'dense8-figures.json',dict(state='complete',factor=8,marker_area_points2=config['spatial_plot']['marker_area_points2'],marker_diameter_ratio=.5,plots=paths,
        samples={r['material']:dict(original=r['original_slab_count'],plotted=r['dense_slab_count']) for r in manifest['frames'] if r['material'] in task['materials']},
        numerical_metrics_unchanged=True))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','worker']);p.add_argument('--config',required=True);p.add_argument('--name');p.add_argument('--watch',action='store_true');a=p.parse_args();c=load_config(a.config);root=Path(c['output'])
    if a.stage=='prepare':prepare(c);return
    deadline=float('inf')
    if 'SLURM_JOB_ID' in os.environ:
        from src.training_methods.shared_pretraining.queue import deadline_for_job
        deadline=deadline_for_job()
    while True:
        pending=0
        for t in c['tasks']:
            if a.name and t['name']!=a.name:continue
            folder=root/'technical/evaluations'/t['name']
            if (folder/'dense8-figures.json').exists() or (folder/'dense8-failed.json').exists() or (folder/'failed.json').exists():continue
            if not (folder/'complete.json').exists():pending+=1;continue
            write(root/'technical/dense8-status.json',dict(state='running',task=t['name'],factor=8,pid=os.getpid()))
            try:
                evaluate(c,t)
                # Numerical report stays on its immutable producer/definitions.
                launch=json.loads((root/'technical/launch.json').read_text())
                with (root/'technical/report.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    subprocess.run([sys.executable,'-m','src.research.encoder_screen.report','--config',launch['config']],cwd=launch['code'],env=dict(os.environ,PCM_PROJECT_ROOT=launch['code']),check=True)
            except Exception as exc:
                write(folder/'dense8-failed.json',dict(error=repr(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
        if not a.watch or not pending or time.time()>deadline-600:break
        time.sleep(20)
    write(root/'technical/dense8-status.json',dict(state='finished',pid=os.getpid(),factor=8))


if __name__=='__main__':main()
