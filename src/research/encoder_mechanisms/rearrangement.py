"""Same-atom MD displacements versus equal-RMS synthetic perturbations."""
import fcntl
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import centered
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path,resolve_path
from src.research.structural_state.common import sha,write_json
from src.experiment_runner.metric_docs import write_metric_table


def prepare(c):
    root=resolve_path(c['output'])/'technical/rearrangement-inputs';root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=root/'pairs.npz';receipt=root/'complete.json'
        if receipt.exists():
            if sha(path)!=json.loads(receipt.read_text())['sha256']:raise ValueError('Displacement assay inputs changed')
            return path
        _,release=read_release(c['fixed_dataset']['root'])
        if release['identity']!=c['fixed_dataset']['identity']:raise ValueError('Displacement assay release changed')
        before=[];after=[];sources=[];frames=[];atom_ids=[];receipts=[]
        for source in release['sources']:
            if source['role']!='test':continue
            raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
            if sha(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Raw MD manifest changed')
            rng=np.random.default_rng(np.random.SeedSequence([c['evaluation_seed'],source['id']]))
            selected=np.sort(rng.choice(source['frame_count']-1,2,replace=False))
            atoms=rng.choice(source['center_atom_ids'],16,replace=False)
            rows=np.searchsorted(raw.atom_ids,atoms);np.testing.assert_array_equal(raw.atom_ids[rows],atoms)
            for frame in selected:
                box=raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float)
                box2=raw.box_high[frame+1].astype(float)-raw.box_low[frame+1].astype(float)
                p=np.mod(raw.positions[frame].astype(float),box)
                q=np.mod(raw.positions[frame+1].astype(float),box2)
                nearest=cKDTree(p,boxsize=box).query(p[rows],k=80,workers=1)[1]
                np.testing.assert_array_equal(nearest[:,0],rows)
                before.append(centered(p,box,rows,nearest));after.append(centered(q,box2,rows,nearest))
                sources.extend([source['id']]*len(rows));frames.extend([int(frame)]*len(rows));atom_ids.extend(atoms)
            receipts.append(dict(source=source['id'],frames=selected.tolist(),center_atom_ids=atoms.tolist(),manifest_sha256=source['manifest_sha256']))
        x=np.concatenate(before).astype(np.float32);y=np.concatenate(after).astype(np.float32)
        delta=y[:,1:]-x[:,1:];rms=np.sqrt(np.mean(np.sum(delta.astype(float)**2,axis=2),axis=1))
        # A center-relative Gaussian control with exactly the same per-pair RMS.
        rng=np.random.default_rng(c['evaluation_seed']+179);noise=rng.normal(size=(3,*x.shape)).astype(np.float32)
        noise[:,:,0]=0
        denom=np.sqrt(np.mean(np.sum(noise[:,:,1:].astype(float)**2,axis=3),axis=2))
        noise*= (rms[None,:]/denom)[:,:,None,None]
        noisy=x[None]+noise
        realized=np.sqrt(np.mean(np.sum((noisy[:,:,1:]-x[None,:,1:]).astype(float)**2,axis=3),axis=2))
        np.testing.assert_allclose(realized,np.broadcast_to(rms,realized.shape),atol=2e-6,rtol=2e-6)
        bonds=np.mean((np.linalg.norm(x[:,1:],axis=2)<3.6)!=(np.linalg.norm(y[:,1:],axis=2)<3.6),axis=1)
        d2=[]
        for a,b in zip(x,y):
            affine=np.linalg.lstsq(a[1:13].astype(float),b[1:13].astype(float),rcond=None)[0]
            d2.append(float(np.mean(np.sum((b[1:13]-a[1:13]@affine)**2,axis=1))))
        np.savez(path,before=x,after=y,synthetic=noisy,source=np.asarray(sources),frame=np.asarray(frames),
            atom_id=np.asarray(atom_ids),displacement_rms_A=rms,bond_change_fraction=bonds,d2min_A2=d2)
        write_json(receipt,dict(sha256=sha(path),sources=receipts,release=release['identity'],
            lag_ps=.75,neighbor_identity='Origin nearest80 atom IDs tracked into next frame; center recentered at both times',
            selection='Two outcome-independent frame origins and 16 fixed benchmark centers per held-out source',
            synthetic='Three isotropic Gaussian center-fixed perturbations; exact per-pair RMS over 79 noncentral atoms',
            warning='Synthetic matched-amplitude control is not a thermal MD distribution'))
        return path


def evaluate(c,q,spec,output,device):
    from src.research.encoder_quality.run import load_encoder,encode
    path=prepare(c)
    with np.load(path) as a:data=dict(a)
    model,_=load_encoder(spec,device)
    x=encode(model,data['before'],256,device,compile_first=True)
    y=encode(model,data['after'],256,device)
    synthetic=np.stack([encode(model,v,256,device) for v in data['synthetic']])
    motion=np.sum((y-x).astype(float)**2,axis=1)
    noise=np.mean(np.sum((synthetic-x[None]).astype(float)**2,axis=2),axis=0)
    # Same-population normalization; no ratio between unrelated assays.
    trace=float(np.mean(np.sum((x-x.mean(0)).astype(float)**2,axis=1)))
    if trace<=0:raise ValueError('Collapsed representation in displacement assay')
    result={}
    for source in np.unique(data['source']):
        keep=data['source']==source
        record=dict(pairs=int(keep.sum()),real_normalized_rms=float(np.sqrt(motion[keep].mean()/(2*trace))),
            synthetic_normalized_rms=float(np.sqrt(noise[keep].mean()/(2*trace))),
            real_to_synthetic_rms=float(np.sqrt(motion[keep].sum()/noise[keep].sum())) if noise[keep].sum()>0 else None)
        for field in ('displacement_rms_A','bond_change_fraction','d2min_A2'):
            r=spearmanr(motion[keep]-noise[keep],data[field][keep]).statistic
            record[f'excess_response_spearman_{field}']=float(r) if np.isfinite(r) else None
        result[str(source)]=record
    metrics=dict(per_source=result,mean_source_real_to_synthetic_rms=float(np.mean([r['real_to_synthetic_rms'] for r in result.values() if r['real_to_synthetic_rms'] is not None])),
        reference_trace=trace,input_sha256=sha(path),lag_ps=.75,
        interpretation='Sensitivity diagnostic at equal tracked-atom RMS; bond and affine-residual associations are observational, not a causal rearrangement classifier')
    dest=output/'analyses/displacement';dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'pair-responses.npz',real_squared=motion,synthetic_squared=noise,
        **{k:data[k] for k in ('source','frame','atom_id','displacement_rms_A','bond_change_fraction','d2min_A2')})
    write_json(dest/'technical/metrics.json',metrics);write_metric_table(metrics,dest,family='encoder_mechanisms')
