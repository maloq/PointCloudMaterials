"""Atomic neighborhood extraction with bounded asynchronous frame preparation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
import torch

from src.project_runtime.paths import resolve_path,dataset_path
from src.data.structural_pretraining.prepare import file_hash,save_json
from src.data.structural_pretraining.support import REFERENCE_RADIUS,support_weights
from src.data.structural_pretraining.batches import collate,move
from src.data.predictive_memory.targets import taper
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.models.encoders.structural import StructuralMACE
from src.training_methods.shared_pretraining.compilation import compile_encoder
from .geometry import stencil,representatives


def encoders(plan,device='cuda'):
    c=plan['structured_config'];root=resolve_path(c['output'])/'technical'
    spec=importlib.util.spec_from_file_location('src.models.encoders._gatr3072',root/'gatr-producer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=torch.load(root/'gatr.pt',map_location='cpu',weights_only=False)
    if file_hash(root/'gatr-producer.py')!=saved['identity']['implementation']['files']['src/models/encoders/structural.py']:
        raise ValueError('Historical GATr producer differs from the checkpoint receipt')
    gatr=module.StructuralGATr().to(device).eval();gatr.load_state_dict(saved['encoder'],strict=True)
    mace=StructuralMACE().to(device).eval();parent=torch.load(root/'parent.pt',map_location='cpu',weights_only=False)
    mace.load_state_dict({k.removeprefix('encoder.'):v for k,v in parent['model'].items() if k.startswith('encoder.')},strict=True)
    mace.structured_compiled=False
    for model in (mace,gatr):model.requires_grad_(False)
    return dict(mace=mace,gatr=gatr)


def observations(points,tree,atoms,scale,architecture,box=None):
    """Preserve each checkpoint's original support and atom-centered inputs."""
    factor=REFERENCE_RADIUS/scale;outer=8. if architecture=='mace' else 17.
    candidates=tree.query_ball_point(points[atoms],outer/factor,return_sorted=True,workers=1)
    samples=[]
    for center,rows in zip(atoms,candidates,strict=True):
        rows=np.asarray(rows);rows=np.r_[center,rows[rows!=center]]
        x=points[rows].astype(float)-points[center].astype(float)
        if box is not None:x-=box*np.round(x/box)
        x=(x*factor).astype(np.float32) if architecture=='mace' else x.astype(np.float32)*factor
        x=x[np.linalg.norm(x,axis=1)<outer]
        w=support_weights(x) if architecture=='mace' else taper(np.linalg.norm(x,axis=1),15.,17.)
        item=dict(positions=x[None],weights=w.astype(np.float32)[None],center=0,times=np.array([0.],np.float32),species=1,
            log_scale=np.log(scale/REFERENCE_RADIUS),physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False)
        if architecture=='mace':
            pairs=cKDTree(x).query_pairs(5.,output_type='ndarray');item['edges']=np.concatenate((pairs,pairs[:,::-1]),axis=0).T.astype(np.int64)
        samples.append(item)
    return samples


def frame_input(raw,frame,center_rows,queries,scale,max_offset):
    box=(raw.box_high[frame]-raw.box_low[frame]).astype(float)
    points=np.mod(raw.positions[frame].astype(float),box);tree=cKDTree(points,boxsize=box)
    ids=[];relative=[]
    if frame<=664:
        for center in center_rows:
            atoms,positions=representatives(points,center,tree,box,queries,max_offset)
            ids.append(atoms);relative.append(positions)
    else:
        ids=center_rows[:,None];relative=np.zeros((len(center_rows),1,3))
    ids=np.array(ids);unique,inverse=np.unique(ids,return_inverse=True)
    return dict(mapping=inverse.reshape(ids.shape),atom_ids=np.array(raw.atom_ids[ids]),relative=np.array(relative),
        graphs={name:observations(points,tree,unique,scale,name,box) for name in ('mace','gatr')})


@torch.no_grad()
def infer(model,samples,architecture,batch_size,device='cuda'):
    result=[]
    for start in range(0,len(samples),batch_size):
        batch=move(collate(samples[start:start+batch_size],architecture),device)
        if architecture=='mace':
            if not model.structured_compiled:
                compile_encoder(model,batch,'bf16');model.structured_compiled=True
            with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=device!='cpu'):
                value=model(batch).float()
        else:value=model(batch).float()
        if value.shape!=(len(samples[start:start+batch_size]),128) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid {architecture} export at graph {start}: {value.shape}')
        result.append(value.cpu().numpy())
    return np.concatenate(result)


def source(plan,source,models,deadline,progress=None):
    c=plan['structured_config'];root=resolve_path(c['context_cache'])/str(source['id']);root.mkdir(parents=True,exist_ok=True)
    receipt=root/'complete.json'
    if receipt.exists():
        if json.loads(receipt.read_text())['identity']!=plan['structured_identity']:raise ValueError('Structured source identity changed')
        return True
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Raw source changed')
    centers=np.searchsorted(raw.atom_ids,source['center_atom_ids']);np.testing.assert_array_equal(raw.atom_ids[centers],source['center_atom_ids'])
    np.testing.assert_allclose(raw.timesteps*source['timestep_fs']/1000,np.arange(801)*.75,atol=1e-6,rtol=0)
    state=root/'progress.json';start=0
    if state.exists():
        previous=json.loads(state.read_text())
        if previous['identity']!=plan['structured_identity']:raise ValueError('Partial structured source identity changed')
        start=previous['completed_frames']
    arrays={}
    shapes={'relative':(167,16,25,3),'atom_ids':(167,16,25),
        **{f'{name}_features':(167,16,25,128) for name in models},**{f'{name}_center':(199,16,128) for name in models}}
    for name,shape in shapes.items():
        p=root/f'{name}.npy'
        arrays[name]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',
            dtype=np.int64 if name=='atom_ids' else np.float32,shape=shape)
    queries=stencil(c['shell_radii_A']);began=time.monotonic()
    # One prepared frame ahead overlaps KD-tree/crop/edge work with GPU inference.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(frame_input,raw,start*4,centers,queries,plan['scale'],c['max_query_offset_A']) if start<199 else None
        for fi in range(start,199):
            if time.time()>deadline-300:return False
            item=pending.result()
            if fi+1<199:pending=pool.submit(frame_input,raw,(fi+1)*4,centers,queries,plan['scale'],c['max_query_offset_A'])
            for name,model in models.items():
                values=infer(model,item['graphs'][name],name,c[f'{name}_extraction_batch'])
                values=values[item['mapping']]
                if fi<167:arrays[f'{name}_features'][fi]=values
                arrays[f'{name}_center'][fi]=values[:,0]
            if fi<167:
                arrays['relative'][fi]=item['relative'];arrays['atom_ids'][fi]=item['atom_ids']
            if fi%8==0 or fi==198:
                for a in arrays.values():a.flush()
                save_json(state,dict(identity=plan['structured_identity'],completed_frames=fi+1,seconds=time.monotonic()-began))
                if progress:progress(fi+1)
    # Existing MACE center timeline is an independent producer/replay check.
    old=np.load(resolve_path(plan['config']['future_cache'])/str(source['id'])/'center.npy')
    np.testing.assert_allclose(arrays['mace_center'],old,atol=1e-4,rtol=1e-4)
    offsets=arrays['relative']-queries[None,None]
    save_json(receipt,dict(identity=plan['structured_identity'],source=source['id'],frames=199,
        max_query_offset_A=float(np.linalg.norm(offsets,axis=-1).max()),
        mace_center_replay_max_error=float(abs(arrays['mace_center']-old).max()),
        files={f'{k}.npy':file_hash(root/f'{k}.npy') for k in arrays},seconds=time.monotonic()-began))
    return True
