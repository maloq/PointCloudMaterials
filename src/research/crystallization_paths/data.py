"""Immutable future-center extraction and GPU-resident, nonleaking timeline joins."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from src.project_runtime.paths import resolve_path, dataset_path
from src.data.structural_pretraining.prepare import file_hash, save_json
from src.data.structural_pretraining.support import REFERENCE_RADIUS, OUTER_RADIUS, EDGE_CUTOFF, support_weights
from src.data.structural_pretraining.batches import collate, move
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.models.encoders.structural import StructuralMACE
from src.training_methods.shared_pretraining.compilation import compile_encoder
from src.research.crystallization_transfer.runtime import parent_state
from src.research.crystallization_transfer.data import Corpus

STATE_DIM = 265  # fixed MACE 128, physical packet 128, bond order 8, instantaneous PTM crystal 1
STEPS = 32
SUBSTEPS = 4


def prepare_source(plan, source, model=None):
    """Extend only center embeddings to 594 ps; original graph cache stays immutable."""
    sid=source['id'];folder=resolve_path(plan['config']['future_cache'])/str(sid)
    folder.mkdir(parents=True,exist_ok=True);receipt=folder/'complete.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['checkpoint_sha256']!=plan['checkpoint_sha256'] or record['identity']!=plan.get('future_cache_identity',plan['identity']):
            raise ValueError(f'Future cache identity changed for source {sid}')
        if file_hash(folder/'center.npy')!=record['sha256']:raise ValueError(f'Future feature checksum: {sid}')
        return model
    if 'future_cache_identity' in plan:raise FileNotFoundError(f'Reused future cache is incomplete: {receipt}')
    original=resolve_path(plan['config']['cache'])/str(sid)
    mapping=np.load(original/'mapping.npy',mmap_mode='r');features=np.load(original/'features.npy',mmap_mode='r')
    frame_ids=np.arange(0,max(plan['anchors'])+max(plan['lags'])+1,4)
    result=np.empty((len(frame_ids),len(source['center_atom_ids']),128),np.float32)
    result[:len(mapping)]=features[mapping[:,:,0],:128]
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError(f'Raw manifest changed: {sid}')
    ids=np.asarray(source['center_atom_ids']);centers=np.searchsorted(raw.atom_ids,ids)
    np.testing.assert_array_equal(raw.atom_ids[centers],ids)
    np.testing.assert_allclose(raw.timesteps*source['timestep_fs']/1000,np.arange(801)*.75,rtol=0,atol=1e-6)
    factor=REFERENCE_RADIUS/plan['scale'];graphs=[]
    for frame in frame_ids[len(mapping):]:
        box=(raw.box_high[frame]-raw.box_low[frame]).astype(float)
        points=np.mod(raw.positions[frame].astype(float),box);tree=cKDTree(points,boxsize=box)
        for center in centers:
            keep=np.asarray(tree.query_ball_point(points[center],OUTER_RADIUS/factor),int)
            keep=np.r_[center,np.sort(keep[keep!=center])]
            x=points[keep]-points[center];x-=box*np.round(x/box);x=(x*factor).astype(np.float32)
            x=x[np.linalg.norm(x,axis=1)<OUTER_RADIUS]
            pairs=cKDTree(x).query_pairs(EDGE_CUTOFF,output_type='ndarray')
            edges=np.concatenate((pairs,pairs[:,::-1]),0).T.astype(np.int64)
            graphs.append(dict(positions=x[None],weights=support_weights(x)[None],center=0,
                times=np.array([0.],np.float32),species=1,log_scale=np.log(plan['scale']/REFERENCE_RADIUS),
                physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False,edges=edges))
    if model is None:
        model=StructuralMACE().cuda().eval();model.load_state_dict(parent_state(plan),strict=True)
        compile_encoder(model,move(collate(graphs[:1],'mace'),'cuda'),'bf16')
    values=[]
    with torch.no_grad():
        for start in range(0,len(graphs),128):
            batch=move(collate(graphs[start:start+128],'mace'),'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(batch).float()
            if z.shape[-1]!=128 or not torch.isfinite(z).all():raise ValueError(f'Invalid future embeddings: {sid}')
            values.append(z.cpu().numpy())
    result[len(mapping):]=np.concatenate(values).reshape(-1,len(ids),128)
    # Check the same compiled producer against an existing cached anchor graph.
    from src.research.crystallization_transfer.data import graph
    a={p.stem:np.load(p,mmap_mode='r') for p in original.glob('*.npy')}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        replay=model(move(collate([graph(a,int(mapping[-1,0,0]),plan['scale'])],'mace'),'cuda')).float().cpu().numpy()[0]
    np.testing.assert_allclose(replay,result[len(mapping)-1,0],atol=1e-4,rtol=1e-4)
    np.save(folder/'center.building.npy',result,allow_pickle=False)
    (folder/'center.building.npy').replace(folder/'center.npy')
    save_json(receipt,dict(identity=plan['identity'],checkpoint_sha256=plan['checkpoint_sha256'],
        sha256=file_hash(folder/'center.npy'),frames=frame_ids.tolist(),atom_ids=ids.tolist(),
        replay_max_abs_error=float(np.max(abs(replay-result[len(mapping)-1,0])))))
    return model


class ResidentPaths:
    """Source timelines are stored once; overlapping windows are indexed on device."""
    def __init__(self,plan,spec,device='cuda'):
        self.corpus=Corpus(plan);self.plan=plan;self.spec=spec;self.device=torch.device(device)
        self.full_training_groups={s:ids.copy() for s,ids in self.corpus.groups.items()}
        self.full_training_indices=list(self.corpus.splits['train'])
        self.sources=plan['sources'];self.lookup={s['id']:i for i,s in enumerate(self.sources)}
        features=[];geometry=[];states=[];onsets=[]
        for source in self.sources:
            sid=source['id'];a=self.corpus.arrays[sid]
            z=np.load(resolve_path(plan['config']['future_cache'])/str(sid)/'center.npy')
            if z.shape!=(199,16,128):raise ValueError(f'Unexpected center timeline {sid}: {z.shape}')
            features.append(np.asarray(a['features'][a['mapping'],:128]))
            geometry.append(np.asarray(a['relative']))
            state=np.concatenate((z,a['packet'][:,::4].transpose(1,0,2)[:199],
                a['order'][:,::4].transpose(1,0,2)[:199],
                np.isin(a['labels'][:,::4],[1,2,3]).T[:199,:,None]),-1).astype(np.float32)
            if state.shape!=(199,16,STATE_DIM) or not np.isfinite(state).all():raise ValueError(f'Invalid state timeline {sid}')
            states.append(state);onsets.append(np.array(a['onset']))
        self.features=torch.as_tensor(np.stack(features),device=device)
        self.geometry=torch.as_tensor(np.stack(geometry),device=device)
        self.states=torch.as_tensor(np.stack(states),device=device)
        self.onsets=torch.as_tensor(np.stack(onsets),device=device)
        self.rows=torch.tensor([[self.lookup[s],plan['anchors'][a],c,t] for s,a,c,t in self.corpus.rows],dtype=torch.long,device=device)
        self.offsets=torch.tensor({0:[0],3:[-4,0],12:[-16,-4,0],48:[-64,-16,-4,0]}[spec['history_ps']],device=device)
        self.future=torch.arange(1,STEPS+1,device=device)*4
        self.mean=torch.zeros(STATE_DIM,device=device);self.scale=torch.ones(STATE_DIM,device=device)
        # Equal source weights and each source's natural eligible-window distribution.
        mean=torch.zeros(STATE_DIM,dtype=torch.float64,device=device);second=mean.clone()
        for ids in self.corpus.groups.values():
            sm=mean.clone().zero_();ss=sm.clone()
            for start in range(0,len(ids),512):
                y=self.targets(ids[start:start+512],normalize=False)['state'].double()
                sm+=y.sum((0,1));ss+=y.square().sum((0,1))
            mean+=sm/(len(ids)*STEPS*len(self.corpus.groups));second+=ss/(len(ids)*STEPS*len(self.corpus.groups))
        self.mean=mean.float();self.scale=(second-mean.square()).clamp_min(1e-12).sqrt().float()

    def observed(self,indices):
        s,a,c,temp=self.rows[indices].unbind(-1);frame=(a[:,None]+self.offsets)//4
        nodes=1 if self.spec['radius_A']==0 else 7
        f=self.features[s[:,None],frame,c[:,None],:nodes]
        r=self.geometry[s[:,None],frame,c[:,None],:nodes]
        dt=(self.offsets*.75).expand(len(s),nodes,-1).transpose(1,2)
        g=torch.cat((r,dt[...,None]),-1)
        temps=torch.tensor([400,450,500,510,520],device=self.device)
        condition=torch.cat(((temp[:,None]==temps).float(),(a*.75/600)[:,None],(a*.75/600).square()[:,None]),-1)
        result=dict(features=f.flatten(1,2),geometry=g.flatten(1,2),condition=condition)
        if self.spec.get('motion_input',False):
            # Packet speeds/radial velocities (80:112) and motion moments (117:128).
            columns=torch.tensor([*range(208,240),*range(245,256)],device=self.device)
            current=self.states[s,a//4,c]
            result['motion']=((current-self.mean)/self.scale)[:,columns]
        return result

    def targets(self,indices,normalize=True):
        s,a,c,_=self.rows[indices].unbind(-1)
        y=self.states[s[:,None],(a[:,None]+self.future)//4,c[:,None]]
        delay=self.onsets[s,c]-a
        # Dense onset labels retain the original 0.75 ps resolution and 96 ps censoring.
        event=(delay-1).clamp(max=128)
        occurred=torch.arange(1,129,device=self.device)[None]>=delay[:,None]
        return dict(state=(y-self.mean)/self.scale if normalize else y,event=event,
            occurred=occurred.reshape(-1,STEPS,SUBSTEPS).float())

    def event_bins(self,indices):
        s,a,c,_=self.rows[indices].unbind(-1)
        return (self.onsets[s,c]-a-1).clamp(max=128)

    def baseline(self,indices):
        s,a,c,_=self.rows[indices].unbind(-1)
        return ((self.states[s,a//4,c]-self.mean)/self.scale)[:,None].expand(-1,STEPS,-1)

    def set_context(self,spec):
        self.spec=spec
        self.offsets=torch.tensor({0:[0],3:[-4,0],12:[-16,-4,0],48:[-64,-16,-4,0]}[spec['history_ps']],device=self.device)
