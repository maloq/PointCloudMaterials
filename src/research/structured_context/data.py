"""Deduplicated resident feature timelines; shared physical/event labels."""
import json
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash
from src.research.crystallization_transfer.data import Corpus
from src.research.context_night.context import ContextPaths,LOCAL_COLUMNS
from src.research.crystallization_paths.data import STATE_DIM,STEPS


class StructuredPaths(ContextPaths):
    def __init__(self,plan,spec,device='cuda'):
        self.corpus=Corpus(plan);self.plan=plan;self.spec=spec;self.device=torch.device(device)
        self.full_training_groups={s:ids.copy() for s,ids in self.corpus.groups.items()}
        self.full_training_indices=list(self.corpus.splits['train'])
        self.sources=plan['sources'];self.lookup={s['id']:i for i,s in enumerate(self.sources)}
        cache=resolve_path(plan['structured_config']['context_cache']);name=spec['encoder']
        features=[];geometry=[];states=[];onsets=[];information=[]
        original=json.loads(resolve_path(plan['config']['reuse_plan']).read_text())
        assay=resolve_path(original['config']['assay_cache'])
        for source in self.sources:
            sid=source['id'];folder=cache/str(sid);a=self.corpus.arrays[sid]
            receipt=json.loads((folder/'complete.json').read_text())
            if receipt['identity']!=plan['structured_identity']:raise ValueError(f'Structured cache mismatch: {sid}')
            values={}
            for field in ('relative',f'{name}_features',f'{name}_center'):
                path=folder/f'{field}.npy'
                if file_hash(path)!=receipt['files'][path.name]:raise ValueError(f'Corrupt structured features: {path}')
                values[field]=np.load(path)
            features.append(values[f'{name}_features']);geometry.append(values['relative'])
            state=np.concatenate((values[f'{name}_center'],a['packet'][:,::4].transpose(1,0,2)[:199],
                a['order'][:,::4].transpose(1,0,2)[:199],np.isin(a['labels'][:,::4],[1,2,3]).T[:199,:,None]),-1).astype(np.float32)
            if state.shape!=(199,16,STATE_DIM) or not np.isfinite(state).all():raise ValueError(f'Invalid {name} timeline {sid}')
            states.append(state);onsets.append(np.array(a['onset']))
            path=assay/source['shard']
            if file_hash(path)!=source['shard_sha256']:raise ValueError('Assay changed')
            with np.load(path) as p:
                local=np.concatenate((p['packet'],p['order']),-1)[...,LOCAL_COLUMNS]
                information.append(np.concatenate((local,p['shell'][...,[0,1,6,7]]),-1)[:,0:665:4].transpose(1,0,2))
        self.features=torch.as_tensor(np.stack(features),device=device)
        self.geometry=torch.as_tensor(np.stack(geometry),device=device)
        self.states=torch.as_tensor(np.stack(states),device=device)
        self.onsets=torch.as_tensor(np.stack(onsets),device=device)
        self.information=torch.tensor(np.stack(information),device=device)
        self.rows=torch.tensor([[self.lookup[s],plan['anchors'][a],c,t] for s,a,c,t in self.corpus.rows],dtype=torch.long,device=device)
        self.future=torch.arange(1,STEPS+1,device=device)*4
        self.extra_encoder=torch.zeros((len(self.rows),128),device=device);self.extra_identity=None
        self.mean=torch.zeros(STATE_DIM,device=device);self.scale=torch.ones(STATE_DIM,device=device)
        mean=torch.zeros(STATE_DIM,dtype=torch.float64,device=device);second=mean.clone()
        for ids in self.corpus.groups.values():
            sm=mean.clone().zero_();ss=sm.clone()
            for start in range(0,len(ids),512):
                y=self.targets(ids[start:start+512],normalize=False)['state'].double()
                sm+=y.sum((0,1));ss+=y.square().sum((0,1))
            mean+=sm/(len(ids)*STEPS*len(self.corpus.groups));second+=ss/(len(ids)*STEPS*len(self.corpus.groups))
        self.mean=mean.float();self.scale=(second-mean.square()).clamp_min(1e-12).sqrt().float()
        self.calibrate_information();self.set_context(spec)

    def observed(self,indices):
        s,a,c,temp=self.rows[indices].unbind(-1);frame=(a[:,None]+self.offsets)//4
        f=self.features[s[:,None],frame,c[:,None]];r=self.geometry[s[:,None],frame,c[:,None]]
        dt=(self.offsets*.75).expand(len(s),25,-1).transpose(1,2)
        geometry=torch.cat((r,dt[...,None]),-1)
        temps=torch.tensor([400,450,500,510,520],device=self.device)
        condition=torch.cat(((temp[:,None]==temps).float(),(a*.75/600)[:,None],(a*.75/600).square()[:,None]),-1)
        return dict(features=f.flatten(1,2),geometry=geometry.flatten(1,2),condition=condition,information=self.raw_information(indices))
