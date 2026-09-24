"""Matched archived observation histories and identical dense future MD targets."""
import json
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash
from src.research.crystallization_transfer.data import Corpus
from src.research.crystallization_paths.data import STATE_DIM,STEPS
from src.research.context_night.context import ContextPaths,LOCAL_COLUMNS


def restrict_corpus(corpus,histories):
    """Retain original anchor indices and labels; change only row membership."""
    old_rows=corpus.rows;old_splits=corpus.splits
    keep=[i for i,(s,a,c,t) in enumerate(old_rows) if str(corpus.plan['anchors'][a]) in histories[str(s)]]
    remap={old:new for new,old in enumerate(keep)}
    corpus.rows=[old_rows[i] for i in keep]
    corpus.splits={role:[remap[i] for i in ids if i in remap] for role,ids in old_splits.items()}
    corpus.source_ids=np.array([r[0] for r in corpus.rows])
    corpus.events=corpus.events[keep]
    train=np.array(corpus.splits['train'])
    corpus.groups={s:train[corpus.source_ids[train]==s] for s in np.unique(corpus.source_ids[train])}
    if any(not ids for ids in corpus.splits.values()):raise ValueError('Archive matching emptied a source split')


class ReusePaths(ContextPaths):
    def __init__(self,plan,spec,device='cuda'):
        self.plan=plan;self.spec=spec;self.device=torch.device(device)
        self.corpus=Corpus(plan);restrict_corpus(self.corpus,plan['observed_histories'])
        self.full_training_groups={s:ids.copy() for s,ids in self.corpus.groups.items()}
        self.full_training_indices=list(self.corpus.splits['train'])
        self.sources=plan['sources'];self.lookup={s['id']:i for i,s in enumerate(self.sources)}
        cache=resolve_path(plan['reuse_config']['context_cache'])
        reference=json.loads(resolve_path(plan['reuse_config']['reference_plan']).read_text())
        oldcache=resolve_path(reference['structured_config']['context_cache'])
        assay_plan=json.loads(resolve_path(plan['config']['reuse_plan']).read_text())
        assay=resolve_path(assay_plan['config']['assay_cache'])
        features=[];geometry=[];information=[];states=[];onsets=[];lookup={};cursor=0
        for source in self.sources:
            sid=source['id'];a=self.corpus.arrays[sid];folder=cache/str(sid)
            receipt=json.loads((folder/'complete.json').read_text());path=folder/'observations.npz'
            if receipt['identity']!=plan['structured_identity'] or file_hash(path)!=receipt['sha256']:
                raise ValueError(f'Reused observations changed: {sid}')
            with np.load(path) as cold:
                frames=cold['frames'];required=sorted({f for hist in plan['observed_histories'][str(sid)].values() for f in hist})
                np.testing.assert_array_equal(frames,required)
                if spec['observation_domain']=='relaxed':
                    f=cold['features'];g=cold['relative'];info=cold['information']
                elif spec['observation_domain']=='observed':
                    old=oldcache/str(sid);oldreceipt=json.loads((old/'complete.json').read_text())
                    if oldreceipt['identity']!=reference['structured_identity']:raise ValueError('Reference context identity changed')
                    for name in ('mace_features','relative'):
                        if file_hash(old/f'{name}.npy')!=oldreceipt['files'][f'{name}.npy']:raise ValueError('Reference features changed')
                    f=np.load(old/'mace_features.npy',mmap_mode='r')[frames//4]
                    g=np.load(old/'relative.npy',mmap_mode='r')[frames//4]
                    path=assay/source['shard']
                    if file_hash(path)!=source['shard_sha256']:raise ValueError('Observed descriptor source changed')
                    with np.load(path) as original:
                        local=np.concatenate((original['packet'],original['order']),-1)[...,LOCAL_COLUMNS]
                        info=np.concatenate((local,original['shell'][...,[0,1,6,7]]),-1)[:,frames].transpose(1,0,2)
                else:raise ValueError(spec['observation_domain'])
                if f.shape!=(len(frames),16,25,128) or g.shape!=(len(frames),16,25,3) or info.shape!=(len(frames),16,97):
                    raise ValueError(f'Invalid reused observation shapes: {sid}')
                features.append(f);geometry.append(g);information.append(info)
            for i,frame in enumerate(frames):lookup[(sid,int(frame))]=cursor+i
            cursor+=len(frames)
            # A common immutable future latent space and physical MD targets.
            old=oldcache/str(sid);oldreceipt=json.loads((old/'complete.json').read_text())
            if file_hash(old/'mace_center.npy')!=oldreceipt['files']['mace_center.npy']:raise ValueError('Shared target embeddings changed')
            z=np.load(old/'mace_center.npy')
            state=np.concatenate((z,a['packet'][:,::4].transpose(1,0,2)[:199],a['order'][:,::4].transpose(1,0,2)[:199],
                np.isin(a['labels'][:,::4],[1,2,3]).T[:199,:,None]),-1).astype(np.float32)
            if state.shape!=(199,16,STATE_DIM) or not np.isfinite(state).all():raise ValueError('Invalid shared future targets')
            states.append(state);onsets.append(np.array(a['onset']))
        self.features=torch.tensor(np.concatenate(features),device=device)
        self.geometry=torch.tensor(np.concatenate(geometry),device=device)
        self.information=torch.tensor(np.concatenate(information),device=device)
        if not all(torch.isfinite(x).all() for x in (self.features,self.geometry,self.information)):raise ValueError('Nonfinite archived inputs')
        self.states=torch.tensor(np.stack(states),device=device);self.onsets=torch.tensor(np.stack(onsets),device=device)
        self.rows=torch.tensor([[self.lookup[s],plan['anchors'][a],c,t] for s,a,c,t in self.corpus.rows],device=device,dtype=torch.long)
        mapped=[];offsets=[]
        for s,a,c,t in self.corpus.rows:
            anchor=plan['anchors'][a];hist=plan['observed_histories'][str(s)][str(anchor)]
            mapped.append([lookup[(s,f)] for f in hist]);offsets.append([(f-anchor)*.75 for f in hist])
        self.history=torch.tensor(mapped,device=device,dtype=torch.long)
        self.history_offsets=torch.tensor(offsets,device=device,dtype=torch.float32)
        if torch.any(self.history_offsets>0) or torch.any(torch.diff(self.history_offsets,dim=1)<=0):raise ValueError('Noncausal observation history')
        self.future=torch.arange(1,STEPS+1,device=device)*4
        self.mean=torch.zeros(STATE_DIM,device=device);self.scale=torch.ones(STATE_DIM,device=device)
        mean=torch.zeros(STATE_DIM,device=device,dtype=torch.float64);second=mean.clone()
        for ids in self.corpus.groups.values():
            sm=mean.clone().zero_();ss=sm.clone()
            for start in range(0,len(ids),512):
                y=self.targets(ids[start:start+512],normalize=False)['state'].double()
                sm+=y.sum((0,1));ss+=y.square().sum((0,1))
            mean+=sm/(len(ids)*STEPS*len(self.corpus.groups));second+=ss/(len(ids)*STEPS*len(self.corpus.groups))
        self.mean=mean.float();self.scale=(second-mean.square()).clamp_min(1e-12).sqrt().float()
        self.calibrate_information();self.set_context(spec)

    def raw_information(self,indices):
        c=self.rows[indices,2];info=self.information[self.history[indices],c[:,None]]
        current=info[:,-1]
        return torch.cat((current[:,:93],current[:,:93]-info[:,-2,:93],current[:,:93]-info[:,-3,:93],
            torch.zeros_like(current[:,:93]),current[:,93:],current.new_zeros(len(c),128)),-1)

    def observed(self,indices):
        s,a,c,temp=self.rows[indices].unbind(-1);h=self.history[indices]
        f=self.features[h,c[:,None]];r=self.geometry[h,c[:,None]]
        dt=self.history_offsets[indices,:,None,None].expand(-1,-1,25,1)
        temps=torch.tensor([400,450,500,510,520],device=self.device)
        condition=torch.cat(((temp[:,None]==temps).float(),(a*.75/600)[:,None],(a*.75/600).square()[:,None]),-1)
        return dict(features=f.flatten(1,2),geometry=torch.cat((r,dt),-1).flatten(1,2),condition=condition,information=self.raw_information(indices))

    def set_context(self,spec):
        if spec['observed_frames']!=3 or spec['target_encoder']!='reference_mace':raise ValueError('Unexpected reuse protocol')
        self.spec=spec
