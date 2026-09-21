"""Strictly observed descriptors, shared capacity and source-held-out normalization."""
import json
import numpy as np
import torch
from torch import nn
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash
from src.research.crystallization_paths.data import ResidentPaths
from src.research.crystallization_paths.refined_model import RefinedForecaster

LOCAL_COLUMNS=np.r_[0:80,112:117,128:136]
INFO_DIM=504  # current local93, three history differences279, outer shells4, new encoder128


def information_values(timelines,rows):
    s,a,c,_=rows.unbind(-1);f=a//4
    if torch.any(a%4!=0) or torch.any(a<16):raise ValueError('Observed context requires 3ps grid and >=12ps past')
    current=timelines[s,f,c]
    return torch.cat((current[:,:93],*[current[:,:93]-timelines[s,f-d,c,:93] for d in (1,2,4)],current[:,93:]),-1)


def context_mask(kind,device=None):
    mask=torch.zeros(INFO_DIM,device=device)
    if kind in ('history','both','both_new'):mask[:372]=1
    if kind in ('shells','both','both_new'):mask[372:376]=1
    if kind=='both_new':mask[376:]=1
    if kind not in ('control','shells','history','both','both_new'):raise ValueError(kind)
    return mask


class ContextPaths(ResidentPaths):
    def __init__(self,plan,spec,device='cuda'):
        super().__init__(plan,spec,device)
        assay_plan=json.loads(resolve_path(plan['config']['reuse_plan']).read_text());source_cache=resolve_path(assay_plan['config']['assay_cache']);values=[]
        for source in self.sources:
            p=source_cache/source['shard']
            if file_hash(p)!=source['shard_sha256']:raise ValueError(f'Context assay changed: {p}')
            with np.load(p) as a:
                np.testing.assert_array_equal(a['atom_ids'],source['center_atom_ids'])
                local=np.concatenate((a['packet'],a['order']),-1)[...,LOCAL_COLUMNS]
                shell=a['shell'][...,[0,1,6,7]]
                values.append(np.concatenate((local,shell),-1)[:,0:665:4].transpose(1,0,2))
        self.information=torch.tensor(np.stack(values),device=self.device)
        self.extra_encoder=torch.zeros((len(self.rows),128),device=self.device)
        self.extra_identity=None
        self.calibrate_information()

    @torch.no_grad()
    def calibrate_information(self):
        mean=torch.zeros(INFO_DIM,dtype=torch.float64,device=self.device);second=mean.clone()
        for ids in self.full_training_groups.values():
            sm=mean.clone().zero_();ss=sm.clone()
            for start in range(0,len(ids),1024):
                x=self.raw_information(ids[start:start+1024]).double();sm+=x.sum(0);ss+=x.square().sum(0)
            mean+=sm/(len(ids)*len(self.full_training_groups));second+=ss/(len(ids)*len(self.full_training_groups))
        self.information_mean=mean.float();self.information_scale=(second-mean.square()).clamp_min(1e-6).sqrt().float()

    def raw_information(self,indices):
        return torch.cat((information_values(self.information,self.rows[indices]),self.extra_encoder[indices]),-1)

    def observed(self,indices):
        result=super().observed(indices)
        result['information']=self.raw_information(indices)
        return result

    def load_encoder(self,folder,population_path):
        folder=resolve_path(folder);pop=np.load(population_path);expected=np.array([r[:3] for r in self.corpus.rows])
        np.testing.assert_array_equal(pop['rows'],expected)
        record=json.loads((folder/'record.json').read_text())
        if record['population_sha256']!=file_hash(population_path):raise ValueError('New encoder population mismatch')
        if record['protected_overlap']:raise ValueError('Encoder ancestry leakage')
        result=np.empty((len(expected),128),np.float32)
        for sid in np.unique(pop['source']):
            p=folder/'features'/f'{sid}.npy';receipt=json.loads(p.with_suffix('.json').read_text())
            if file_hash(p)!=receipt['sha256'] or receipt['checkpoint_sha256']!=record['checkpoint_sha256']:raise ValueError(f'Changed encoder extraction: {p}')
            result[pop['source']==sid]=np.load(p)
        self.extra_encoder.copy_(torch.from_numpy(result).to(self.device));self.extra_identity=record['checkpoint_sha256'];self.calibrate_information()


class ContextForecaster(RefinedForecaster):
    def __init__(self,spec):
        super().__init__(spec);w=spec['head_width']
        self.information_head=nn.Sequential(nn.Linear(INFO_DIM,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w))
        nn.init.zeros_(self.information_head[-1].weight);nn.init.zeros_(self.information_head[-1].bias)
        self.register_buffer('information_mean',torch.zeros(INFO_DIM));self.register_buffer('information_scale',torch.ones(INFO_DIM))
        self.register_buffer('information_mask',context_mask(spec['information_context']))

    def encode(self,observed):
        inputs={k:v for k,v in observed.items() if k!='information'}
        if self.spec.get('remove_encoder_context',False):
            # Mean imputation makes every normalized learned feature exactly
            # zero, while retaining observed positions/time and head capacity.
            inputs['features']=self.context.normalization.mean.expand_as(inputs['features'])
        base=super().encode(inputs)
        x=(observed['information']-self.information_mean)/self.information_scale
        return base+self.information_head(x*self.information_mask)

    def initialize_information(self,data):
        self.information_mean.copy_(data.information_mean);self.information_scale.copy_(data.information_scale)
        path=resolve_path(self.spec['initial_checkpoint']);saved=torch.load(path,map_location=data.device,weights_only=False)
        if file_hash(path)!=self.spec['initial_checkpoint_sha256']:raise ValueError('Path warm checkpoint changed')
        result=self.load_state_dict(saved['model'],strict=False)
        expected={k for k in self.state_dict() if k.startswith('information_')}
        if set(result.missing_keys)!=expected or result.unexpected_keys:raise ValueError(f'Path warm state mismatch: {result}')
