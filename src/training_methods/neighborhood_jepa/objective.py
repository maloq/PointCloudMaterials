"""Joint target gradients, domain-conditional SIGReg and fixed physical anchors."""
import torch
from torch import nn
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from src.training_methods.structural_pretraining.objective import block_errors,PHYSICAL_BLOCKS,TDA_BLOCKS,vicreg
from src.data.structural_pretraining.bond_order import bond_order_targets,bond_order_errors
from .model import split_tensors


class Objective(nn.Module):
    def __init__(self,manifest,spec):
        super().__init__();self.spec=spec
        for name in ['physical','tda']:
            for field in ['mean','std']:self.register_buffer(f'{name}_{field}',torch.tensor(manifest['normalization'][name][field],dtype=torch.float32))
        groups=len(manifest['groups']);self.register_buffer('z_mean',torch.zeros(groups,128));self.register_buffer('z_scale',torch.ones(groups,128))
        self.register_buffer('baseline_physical',torch.zeros(groups,85));self.register_buffer('baseline_tda',torch.zeros(groups,144))
        self.sigreg=SlicingUnivariateTest(EppsPulley(t_max=3.,n_points=17,integration='trapezoid'),num_slices=256)

    def normalized(self,z,group):
        values=[]
        for g in range(len(self.z_mean)):
            mask=group==g
            if not bool(mask.any()):continue
            if self.training:
                reference=z[mask,1:,0,:128].flatten(0,1)
                mean=reference.mean(0);scale=(reference.var(0,unbiased=False)+1e-6).sqrt()
            else:mean=self.z_mean[g];scale=self.z_scale[g]
            values.append((mask,(z[mask,:,:,:128]-mean)/scale))
        inv=torch.zeros_like(z[...,:128])
        for mask,value in values:inv[mask]=value
        return torch.cat((inv,z[...,128:]),-1)

    def forward(self,model,encoded,target):
        b=len(target['group']);k=self.spec['neighbors']+1
        raw=encoded.reshape(b,3,k,-1);z=self.normalized(raw,target['group'])
        centers=z[:,1:,0,:128].reshape(-1,128)
        physical=block_errors(model.physical(centers,target['group'].repeat_interleave(2)),(target['physical'].flatten(0,1)-self.physical_mean)/self.physical_std,PHYSICAL_BLOCKS).mean()
        tda=block_errors(model.tda(centers,target['group'].repeat_interleave(2)),(target['tda'].flatten(0,1)-self.tda_mean)/self.tda_std,TDA_BLOCKS).mean()
        bond_target=bond_order_targets(target['bonds'].flatten(0,1))
        bond=bond_order_errors(model.bond(raw[:,1:,0,128:].reshape(-1,raw.shape[-1]-128)),bond_target).mean()
        zero=encoded.sum()*0;inv_loss=zero;eq_loss=zero;future=zero
        if self.spec['prediction']!='none':
            times=[1] if self.spec['prediction']=='spatial' else [0,1,2]
            ti=[];ni=[]
            for t in times:
                for j in range(1,k):ti.append(t);ni.append(j)
            if self.spec['prediction']=='temporal':ti.append(2);ni.append(0)
            ti=torch.tensor(ti,device=z.device);ni=torch.tensor(ni,device=z.device)
            position=target['position'][:,ni];lag=target['times'][:,ti]
            predicted_inv,predicted_eq=model.predict(z[:,1,0],z[:,0,0],position,lag,target['times'][:,0])
            actual=z[:,ti,ni]
            inv_loss=(predicted_inv-actual[...,:128]).square().mean()
            eq_loss=torch.stack([(p-a).square().mean()/.01 for p,a in zip(split_tensors(predicted_eq),split_tensors(actual[...,128:]))]).mean()
        if self.spec['prediction']=='temporal':
            p=block_errors(model.physical(predicted_inv[:,-1],target['group']),(target['physical'][:,1]-self.physical_mean)/self.physical_std,PHYSICAL_BLOCKS).mean()
            h=block_errors(model.tda(predicted_inv[:,-1],target['group']),(target['tda'][:,1]-self.tda_mean)/self.tda_std,TDA_BLOCKS).mean()
            future=self.spec['future_weight']*(p+.25*h)
        regularizer=zero
        # No between-material mean separation can satisfy the within-group regularizer.
        for g in range(len(self.z_mean)):
            mask=target['group']==g
            if not bool(mask.any()):continue
            if self.spec['regularizer']=='sigreg':
                q=model.projector(z[mask,1,0,:128]);value=self.sigreg(q)
            elif self.spec['regularizer']=='vicreg':
                q0=model.projector(z[mask,1,0,:128])
                q1=model.projector(z[mask,2,0,:128] if self.spec['temporal_pair'] else z[mask,1,1,:128])
                value,_=vicreg(q0,q1)
            else:raise ValueError(self.spec['regularizer'])
            regularizer=regularizer+mask.float().mean()*value
        terms=dict(physical=physical,tda=.25*tda,bond=.1*bond,future=future,
            prediction_invariant=self.spec['prediction_weight']*inv_loss,
            prediction_equivariant=self.spec['prediction_weight']*eq_loss,
            regularizer=self.spec['sigreg_weight']*regularizer)
        return sum(terms.values()),terms
