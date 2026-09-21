"""Train-only linear order/angular heads on the same exported local MACE state."""
import numpy as np
import torch
from torch import nn
from src.training_methods.neighborhood_jepa.regularization.model import Model as BaseModel
from src.training_methods.neighborhood_jepa.regularization.objective import Objective as BaseObjective
from src.training_methods.neighborhood_jepa.regularization.runtime import evaluate as base_evaluate


class Model(BaseModel):
    def __init__(self,channels,spec,seed):
        super().__init__(channels,spec,seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed+4)
            self.linear_order=nn.Linear(128,8);self.linear_angular=nn.Linear(128,16)

    def initialize_information(self,saved):
        result=self.load_state_dict(saved['model'],strict=False)
        expected={k for k in self.state_dict() if k.startswith(('linear_order.','linear_angular.'))}
        if set(result.missing_keys)!=expected or result.unexpected_keys:raise ValueError(f'Information transfer mismatch: {result}')


class Objective(BaseObjective):
    def forward(self,model,encoded,target):
        _,terms=super().forward(model,encoded,target)
        z=encoded.reshape(len(target['group']),len(self.plan.views),-1)
        center=z[:,[self.plan.slot(t,0) for t in (1,2)],:128]
        order=((model.linear_order(center)-(target['order']-self.order_mean)/self.order_std).square()).mean()
        angular=((model.linear_angular(center)-(target['physical'][...,64:80]-self.physical_mean[64:80])/self.physical_std[64:80]).square()).mean()
        terms.update(linear_order=self.spec['linear_order_weight']*order,linear_angular=self.spec['linear_angular_weight']*angular)
        self.diagnostics.update(linear_order_unweighted=float(order.detach()),linear_angular_unweighted=float(angular.detach()))
        return sum(terms.values()),terms


@torch.no_grad()
def evaluate(model,objective,data,config,baselines):
    metrics,a=base_evaluate(model,objective,data,config,baselines)
    z=torch.tensor(a['invariant'],device='cuda');order=(torch.tensor(a['order_target'],device='cuda')-objective.order_mean)/objective.order_std
    values=torch.stack(((model.linear_order(z)-order).square().mean(-1),
        (model.linear_angular(z)-torch.tensor(a['physical_target'][:,0,64:80],device='cuda')).square().mean(-1)),-1).cpu().numpy()
    mean=np.mean([values[a['sources']==s].mean(0) for s in np.unique(a['sources'])],0)
    metrics.update(linear_order=float(mean[0]),linear_angular=float(mean[1]))
    # Identical checkpoint criterion in controls and treatments, independent of
    # the strength or presence of a train-only linear auxiliary head.
    metrics['selection_score']=metrics['physical']+.25*metrics['tda']+.25*metrics['order']
    return metrics,a
