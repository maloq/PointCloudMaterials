"""Check optimizer-step scheduling and the 0.1 ps temporal-loss gate."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from src.utils.training_utils import build_step_cosine_scheduler
from src.training_methods.pretrained_mace import objective


def main():
    torch.manual_seed(12)
    parameters=[nn.Parameter(torch.ones(1)),nn.Parameter(torch.ones(1))]
    optimizer=torch.optim.AdamW([dict(params=[parameters[0]],lr=3e-5),dict(params=[parameters[1]],lr=3e-4)])
    scheduler=build_step_cosine_scheduler(optimizer,total_steps=20,warmup_steps=2,start_factor=.05,min_lr=1e-6)
    rates=[[g['lr'] for g in optimizer.param_groups]]
    for _ in range(20):optimizer.step();scheduler.step();rates.append([g['lr'] for g in optimizer.param_groups])
    rates=np.array(rates)
    np.testing.assert_allclose(rates[0],[1.5e-6,1.5e-5]);np.testing.assert_allclose(rates[2],[3e-5,3e-4]);np.testing.assert_allclose(rates[-1],[1e-6,1e-6])
    assert (np.diff(rates[:3],axis=0)>0).all() and (np.diff(rates[2:],axis=0)<0).all()
    # Each six-example batch has one Al shooting, one Al continuation, two Mg,
    # and two Ta samples. The shooting sample has no temporal invariance target.
    m=torch.tensor([0,0,1,1,2,2]);eligible=torch.tensor([False,True,True,True,True,True])
    model=SimpleNamespace(tda=nn.Linear(256,32),forecast=nn.Linear(261,256))
    cfg=dict(spatial_weight=0.,temporal_weight=1.,tda_weight=0.,prediction_weight=0.)
    z=torch.randn(6,4,256,requires_grad=True);target=torch.randn(6,4,32);condition=torch.randn(6,5)
    loss,parts=objective(model,z,target,condition,m,cfg,temporal_mask=eligible);loss.backward()
    torch.testing.assert_close(z.grad[0,2],torch.zeros(256),rtol=0,atol=0)
    assert float(z.grad[1,2].norm())>0
    changed=z.detach().clone();changed[0,2]+=100
    _,after=objective(model,changed,target,condition,m,cfg,temporal_mask=eligible)
    torch.testing.assert_close(parts['temporal_mse'],after['temporal_mse'],rtol=0,atol=0)
    result=dict(scheduler_rates=rates.tolist(),excluded_shooting_temporal_gradient_norm=float(z.grad[0,2].norm()),included_continuation_temporal_gradient_norm=float(z.grad[1,2].norm()))
    Path('output/pretrained_mace_80_dt01_cosine_20260906/objective_schedule_checks.json').write_text(json.dumps(result,indent=2)+'\n');print('SCHEDULE_AND_TEMPORAL_GATE_PASSED',flush=True)

if __name__=='__main__':main()
