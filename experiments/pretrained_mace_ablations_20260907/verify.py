"""Check queue dependencies, matched configs, loss removal and frozen probes."""
import json
import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import time
import numpy as np
import torch
from torch import nn
from src.training_methods.pretrained_mace import objective,spread
from src.training_methods.pretrained_mace_queue import predecessor_ready,process_identity,run_command
from src.training_methods import pretrained_mace_queue as queue
from src.analysis.pretrained_mace_ablation import ridge_score
from src.data_utils.pretrained_mace import Quadruplets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='experiments/pretrained_mace_ablations_20260907/plan.json');args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    configs=[json.loads(Path(r['config']).read_text()) for r in plan['runs']]
    baseline=configs[0]
    variable={'output','checkpoint_directory','analysis_config','wandb'}
    for item,cfg in zip(plan['runs'],configs):
        assert 'warm_start' not in cfg
        for k in baseline:
            if k not in variable and k!=item['removed_weight']:assert baseline[k]==cfg[k],(item['name'],k)
        if item['removed_weight']:assert cfg[item['removed_weight']]==0
    data=Quadruplets(baseline)
    assert data.epoch_steps(baseline['batch_size'])*baseline['epochs']==plan['expected_steps']
    torch.manual_seed(3)
    model=SimpleNamespace(tda=nn.Linear(256,32),forecast=nn.Linear(261,256))
    z=torch.randn(12,4,256,requires_grad=True);target=torch.randn(12,4,32);c=torch.randn(12,5);m=torch.tensor([0,0,1,1,2,2]*2);mask=torch.tensor([False,True,True,True,True,True]*2)
    full,parts=objective(model,z,target,c,m,baseline,mask)
    a,s,t,f=z.unbind(1)
    terms={'spatial_weight':25*parts['spatial_mse']+.5*(spread(a,m)+spread(s,m)),
           'temporal_weight':25*parts['temporal_mse']+.5*(spread(a[mask],m[mask])+spread(t[mask],m[mask])),
           'tda_weight':parts['tda_mse'],'prediction_weight':parts['forecast_mse']}
    for item,cfg in zip(plan['runs'][1:],configs[1:]):
        ablated,_=objective(model,z,target,c,m,cfg,mask);key=item['removed_weight']
        torch.testing.assert_close(full-ablated,baseline[key]*terms[key])
    with tempfile.TemporaryDirectory() as temp:
        state=Path(temp)/'status.json';dep=dict(status=str(state),pid=os.getpid(),process_identity=process_identity(os.getpid()))
        state.write_text('{"state":"complete"}')
        assert not predecessor_ready(dep) # Complete status alone cannot overlap a live predecessor.
        dep['process_identity']='finished_process'
        assert predecessor_ready(dep)
        for value in ('failed','training'):
            state.write_text(json.dumps(dict(state=value)))
            with TestCase().assertRaises(RuntimeError):predecessor_ready(dep)
        with TestCase().assertRaisesRegex(RuntimeError,'exited 7'):
            run_command([sys.executable,'-c','raise SystemExit(7)'],Path(temp)/'child.log',float('inf'))
    rng=np.random.default_rng(4);x=rng.normal(size=(200,8));v=rng.normal(size=(40,8));weight=rng.normal(size=(8,4))
    metrics,_=ridge_score(x,x@weight,v,v@weight,alpha=.01)
    assert metrics['r2']>.999
    # If a later training cannot fit, analyze the completed run and leave the
    # unfinished one pending. No GPU or Slurm commands are used in this check.
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'data_summary.json').write_text('{}')
        trial=dict(plan,output=str(root/'queue'),prepared_output=str(root),estimated_training_seconds=100,estimated_probe_seconds=10,estimated_analysis_seconds=10,remove_completed_inference_cache=False)
        trial['runs']=[];trial['_path']='unused_in_mock'
        for name in ('first','second'):
            config=root/(name+'.json');config.write_text(json.dumps(dict(output=str(root/name))))
            trial['runs'].append(dict(name=name,config=str(config)))
        commands=[]
        def command(argv,log,end):
            commands.append(argv)
            if '--stage' in argv and argv[-1]=='train':
                (root/'first/training_summary.json').write_text(json.dumps(dict(steps=plan['expected_steps'],partial_epoch=False)))
        with patch.object(queue,'predecessor_ready',return_value=True),patch.object(queue,'deadline',side_effect=[time.time()+1000,time.time()-1,time.time()+1000]),patch.object(queue,'run_command',side_effect=command):
            queue.run(trial)
        state=json.loads((root/'queue/status.json').read_text())
        assert state['state']=='pending_allocation_time' and state['jobs']=={'first':'complete','second':'pending'}
        assert sum('--stage' in cmd and cmd[-1]=='analysis' for cmd in commands)==1
    result=dict(matched_updates=plan['expected_steps'],loss_removals='passed',dependency_and_failure_gate='passed',analyze_completed_when_time_short='passed',linear_probe_r2=metrics['r2'])
    Path(plan['output'],'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ABLATION_QUEUE_CHECKS_PASSED',result)


if __name__=='__main__':main()
