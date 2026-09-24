"""Real archived-data checks: paired joins, causal inputs and eight short fits."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,digest,file_hash
from src.research.crystallization_paths.runtime import fit
from src.research.crystallization_transfer.runtime import setup
from .reuse import extract_source
from .relaxed import load_encoder
from .reuse_data import ReusePaths


def verify(plan):
    setup();root=resolve_path(plan['reuse_config']['output'])/'technical'
    if not json.loads((root/'precision.json').read_text())['passed']:raise ValueError('Precision gate has not passed')
    smoke=copy.deepcopy(plan);sources=[]
    for role in ('train','selection','calibration','test'):
        candidates=[s for s in plan['sources'] if s.get('validation_role',s['split'])==role]
        sources.append(candidates[0])
    smoke['sources']=sources
    smoke['reuse_config']=dict(smoke['reuse_config'],context_cache=str(root/'smoke-cache'),output=str(root/'smoke'))
    smoke['config']=dict(smoke['config'],output=str(root/'smoke'),batch_size=8,selection_samples=2,evaluation_samples=2,selection_per_source=8)
    smoke['observed_histories']={}
    for source in sources:
        sid=str(source['id']);a=min(plan['observed_histories'][sid],key=int)
        smoke['observed_histories'][sid]={a:plan['observed_histories'][sid][a]}
    smoke['structured_identity']=digest(dict(original=plan['structured_identity'],smoke=sources,histories=smoke['observed_histories']))
    smoke['identity']=digest(smoke)
    save_json(root/'smoke-plan.json',smoke)
    model=load_encoder(plan,'cuda');seconds=[]
    for source in sources:
        begin=time.monotonic();extract_source(smoke,source,model,time.time()+3600);seconds.append(time.monotonic()-begin)
    del model;torch.cuda.empty_cache()
    specs=json.loads((root/'queue.json').read_text());datasets={};checks=[]
    for domain in ('relaxed','observed'):
        spec=next(s for s in specs if s['observation_domain']==domain)
        datasets[domain]=ReusePaths(smoke,spec)
    a,b=datasets['relaxed'],datasets['observed']
    assert a.corpus.rows==b.corpus.rows
    torch.testing.assert_close(a.states,b.states,rtol=0,atol=0)
    torch.testing.assert_close(a.mean,b.mean,rtol=0,atol=0)
    torch.testing.assert_close(a.history_offsets,b.history_offsets,rtol=0,atol=0)
    # Changing every future target must not change any observed input.
    ids=list(range(min(4,len(a.rows))));before={k:v.clone() for k,v in a.observed(ids).items()}
    saved=a.states.clone();a.states.fill_(12345)
    for k,v in a.observed(ids).items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    a.states.copy_(saved);del saved
    for original in specs:
        spec=copy.deepcopy(original);spec.update(patience=0,minimum_epochs=1)
        spec['training']=dict(spec['training'],sources=1,epochs=1,budget='epochs')
        spec['name']='smoke-'+spec['name']
        data=datasets[spec['observation_domain']]
        ok=fit(smoke,spec,data,time.time()+3600)
        if not ok:raise RuntimeError('Smoke fit did not complete')
        folder=root/'smoke/technical/runs'/spec['name'];metrics=json.loads((folder/'metrics.json').read_text())
        checks.append(dict(name=spec['name'],passed=True,brier=metrics['dense_integrated_brier']))
    save_json(root/'validation.json',dict(passed=True,checks=checks,paired_targets_identical=True,causal_inputs=True,
        extraction_source_seconds=seconds,meaning='One-epoch four-source pipeline smoke, not scientific comparison results.'))
    print(json.dumps(dict(passed=True,checks=len(checks),extraction_source_seconds=seconds)),flush=True)
