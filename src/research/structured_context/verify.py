"""Real-encoder replay, layout tests, and gradients for all queued forecasters."""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
import torch
from src.project_runtime.paths import resolve_path,dataset_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.data.structural_pretraining.batches import collate,move
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.research.crystallization_paths.runtime import make_model,fit
from src.research.crystallization_transfer.runtime import setup
from .geometry import stencil
from .extract import encoders,observations,frame_input,infer,source as extract_source
from .data import StructuredPaths


def verify(plan):
    setup();c=plan['structured_config'];root=resolve_path(c['output'])/'technical'
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_structured_context.py'],check=True)
    models=encoders(plan)
    # Directly replay the exact six-static-frame analysis, not an approximate new adapter.
    static=resolve_path(c['gatr_static_analysis'])/'technical/analysis_inference_cache.npz'
    with np.load(static) as a:
        coords=a['coords'];z=a['inv_latents']
        # The static producer concatenates snapshots in protocol order; temporal
        # anchor_frame_indices is intentionally empty for this static dataset.
        first=np.array([0,1000,50000])
        centers=coords[first];reference=z[first]
    points=np.load(dataset_path('Al')/'inherent_configurations_off/166ps.npy').astype(float);tree=cKDTree(points)
    distance,atoms=tree.query(centers)
    np.testing.assert_allclose(distance,0,atol=1e-5)
    samples=observations(points,tree,atoms,plan['scale'],'gatr')
    encoded=infer(models['gatr'],samples,'gatr',3)
    np.testing.assert_allclose(encoded,reference,atol=2e-6,rtol=2e-5)
    print('Exact requested GATr static replay:',float(abs(encoded-reference).max()),flush=True)
    # One independent training source, four observed times, four tracked centers.
    source=next(s for s in plan['sources'] if s['split']=='train');cache=resolve_path(plan['config']['cache'])/str(source['id'])
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    ids=np.searchsorted(raw.atom_ids,source['center_atom_ids'][:4]);q=stencil(c['shell_radii_A'])
    values={name:[] for name in models};geometry=[];timings={name:[] for name in models}
    for frame in (0,48,60,64):
        item=frame_input(raw,frame,ids,q,plan['scale'],c['max_query_offset_A'])
        geometry.append(item['relative'])
        for name,model in models.items():
            torch.cuda.synchronize();begin=time.monotonic()
            output=infer(model,item['graphs'][name],name,c[f'{name}_extraction_batch'])
            torch.cuda.synchronize();timings[name].append(dict(seconds=time.monotonic()-begin,graphs=len(output)))
            values[name].append(output[item['mapping']])
    old=np.load(resolve_path(plan['config']['future_cache'])/str(source['id'])/'center.npy')
    mace=np.stack(values['mace'],1)
    np.testing.assert_allclose(mace[:,:,0],old[[0,12,15,16],:4].transpose(1,0,2),atol=1e-4,rtol=1e-4)
    queries=np.stack(geometry,1);dt=np.broadcast_to(np.array([-48.,-12.,-3.,0.])[None,:,None,None],(4,4,25,1))
    geo=torch.tensor(np.concatenate((queries,dt),-1).reshape(4,100,4),device='cuda',dtype=torch.float32)
    a={k:np.load(cache/f'{k}.npy') for k in ('packet','order','labels','onset')}
    future=np.arange(68,196,4);physical=np.concatenate((a['packet'][:4,future],a['order'][:4,future],np.isin(a['labels'][:4,future],[1,2,3])[...,None]),-1)
    target_z={'mace':old[17:49,:4].transpose(1,0,2),'gatr':[]}
    for frame in future:
        box=(raw.box_high[frame]-raw.box_low[frame]).astype(float);points=np.mod(raw.positions[frame].astype(float),box);tree=cKDTree(points,boxsize=box)
        samples=observations(points,tree,ids,plan['scale'],'gatr',box)
        target_z['gatr'].append(infer(models['gatr'],samples,'gatr',4))
    target_z['gatr']=np.stack(target_z['gatr'],1)
    results=[]
    for spec in json.loads((root/'queue.json').read_text()):
        name=spec['encoder'];features=torch.tensor(np.stack(values[name],1).reshape(4,100,128),device='cuda')
        obs=dict(features=features,geometry=geo,condition=torch.zeros(4,7,device='cuda'),information=torch.zeros(4,504,device='cuda'))
        y=torch.tensor(np.concatenate((target_z[name],physical),-1),device='cuda')
        mean=y.mean((0,1));scale=y.std((0,1),unbiased=False).clamp_min(.01)
        target=(y-mean)/scale;delay=torch.tensor(a['onset'][:4]-64,device='cuda');event=(delay-1).clamp(max=128)
        outcome=(torch.arange(1,129,device='cuda')[None]>=delay[:,None]).reshape(4,32,4).float()
        spec=dict(spec,training_event_cdf=np.linspace(.001,.5,128).tolist());model=make_model(spec).cuda()
        x,w=model.context.inputs(features,geo);model.context.normalization.calibrate(x,w)
        model.target_mean.copy_(mean);model.target_scale.copy_(scale)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        loss_value=[]
        for step in range(4):
            model.train();loss=model.loss(obs,dict(state=target,event=event,occurred=outcome,present=target[:,0]),0.).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            optimizer.step();loss_value.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():path,cdf=model.forecast(obs,samples=4,diffusion_steps=4)
        if not torch.isfinite(path).all() or torch.any(cdf[:,1:]<cdf[:,:-1]-1e-6):raise ValueError(f'Invalid free rollout: {spec["name"]}')
        results.append(dict(name=spec['name'],loss=loss_value,gradient_norm=float(norm),path_shape=list(path.shape)))
        print('Real-batch gradients and open-loop forecast passed:',spec['name'],flush=True)
    save_json(root/'validation.json',dict(passed=True,unit_tests=3,gatr_static_replay_max_error=float(abs(encoded-reference).max()),
        original_gatr_analysis=str(static),source=source['id'],timings=timings,smoke_fits=results,
        meaning='Execution/identity checks, not scientific fit quality or convergence'))


def verify_pipeline(plan):
    """Tiny disposable fits exercise actual joins, checkpoint selection and exports.

    One real source per role, retaining the original roles. No result from these
    technical checks selects production settings or contributes a research score.
    Their extracted feature timelines are exact production caches and are reused.
    """
    import copy
    setup();models=encoders(plan);sources=[]
    for role in ('train','selection','calibration','test'):
        item=next(s for s in plan['sources'] if s.get('validation_role',s['split'])==role)
        sources.append(item)
        print('Preparing pipeline-check source',item['id'],role,flush=True)
        if not extract_source(plan,item,models,time.time()+3600):raise RuntimeError('Preflight extraction interrupted')
    del models;torch.cuda.empty_cache()
    small=copy.deepcopy(plan);small['sources']=sources
    small['config'].update(output=plan['structured_config']['output']+'/technical/pipeline-check',
        selection_per_source=4,selection_samples=2,evaluation_samples=2,batch_size=16)
    small['identity']=digest(small)
    root=resolve_path(plan['structured_config']['output'])/'technical'
    save_json(root/'pipeline-check/technical/plan.json',small)
    checked=[]
    for reference in json.loads((root/'queue.json').read_text()):
        if reference['method']!='direct':continue
        spec=copy.deepcopy(reference);spec.update(name=reference['encoder']+'-execution-check')
        spec['training'].update(sources=1,window_fraction=.002,epochs=1)
        data=StructuredPaths(small,spec)
        # Input center and own-backbone future timeline must agree at every origin.
        ids=data.corpus.splits['train'][:8];s,a,c,_=data.rows[ids].unbind(-1)
        torch.testing.assert_close(data.observed(ids)['features'][:,-25],data.states[s,a//4,c,:128])
        assert fit(small,spec,data,time.time()+3600)
        checked.append(spec['name']);del data;torch.cuda.empty_cache()
        print('Full fit/checkpoint/metric pipeline passed:',spec['name'],flush=True)
    path=root/'validation.json';receipt=json.loads(path.read_text());receipt['pipeline_checks']=checked
    receipt['compiled_mace_full_source_replays']=[s['id'] for s in sources];save_json(path,receipt)
    from .figure import render
    render(plan,sources[0])
