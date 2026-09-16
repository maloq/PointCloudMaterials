"""Source-held-out local information, stability, and velocity interventions."""
from pathlib import Path

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_table
from src.experiment_runner.registry import sha256,write_json
from .train import GROUPS,encode,setup


def evaluate_checkpoint(config,args):
    model,heads,clouds,target,old,meta,norm=setup(config,args)
    root=Path(config['output']);run=root/'technical'/args.variant
    checkpoint=torch.load(run/'best.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(checkpoint['encoder_state'],strict=True);heads.load_state_dict(checkpoint['head_state'],strict=True)
    model.eval();heads.eval();rng=np.random.default_rng(config['seed']+73)
    test=np.array([i for i,m in enumerate(meta) if m['split']=='test'])
    train=np.array([i for i,m in enumerate(meta) if m['split']=='train'])
    # Estimate distance scales using training states only, independently for each
    # named block. Zero blocks in the coordinates-only control are undefined.
    reference=rng.choice(train,min(512,len(train)),replace=False)
    ztrain=encode(config,model,[clouds[2*i] for i in reference],args.device).cpu().numpy()
    rows=np.ravel(np.c_[2*test,2*test+1]);selected=[clouds[i] for i in rows]
    z=encode(config,model,selected,args.device)
    with torch.no_grad(): prediction=heads(z).cpu().numpy()
    expected=target[rows].cpu().numpy();errors=(prediction-expected)**2
    zz=z.cpu().numpy();scales={};stability={}
    for name,section in [('structure',slice(0,256)),('activity',slice(256,288)),('flow',slice(288,304))]:
        scale=float(2*np.var(ztrain[:,section],axis=0).sum());scales[name]=scale
        increments=np.sum((zz[::2,section]-zz[1::2,section])**2,axis=1)
        stability[name]=dict(mean_squared_increment=float(increments.mean()),
            training_mean_squared_independent_distance=scale,
            normalized_temporal_change=float(increments.mean()/scale) if scale>1e-12 else None)
    source=[]
    for source_id in sorted({meta[i]['source_id'] for i in test}):
        local=np.flatnonzero([meta[i]['source_id']==source_id for i in test])
        both=np.ravel(np.c_[2*local,2*local+1])
        source.append(dict(source_id=source_id,**{name:float(errors[both,section].mean()) for name,section in GROUPS.items()}))
    physical={name:float(np.mean([r[name] for r in source])) for name in GROUPS}
    # Measured velocities are altered while positions, atom identities, and the
    # current-time targets remain fixed. Shuffling preserves each patch's speeds.
    interventions={};anchors=[clouds[2*i] for i in test]
    true=target[2*test].cpu().numpy()
    for name in ('zero','shuffle','reverse','float16_velocity_roundtrip'):
        altered=[]
        for x,v in anchors:
            vv={'zero':lambda:np.zeros_like(v),'shuffle':lambda:v[rng.permutation(len(v))],
                'reverse':lambda:-v,'float16_velocity_roundtrip':lambda:v.astype(np.float16).astype(np.float32)}[name]()
            altered.append((x,vv))
        za=encode(config,model,altered,args.device)
        with torch.no_grad():pa=heads(za).cpu().numpy()
        expected_altered=true.copy()
        if name=='reverse':expected_altered[:,166:]*=-1
        interventions[name]=dict(
            structural_embedding_relative_change=float((za[:,:256]-z[::2,:256]).norm()/z[::2,:256].norm()),
            motion_even_mse=float(np.mean((pa[:,160:166]-expected_altered[:,160:166])**2)),
            motion_odd_mse=float(np.mean((pa[:,166:]-expected_altered[:,166:])**2)),
            activity_mean_squared_embedding_change=float((za[:,256:288]-z[::2,256:288]).square().sum(1).mean()),
            flow_mean_squared_embedding_change=float((za[:,288:]-z[::2,288:]).square().sum(1).mean()))
    # Bootstrap independent preparation lineages, never atom rows.
    bootstrap={}
    sampled=rng.integers(0,len(source),size=(2000,len(source)))
    for name in GROUPS:
        values=np.array([r[name] for r in source]);interval=np.quantile(values[sampled].mean(1),[.025,.975])
        bootstrap[name]=dict(mean=float(values.mean()),source_bootstrap_95_percent_interval=interval.tolist())
    report=dict(protocol='mace_local_phase_space_v1',variant=args.variant,selected_epoch=checkpoint['epoch'],
        test_independent_sources=len(source),test_local_pairs=len(test),
        physical_normalized_MSE=physical,source_uncertainty=bootstrap,
        temporal_stability=stability,velocity_interventions=interventions,
        caveat='Instantaneous local phase-space encoder; stability and information are evaluated separately by block. No forecast or cluster-separation objective.')
    np.savez(run/'test-predictions.npz',embedding=zz,target=expected,prediction=prediction,pair_ids=test,
             training_reference_embeddings=ztrain,training_reference_pair_ids=reference)
    write_json(run/'evaluation.json',report);write_json(run/'per-source-errors.json',source)
    write_metric_table(report,root,family='mace_velocity',name=args.variant)
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    write_json(run/'status.json',dict(state='complete',selected_epoch=checkpoint['epoch'],
        completed_epochs=last['epoch'],checkpoint_sha256=sha256(run/'best.pt')))
    print('EVALUATION',args.variant,physical,stability,flush=True)
