"""Held-out distance/direction, spatial alarms, exported-state and noise diagnostics."""
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows
from src.research.spatial_distance.model import cdf,capped_mean,capped_median
from src.research.spatial_distance.train import distance_metrics
from src.research.spatial_distance.confidence import first_alarm
from src.research.supervised_onset.tracking import update_training_summary
from src.research.encoder_context.geometry import physical_targets
from src.research.crystallization_origin.extract import raw_source,frame_geometry
from .data import ResidentContexts,context_atoms
from .model import JointCrystalVector,objective
from .train import compile_model


RADII=(4,8,12,20,32)


@torch.no_grad()
def predict(model,batch,c,directional):
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch)
    loss=objective(out,batch,c['loss'],directional)
    weights=out['parts'][0].exp()
    mean_vector=(weights[...,None]*out['direction']).sum(1)
    length=mean_vector.norm(dim=-1)
    unit=mean_vector/length[:,None].clamp_min(1e-8)
    angle=torch.rad2deg(torch.acos((unit*batch['direction']).sum(-1).clamp(-1,1)))
    context=(weights[...,None]*out['state']).sum(1)
    result=dict(log_likelihood=-loss['distance_nll'],direction_nll=loss['direction_nll'],
        cdf=cdf(out['parts'],batch['distance'].new_tensor(RADII)),mean_A=capped_mean(out['parts'],64),
        median_A=capped_median(out['parts'],64),direction=unit,direction_resultant=length,
        angle_deg=angle,local_z=out['z'][:,0],local_v=out['v'][:,0],context_z=context)
    result['physical']=physical_targets(batch['positions'][batch['inverse'][:,0]])
    return {k:v.float().cpu().numpy() for k,v in result.items()}


def spectrum(values):
    x=np.asarray(values,dtype=np.float64);x=x-x.mean(0)
    if x.ndim==3:cov=np.einsum('ncm,ndm->cd',x,x)/(len(x)*x.shape[-1])
    else:cov=x.T@x/len(x)
    eig=np.linalg.eigvalsh(cov).clip(0,None)[::-1];total=eig.sum()
    if total<=0:return dict(trace=0.,d95=0,effective_rank=0.,participation_rank=0.)
    p=eig/total
    return dict(trace=float(total),d95=int(np.searchsorted(np.cumsum(p),.95)+1),
        effective_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),participation_rank=float(1/(p@p)))


def point_tables(values,meta,directional):
    result=[];angles=[];reliability=[]
    subsets=[('fixed_at_risk',meta['kind']==0),('uniform',meta['kind']==1),('controlled_scan',meta['kind']==2)]
    for role in ('selection','calibration','test'):
        for population,mask in subsets:
            ids=np.flatnonzero(mask&(meta['role']==role))
            if not len(ids):continue
            scores=distance_metrics({k:values[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},
                meta['distance'][ids],meta['source'][ids],64)
            result.append(dict(population=population,role=role,**scores))
            for name,take in [('all',np.ones(len(ids),bool)),('local_visible',meta['visible_local'][ids]),
                ('context_only_visible',~meta['visible_local'][ids]&meta['visible_context'][ids]),('context_clear',~meta['visible_context'][ids])]:
                rows=ids[take];w=source_weights(meta['source'][rows]) if len(rows) else np.array([])
                for k,radius in enumerate(RADII):
                    for threshold in (.5,.75,.95):
                        selected=values['cdf'][rows,k]>threshold;mass=w[selected].sum()
                        reliability.append(dict(population=population,role=role,subset=name,radius_A=radius,threshold=threshold,
                            rows=len(rows),exceedances=int(selected.sum()),coverage=float(mass),
                            precision=float(w[selected]@(meta['distance'][rows[selected]]<=radius)/mass) if mass else None,
                            mean_probability=float(w[selected]@values['cdf'][rows[selected],k]/mass) if mass else None))
            if not directional:continue
            for low,high in ((0,8),(8,16),(16,32),(32,64)):
                eligible=ids[(meta['distance'][ids]>low)&(meta['distance'][ids]<=high)]
                valid=eligible[meta['direction_valid'][eligible]]
                good=valid[values['direction_resultant'][valid]>1e-6]
                w=source_weights(meta['source'][good]) if len(good) else np.array([])
                angles.append(dict(population=population,role=role,low_A=low,high_A=high,rows=len(eligible),
                    target_valid=len(valid),prediction_defined=len(good),ambiguous_targets=int(meta['ambiguous'][eligible].sum()),
                    angular_mean_deg=float(w@values['angle_deg'][good]) if len(good) else None,
                    within30deg=float(w@(values['angle_deg'][good]<=30)) if len(good) else None,
                    direction_nll=float(w@values['direction_nll'][good]) if len(good) else None))
    return result,angles,reliability


def alarm_tables(values,meta,plan):
    rows=[]
    for k,radius in enumerate(RADII):
        for threshold in (.5,.75,.95):
            toward=[];away=[];visible=0
            for record in plan['paths']:
                if record['role']!='test':continue
                ids=np.flatnonzero(meta['path']==record['index']);ids=ids[np.argsort(meta['travel'][ids],kind='stable')]
                if len(ids)!=record['rows']:raise ValueError('Incomplete scan path')
                alarm=first_alarm(values['cdf'][ids,k],threshold,2) if len(ids)>=2 else None
                if record['kind']=='toward':
                    toward.append(np.nan if alarm is None else meta['distance'][ids[alarm]])
                    if alarm is not None:visible+=int(meta['visible_context'][ids[alarm-1:alarm+1]].any())
                else:away.append(alarm is not None)
            a=np.asarray(toward);found=np.isfinite(a)
            rows.append(dict(radius_A=radius,threshold=threshold,consecutive=2,toward_paths=len(a),
                detections=int(found.sum()),misses=int((~found).sum()),
                conditional_median_warning_A=float(np.median(a[found])) if found.any() else None,
                recall_at12A=float((a>=12).mean()),recall_at20A=float((a>=20).mean()),
                away_paths=len(away),false_alarms=int(np.sum(away)),away_false_alarm_rate=float(np.mean(away)),
                confirmed_crystal_visible_at_alarm=visible))
    return rows


def physical_probes(values,meta,eligible=None):
    train=np.flatnonzero((meta['role']=='train')&(meta['kind']<2));test=np.flatnonzero((meta['role']=='test')&(meta['kind']==0))
    if eligible is not None:train=train[eligible[train]];test=test[eligible[test]]
    # Fixed ridge diagnostic, no optimization/selection using held-out physical targets.
    rows=[]
    for field in ('local_z','context_z'):
        x=values[field][train].astype(float);mean=x.mean(0);scale=x.std(0).clip(1e-6)
        a=np.c_[(x-mean)/scale,np.ones(len(x))];target=values['physical'][train].astype(float)
        ym=target.mean(0);ys=target.std(0).clip(1e-6);y=(target-ym)/ys
        penalty=np.eye(a.shape[1])*1e-2;penalty[-1,-1]=0
        coef=np.linalg.solve(a.T@a/len(a)+penalty,a.T@y/len(a))
        pred=np.c_[(values[field][test]-mean)/scale,np.ones(len(test))]@coef
        truth=(values['physical'][test]-ym)/ys;weights=source_weights(meta['source'][test])
        mse=np.einsum('n,nk->k',weights,(pred-truth)**2)
        for i,value in enumerate(mse):rows.append(dict(embedding=field,target_index=i,standardized_test_mse=float(value),train_rows=len(train),test_rows=len(test)))
    return rows


def patch_batch(points,box,atom,shells,device):
    tree=cKDTree(points,boxsize=box);query,actual=context_atoms(points,box,tree,atom,shells)
    neighbors=tree.query(points[query],k=80,workers=1)[1]
    xyz=points[neighbors]-points[query,None];xyz-=box*np.rint(xyz/box)
    return dict(positions=torch.tensor(xyz,dtype=torch.float32,device=device),
        actual=torch.tensor(actual[None],device=device),inverse=torch.arange(25,device=device)[None])


@torch.no_grad()
def diagnostics(model,data,c,values,analysis):
    plan=json.loads((data.root/'plan.json').read_text());items={s['id']:s for s in plan['sources']}
    @lru_cache(maxsize=4)
    def raw(sid):return raw_source(items[sid])
    @lru_cache(maxsize=4)
    def frame(sid,index):return frame_geometry(raw(sid),index)
    meta=data.meta;pool=np.flatnonzero((meta['role']=='test')&(meta['kind']==0)&(meta['frame']>0))
    if c.get('observation_filter') in ('no_visible_interface','liquid_no_visible_crystal'):pool=pool[data.eligibility[pool]]
    rng=np.random.default_rng(c['seed']+91)
    ids=rng.choice(pool,min(c['evaluation']['diagnostic_rows'],len(pool)),replace=False)
    device=next(model.parameters()).device
    def embedding(batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch)
        z=out['z'][:,0];state=(out['parts'][0].exp()[...,None]*out['state']).sum(1)
        return torch.cat((z,state),-1).float().cpu().numpy()[0]
    increments=[];noise={f:[] for f in c['evaluation']['noise_rms_fractions']};nearest=[]
    scalar_rotation=[];vector_rotation=[];snapshot_rows=[]
    for i,index in enumerate(ids):
        sid=int(meta['source'][index]);anchor=int(meta['frame'][index]);r=raw(sid)
        atom=int(np.searchsorted(r.atom_ids,meta['atom'][index]))
        points,box=frame(sid,anchor)
        current=patch_batch(points,box,atom,c['dataset']['shells'],device)
        base=embedding(current)
        previous,old_box=frame(sid,anchor-1)
        increments.append(base-embedding(patch_batch(previous,old_box,atom,c['dataset']['shells'],device)))
        tree=cKDTree(points,boxsize=box);local=tree.query(points[atom],k=80)[1]
        nn=float(tree.query(points[local],k=2)[0][:,1].mean());nearest.append(nn)
        for fraction in noise:
            delta=rng.normal(size=points.shape)*(fraction*nn/np.sqrt(3))
            perturbed=np.mod(points+delta,box)
            noise[fraction].append(embedding(patch_batch(perturbed,box,atom,c['dataset']['shells'],device))-base)
        if i<16:
            rotation=np.linalg.qr(rng.normal(size=(3,3)))[0];rotation[:,0]*=np.linalg.det(rotation)
            rot=torch.tensor(rotation,dtype=torch.float32,device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a=model(current);b=model(dict(current,positions=current['positions']@rot.T,actual=current['actual']@rot.T))
            scalar_rotation.append(float((a['z']-b['z']).norm()/a['z'].norm().clamp_min(1e-8)))
            vector_rotation.append(float((a['v']@rot.T-b['v']).norm()/a['v'].norm().clamp_min(1e-8)))
        snapshot_rows.append((sid,anchor,int(meta['atom'][index])))
        if (i+1)%32==0:print(json.dumps(dict(stage='noise-and-stability',completed=i+1,total=len(ids))),flush=True)
    rows=[];increments=np.asarray(increments)
    for field,sl in [('local_z',slice(0,128)),('context_z',slice(128,256))]:
        denominator=2*spectrum(values[field][pool])['trace']
        move=increments[:,sl];rms=float(np.sqrt(np.mean(np.sum(move**2,-1))))
        rows.append(dict(embedding=field,perturbation='MD lag 0.75 ps',fraction=None,rows=len(ids),
            normalized_rms_jump=rms/np.sqrt(denominator),**{f'movement_{k}':v for k,v in spectrum(move).items()}))
        for fraction,delta in noise.items():
            move=np.asarray(delta)[:,sl];rms=float(np.sqrt(np.mean(np.sum(move**2,-1))))
            rows.append(dict(embedding=field,perturbation='coordinate noise; full context resampled',fraction=fraction,rows=len(ids),
                normalized_rms_jump=rms/np.sqrt(denominator),**{f'movement_{k}':v for k,v in spectrum(move).items()}))
    csv_rows(analysis/'tables/response.csv',rows)
    write_json(analysis/'technical/diagnostic-inputs.json',dict(rows=snapshot_rows,nearest_neighbor_A=nearest,
        population=(c['observation_filter']+' anchors; previous/noisy frames not conditioned') if c.get('observation_filter') else 'fixed test anchors',
        noise='3D RMS absolute-coordinate perturbation = fraction * mean nearest-neighbor distance over the query nearest-80 patch, before centering; all selected cells perturbed consistently',
        stability='previous raw native Al frame, exactly 0.75 ps; independent snapshot encodings',
        rotation_scalar_relative=scalar_rotation,rotation_vector_relative=vector_rotation))


def run(config_path,variant):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['output'])/variant;tech=root/'technical'
    analysis=root/'analyses/localization-v1'
    if (analysis/'technical/complete.json').exists():return
    receipt=json.loads((tech/'complete.json').read_text())
    if sha(tech/'best.pt')!=receipt['best_sha256']:raise ValueError('Selected checkpoint changed')
    saved=torch.load(tech/'best.pt',map_location='cuda',weights_only=False)
    torch.set_num_threads(1);data=ResidentContexts(c,'cuda')
    model=JointCrystalVector(saved['encoder_config'],c).cuda();model.load_state_dict(saved['model']);model.eval()
    compile_model(model,data,c);directional=variant!='distance_only'
    interface=c.get('target',{}).get('kind')=='crystal_interface_layer'
    unseen=c.get('observation_filter')=='no_visible_interface'
    liquid=c.get('observation_filter')=='liquid_no_visible_crystal'
    snapshot_metric_docs(analysis,'crystal_liquid_distance' if liquid else ('crystal_interface_unseen' if unseen else ('crystal_interface' if interface else 'crystal_vector')))
    cache=analysis/'technical/predictions.npz'
    if cache.exists():
        with np.load(cache) as a:values={k:a[k] for k in a.files}
        if sha(cache)!=json.loads((analysis/'technical/predictions.json').read_text())['sha256']:raise ValueError('Changed predictions')
    else:
        parts=[];n=len(data.meta['atom'])
        for begin in range(0,n,c['microbatch']):
            parts.append(predict(model,data.batch(np.arange(begin,min(n,begin+c['microbatch']))),c,directional))
            if begin%(32*c['microbatch'])==0:print(json.dumps(dict(stage='heldout-export',rows=begin,total=n)),flush=True)
        values={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(cache,**values)
        write_json(analysis/'technical/predictions.json',dict(sha256=sha(cache),checkpoint_sha256=receipt['best_sha256'],dataset_identity=data.identity))
    np.savez(analysis/'technical/rows.npz',**{k:v for k,v in data.meta.items() if k not in ('indices','actual')})
    if liquid:
        mask=data.eligibility
        rows,angles,reliability=point_tables({k:v[mask] for k,v in values.items()},{k:v[mask] for k,v in data.meta.items()},directional)
    else:rows,angles,reliability=point_tables(values,data.meta,directional)
    for name,table in [('distance',rows),('direction',angles),('reliability',reliability)]:
        if table:csv_rows(analysis/'tables'/f'{name}.csv',table)
    plan=json.loads((data.root/'plan.json').read_text())
    phase_rows=[]
    if interface:
        from .interface_evaluate import phase_tables,interface_alarms
        phases=phase_tables(values,data.meta,directional);phase_rows=phases[0]
        for name,table in zip(('distance_by_phase','direction_by_phase','reliability_by_phase'),phases):
            if table:csv_rows(analysis/'tables'/f'{name}.csv',table)
        csv_rows(analysis/'tables/alarms.csv',interface_alarms(values,data.meta,plan))
    elif not liquid:csv_rows(analysis/'tables/alarms.csv',alarm_tables(values,data.meta,plan))
    ranks=[]
    populations=[('all_fixed_test',(data.meta['kind']==0)&(data.meta['role']=='test')),
        ('train_population',(data.meta['kind']<2)&(data.meta['role']=='train'))]
    if interface:populations.append(('uniform_test',(data.meta['kind']==1)&(data.meta['role']=='test')))
    if unseen:
        populations.extend((name+'_unseen',mask & data.eligibility) for name,mask in list(populations))
    if liquid:
        populations=[(name+'_liquid_external_crystal',mask & data.eligibility) for name,mask in populations]
    for population,mask in populations:
        for field in ('local_z','context_z','local_v'):
            ranks.append(dict(population=population,field=field,rows=int(mask.sum()),**spectrum(values[field][mask])))
    csv_rows(analysis/'tables/rank.csv',ranks)
    csv_rows(analysis/'tables/physical-readouts.csv',physical_probes(values,data.meta,data.eligibility if unseen or liquid else None))
    diagnostics(model,data,c,values,analysis)
    tracking=SimpleNamespace(config=c,root=root,technical=tech,identity=receipt['identity'])
    fields={f"evaluation/{row['population']}/{row['role']}/{key}":value for row in rows for key,value in row.items() if key not in ('population','role')}
    if unseen:
        from .unseen import export
        fields.update(export(values,data.meta,plan,analysis,c['comparison_reference']))
    if liquid:
        from .liquid import export
        fields.update(export(values,data,plan,analysis,c))
    for row in angles:
        if row['role']=='test' and row['population']=='fixed_at_risk':
            fields[f"evaluation/direction/{row['low_A']}_{row['high_A']}A/mean_angle_deg"]=row['angular_mean_deg']
    for row in phase_rows:
        if row['role']=='test' and row['population']=='uniform':
            for key,value in row.items():
                if key not in ('phase','population','role'):
                    fields[f"evaluation/interface/{row['phase']}/{key}"]=value
    update_training_summary(tracking,'joint',fields,evaluation='localization-v1')
    write_json(analysis/'technical/complete.json',dict(checkpoint_sha256=receipt['best_sha256'],dataset_identity=data.identity,
        tables={p.name:sha(p) for p in (analysis/'tables').glob('*.csv')}))
