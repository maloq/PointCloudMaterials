"""Frozen-MACE continuous distance likelihood experiments; source-matched splits."""
from contextlib import ExitStack
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

from src.data.fixed_cohort.protocol import sha, write_json, digest
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.equivariant_context.data import ContextCorpus
from src.research.equivariant_context.features import FeatureExtractor, prepare_graphs
from src.research.equivariant_context.normalization import apply
from src.research.equivariant_context.cache import RetainedCache, encoder_cache
from src.research.equivariant_context.geometry_cache import GeometryCache
from src.research.equivariant_context.model import context_fields
from src.research.supervised_onset.model import Model
from src.research.supervised_onset.tracking import tracked_run
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows
from src.research.spatial_approach.train import spatial_labels
from src.research.structured_context.geometry import stencil
from .common import study
from .model import LocalDistance, ContextDistance, log_likelihood, cdf, capped_mean, capped_median
from .confidence import RADII, tables, read


def extract_augmented(s, cache, records, *, paths=False):
    saved=torch.load(s.checkpoint,map_location='cuda',weights_only=False)
    if saved['arm']['input']!='hot' or saved['config']['objective']!='hazard_nll':
        raise ValueError('Expected frozen observed likelihood-trained MACE')
    encoder=Model(saved['encoder_config'],saved['arm']).cuda()
    encoder.load_state_dict(saved['model'],strict=True);encoder.eval().requires_grad_(False)
    extractor=FeatureExtractor(encoder.encoder,'cuda',s.config['batch_size'],compile=s.config['compile'])
    output=[]
    for i,record in enumerate(records):
        geometry=(s.parent/'technical/sources' if paths else s.geometry)/str(record['source'])/record['file']
        index=record['path_id'] if paths else record['frame']
        dest=cache/f'{record["source"]}-{index}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']!=s.identity or old['geometry_sha256']!=record['sha256'] or old['sha256']!=sha(dest):
                raise ValueError(f'Augmented cache changed: {dest}')
        else:
            if sha(geometry)!=record['sha256']:raise ValueError(f'Geometry changed: {geometry}')
            with np.load(geometry) as a:
                patches=[x[np.linalg.norm(x,axis=-1)<8.] for x in a['positions']]
                values=extractor(prepare_graphs(patches,encoder.encoder.cutoff,pin_memory=True))
                values={k:v[a['inverse']] for k,v in values.items()}
                values.update({k:a[k] for k in ('actual','distance','atom','visible_local','visible_context','ptm_local','ptm_context')})
                if paths:values['travel_A']=a['travel_A']
            np.savez(dest,**values)
            write_json(receipt,dict(identity=s.identity,geometry_sha256=record['sha256'],sha256=sha(dest)))
        output.append(dict(record,features=str(dest)))
        if i%25==0:print(json.dumps(dict(stage='scan-features' if paths else 'augmentation-features',completed=i+1,total=len(records))),flush=True)
    del encoder,extractor;torch.cuda.empty_cache()
    write_json(s.technical/('scan-features.json' if paths else 'augmentation-features.json'),dict(identity=s.identity,records=output))
    return output


def extract_fixed(s,cache,geometry_root,cutoff):
    """Rebuild evictable encoder features from checksummed fixed-cohort graphs."""
    saved=torch.load(s.checkpoint,map_location='cuda',weights_only=False)
    model=Model(saved['encoder_config'],saved['arm']).cuda()
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    if model.encoder.cutoff!=cutoff:raise ValueError('Fixed graph cutoff differs from encoder')
    extractor=FeatureExtractor(model.encoder,'cuda',s.config['extraction_batch_size'],compile=s.config['compile'])
    geometry=GeometryCache(geometry_root,cutoff);geometry.domain='hot'
    receipts={}
    for item in s.plan['sources']:
        sid=item['id'];dest=cache/f'{sid}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['identity']!=s.identity or record['sha256']!=sha(dest):raise ValueError(f'Changed fixed features {sid}')
        else:
            rows=np.flatnonzero(s.pop['source']==sid);parts=[]
            for frame in np.unique(s.pop['frame'][rows]):
                ids=rows[s.pop['frame'][rows]==frame]
                path=geometry_root/f'{sid}-{frame}.npz'
                # This workflow consumes the retained hot graph contract and
                # never silently rebuilds missing graphs through a cold branch.
                if not path.is_file():raise FileNotFoundError(f'Required fixed geometry missing: {path}')
                values=geometry.frame(item,int(frame),ids,s.pop['atom'][ids],None,pin_memory=True)
                features=extractor(values['arrays'])
                parts.append(dict(rows=ids,query_atom_ids=values['atoms'],actual=values['actual'],
                                  **{k:v[values['inverse']] for k,v in features.items()}))
            arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            order=np.argsort(arrays['rows']);arrays={k:v[order] for k,v in arrays.items()}
            np.testing.assert_array_equal(arrays['rows'],rows)
            np.savez(dest,**arrays)
            record=dict(identity=s.identity,encoder_sha256=s.pointer['encoder_sha256'],sha256=sha(dest),rows=len(rows))
            write_json(receipt,record)
        receipts[str(sid)]=record
        print(json.dumps(dict(stage='fixed-features',source=sid,completed=len(receipts),total=len(s.plan['sources']))),flush=True)
    write_json(cache/'manifest.json',dict(identity=s.identity,encoder_sha256=s.pointer['encoder_sha256'],shards=receipts))
    del model,extractor;torch.cuda.empty_cache()


def build_corpus(s, augmented):
    corpus=ContextCorpus(SimpleNamespace(config=s.config,identity=s.identity),
                         'hot',('harmonic_hierarchy',),'cuda',s.features)
    parent_identity=digest(json.loads((s.parent/'technical/identity.json').read_text()))
    old=SimpleNamespace(pop=s.pop,plan=s.plan,technical=s.parent/'technical',identity=parent_identity,config=s.config)
    labels=spatial_labels(old)
    n=len(s.pop['source']);total=n+sum(r['rows'] for r in augmented)
    for k,values in list(corpus.features.items()):
        extended=values.new_empty((total,)+values.shape[1:]);extended[:n]=values
        corpus.features[k]=extended
    extras=[];offset=n
    for record in augmented:
        with np.load(record['features']) as a:
            count=len(a['distance']);ids=np.arange(offset,offset+count)
            for k in corpus.required_fields:
                corpus.features[k][ids]=torch.as_tensor(apply(a[k],corpus.scalers.get(k)),device='cuda')
            extras.append({k:a[k] for k in ('distance','visible_local','visible_context','ptm_local','ptm_context')}|
                          dict(source=np.full(count,record['source']),role=np.full(count,record['role']),frame=np.full(count,record['frame']),atom=a['atom']))
            offset+=count
    labels={k:np.concatenate([labels[k],*[e[k] for e in extras]]) for k in ('distance','visible_local','visible_context','ptm_local','ptm_context')}
    pop={k:np.concatenate([s.pop[k],*[e[k] for e in extras]]) for k in ('source','role','frame','atom')}
    corpus.features['visibility']=torch.as_tensor(np.stack([labels['visible_local'],labels['visible_context']],-1),dtype=torch.float32,device='cuda')
    corpus.distance=torch.as_tensor(labels['distance'],device='cuda')
    corpus.original_n=n;corpus.pop=pop
    corpus.split={r:np.flatnonzero(pop['role']==r) for r in ('train','selection','calibration','test')}
    corpus.weights={}
    for role in ('train','selection'):
        ids=corpus.split[role];weight=np.zeros(len(ids))
        for mask in (ids<n,ids>=n):
            weight[mask]=.5*source_weights(pop['source'][ids[mask]])
        if not np.isclose(weight.sum(),1.):raise ValueError('Invalid declared population mixture')
        corpus.weights[role]=torch.as_tensor(weight,device='cuda',dtype=torch.float32)
    return corpus,labels


def batch(model,corpus,ids):
    fields=(model.field,) if isinstance(model,LocalDistance) else context_fields(model.variant)
    return {k:corpus.features[k][ids] for k in fields}|{'nominal':corpus.nominal[None].expand(len(ids),-1,-1)}


@torch.no_grad()
def predictions(model,corpus,ids,size,cap):
    model.eval();results=[]
    for start in range(0,len(ids),size):
        part=ids[start:start+size];params=model(batch(model,corpus,part))
        results.append(dict(log_likelihood=log_likelihood(params,corpus.distance[part],cap).cpu().numpy(),
            cdf=cdf(params,corpus.distance.new_tensor(RADII)).cpu().numpy(),
            mean_A=capped_mean(params,cap).cpu().numpy(),median_A=capped_median(params,cap).cpu().numpy()))
    return {k:np.concatenate([r[k] for r in results]) for k in results[0]}


def distance_metrics(values,distance,sources,cap):
    w=source_weights(sources);target=np.minimum(distance,cap)
    return dict(rows=len(distance),sources=len(np.unique(sources)),nll=float(-w@values['log_likelihood']),
        censored_fraction=float(w@(distance>=cap)),
        capped_mean_rmse_A=float(np.sqrt(w@(values['mean_A']-target)**2)),
        capped_median_mae_A=float(w@np.abs(values['median_A']-target)),
        **{f'brier_within{r}A':float(w@(values['cdf'][:,k]-(distance<=r))**2) for k,r in enumerate(RADII)})


def fit(s,variant,corpus,labels,records):
    c=s.config;cap=c['distance_cap_A'];size=c['batch_size']
    folder=s.root/variant;tech=folder/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():
        receipt=json.loads((tech/'complete.json').read_text())
        if receipt['identity']!=s.identity or any(sha(tech/n)!=h for n,h in receipt['files'].items()):
            raise ValueError(f'Changed completed distance fit: {variant}')
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=(LocalDistance(variant) if variant in ('mace_local','visibility_only') else ContextDistance(variant,**c['predictor'])).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=1e-5)
    fit_ids,val_ids=corpus.split['train'],corpus.split['selection']
    train_w=corpus.weights['train']*len(fit_ids);val_w=corpus.weights['selection'].cpu().numpy()
    best,epoch_start,update=float('inf'),0,0
    if (tech/'last.pt').exists():
        last=torch.load(tech/'last.pt',map_location='cuda',weights_only=False)
        if last['identity']!=s.identity:raise ValueError('Resume identity differs')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
        best,epoch_start,update=last['best'],last['epoch'],last['update']
    def save(path,epoch):
        temporary=path.with_suffix('.building.pt')
        torch.save(dict(identity=s.identity,model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,
            epoch=epoch,update=update,scalers=corpus.scalers,variant=variant,config=c),temporary)
        temporary.replace(path)
    inputs=dict(encoder=dict(trainable=False,checkpoint_sha256=s.pointer['encoder_sha256'],geometry='nearest 80 atoms cropped at 8 A; 5 A edges; 2 interaction blocks; constant atom channel',
        history=False,velocity=False,conditions=[],relaxation=False,embedding=128),
        predictor=dict(fields=['visible_local','visible_context'] if variant=='visibility_only' else
            (['z_at_focal_patch'] if variant=='mace_local' else list(context_fields(variant))+['nominal_patch_offsets']),
            patch_centers=1 if variant=='mace_local' else 25,maximum_input_support_A=8 if variant=='mace_local' else 32,
            conditions=[],time_inputs=False,training_only_teacher=None,label_assisted_control=variant=='visibility_only'),
        target='current nearest confirmed reference crystal distance; right-censored at 64 A',
        population='equal mixture of fixed at-risk and uniformly sampled atom centers, equal sources within each half')
    write_json(tech/'prediction-context.json',inputs)
    tracking=SimpleNamespace(config=dict(c,wandb=dict(c['wandb'],display_name=f'Distance in Å | {variant} | spatial augmentation')),
                             root=folder,technical=tech,identity=s.identity)
    with tracked_run(tracking,variant,job_type='control' if variant=='visibility_only' else 'predictor') as run:
        run.summary.update({'objective':'zero-inflated lognormal censored distance NLL','encoder/trainable':False,
            'prediction_context':inputs,'data/train_rows':len(fit_ids),'data/validation_rows':len(val_ids),
            'model/predictor_parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint/selection_rule':'minimum mixture/source-weighted validation censored NLL after epoch 12'})
        for epoch in range(epoch_start,c['training']['epochs']):
            started=time.monotonic();model.train();order=np.random.default_rng(c['seed']+epoch).permutation(len(fit_ids));total=0.
            for start in range(0,len(order),size):
                index=order[start:start+size];ids=fit_ids[index];optimizer.zero_grad(set_to_none=True);accumulated=0.
                for begin in range(0,len(ids),c['microbatch']):
                    part=ids[begin:begin+c['microbatch']]
                    loss=-(log_likelihood(model(batch(model,corpus,part)),corpus.distance[part],cap)*train_w[index[begin:begin+len(part)]]).sum()/len(ids)
                    if not torch.isfinite(loss):raise FloatingPointError(f'{variant}: nonfinite loss at {epoch}/{update}')
                    loss.backward();accumulated+=float(loss.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();update+=1
                total+=accumulated*len(ids)
            val=predictions(model,corpus,val_ids,size,cap);nll=float(-val_w@val['log_likelihood'])
            if not np.isfinite(nll):raise FloatingPointError('Nonfinite selection likelihood')
            if epoch+1>=c['training']['minimum_selection_epoch'] and nll<best:
                best=nll;save(tech/'best.pt',epoch+1)
                run.summary['checkpoint/epoch']=epoch+1;run.summary['checkpoint/validation_distance_nll']=best
            record=dict(optimizer_update=update,**{'train/epoch':epoch+1,'train/distance_nll':total/len(fit_ids),
                'validation/distance_nll':nll,'validation/capped_distance_mae_A':float(val_w@np.abs(val['median_A']-np.minimum(labels['distance'][val_ids],cap))),
                'train/epoch_seconds':time.monotonic()-started})
            run.log(record)
            with (tech/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            save(tech/'last.pt',epoch+1);print(json.dumps(dict(variant=variant,**record)),flush=True)
        selected=torch.load(tech/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['model'])
        n=corpus.original_n
        fixed=predictions(model,corpus,np.arange(n),size,cap)|s.pop|{k:v[:n] for k,v in labels.items()}
        np.savez_compressed(tech/'predictions.npz',**fixed)
        paths=[]
        for r in records:
            with np.load(r['features']) as a:
                features={k:torch.as_tensor(apply(a[k],corpus.scalers.get(k)),device='cuda') for k in corpus.required_fields}
                features['visibility']=torch.as_tensor(np.stack([a['visible_local'],a['visible_context']],-1),dtype=torch.float32,device='cuda')
                mini=SimpleNamespace(features=features,nominal=corpus.nominal,distance=torch.as_tensor(a['distance'],device='cuda'))
                paths.append(predictions(model,mini,np.arange(len(a['distance'])),size,cap)|
                    {k:a[k] for k in ('distance','travel_A','visible_local','visible_context','ptm_local','ptm_context')})
        scans={k:np.concatenate([p[k] for p in paths]) for k in paths[0]}
        scans['offsets']=np.r_[0,np.cumsum([len(p['distance']) for p in paths])]
        np.savez_compressed(tech/'path-predictions.npz',**scans);write_json(tech/'paths.json',records)
        analysis=folder/'analyses/distance-v1';snapshot_metric_docs(analysis,'spatial_distance')
        rows=[]
        for role in ('selection','calibration','test'):
            ids=np.flatnonzero(fixed['role']==role)
            metrics=distance_metrics({k:fixed[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},fixed['distance'][ids],fixed['source'][ids],cap)
            rows.append(dict(population='fixed_at_risk',role=role,**metrics))
            if role=='test':
                for key,value in metrics.items():run.summary[f'test/{key}']=value
        ids=np.concatenate([np.arange(scans['offsets'][i],scans['offsets'][i+1]) for i,r in enumerate(records) if r['role']=='test'])
        sources=np.concatenate([np.full(r['rows'],r['source']) for r in records if r['role']=='test'])
        rows.append(dict(population='controlled_scan',role='test',**distance_metrics({k:scans[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},scans['distance'][ids],sources,cap)))
        csv_rows(analysis/'tables/distance.csv',rows)
        confidence=folder/'analyses/confidence-v1';snapshot_metric_docs(confidence,'spatial_confidence')
        a,b,d=tables(variant,fixed,scans,records,fixed['cdf'],scans['cdf'])
        for name,values in [('alarms',a),('paths',b),('confidence-reliability',d)]:csv_rows(confidence/'tables'/f'{name}.csv',values)
        write_json(tech/'metrics.json',rows)
        write_json(tech/'complete.json',dict(identity=s.identity,epoch=selected['epoch'],files={name:sha(tech/name) for name in ('best.pt','predictions.npz','path-predictions.npz','paths.json','metrics.json')}))
    del model,optimizer;torch.cuda.empty_cache()


def worker(config):
    s=study(config);torch.set_num_threads(1)
    records=json.loads((s.parent/'technical/scan-features.json').read_text())['records']
    deadline=time.time()+12*3600
    with ExitStack() as stack:
        s.features=stack.enter_context(encoder_cache(s,'fixed-spatial',s.checkpoint,deadline))
        pointer=json.loads(resolve_path(s.config['fixed_geometry_pointer']).read_text())
        geometry_root=Path(pointer['path']);metadata=json.loads((geometry_root/'entry.json').read_text())['metadata']
        retained=RetainedCache(geometry_root.parent.parent,1)
        with retained.lease(pointer['key'],deadline=deadline,metadata=metadata,shared=True):
            extract_fixed(s,s.features,geometry_root,pointer['cutoff'])
        while not (s.technical/'prepared.json').exists():
            state=s.technical/'prepare-state.json'
            if state.exists() and json.loads(state.read_text())['state']=='failed':raise RuntimeError('Preparation failed; see prepare-state.json')
            time.sleep(15)
        prepared=json.loads((s.technical/'prepared.json').read_text())
        if prepared['identity']!=s.identity:raise ValueError('Prepared identity differs')
        cache=stack.enter_context(encoder_cache(s,'uniform-spatial',s.checkpoint,deadline))
        augmented=extract_augmented(s,cache,prepared['records'])
        scan_cache=stack.enter_context(encoder_cache(s,'scan-spatial',s.checkpoint,deadline))
        records=extract_augmented(s,scan_cache,records,paths=True)
        corpus,labels=build_corpus(s,augmented)
        for variant in s.config['variants']:fit(s,variant,corpus,labels,records)
    write_json(s.technical/'worker-state.json',dict(state='complete',identity=s.identity,variants=s.config['variants']))
