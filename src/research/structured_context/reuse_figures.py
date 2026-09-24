"""Matched archived-data figures, fixed-event comparisons and real paired clouds."""
import argparse
import json
from pathlib import Path
import resource
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve,average_precision_score
from src.project_runtime.paths import resolve_path,dataset_path
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.research.crystallization_transfer.data import Corpus
from src.research.local_predictability.metrics import source_weights
from src.research.local_predictability.metrics import cumulative_risk
from src.research.context_night.metrics import short_logits
from src.research.crystallization_paths.metrics import coarse_logits
from src.research.crystallization_paths.runtime import make_model
from src.research.structured_context.reuse_data import restrict_corpus,ReusePaths
from src.experiment_runner.metric_docs import write_metric_table
from .event_offsets import evaluate as event_evaluate
from .reuse import archived_positions,cold_graphs
from src.data.trajectories.shooting import ShootingBinaryTrajectory

DOMAINS=('observed','relaxed')
METHODS=('direct','ar_mse','mixture','diffusion')


def matched_events(corpus,test_indices,offsets,width,seed):
    """Same onset/control at every lead bin, selected without forecast scores."""
    lookup={(s,corpus.plan['anchors'][a],c):r for r,i in enumerate(test_indices) for s,a,c,t in [corpus.rows[i]]}
    origins={s:sorted({a for ss,a,c in lookup if ss==s}) for s,a,c in lookup}
    records=[];rows=[];leads=[]
    for sid in sorted(origins):
        onsets=corpus.arrays[sid]['onset']
        for center,onset in enumerate(onsets):
            if onset>=801:continue
            selected=[]
            for offset in offsets:
                candidates=[a for a in origins[sid] if offset<=(onset-a)*.75<offset+width and (sid,a,center) in lookup]
                if not candidates:break
                selected.append(max(candidates))
            if len(selected)!=len(offsets):continue
            if len(set(selected))!=len(selected):raise ValueError('Lead bins selected duplicate observations')
            controls=[c for c,future in enumerate(onsets) if future>onset and all((sid,a,c) in lookup for a in selected)]
            if not controls:continue
            rng=np.random.default_rng(np.random.SeedSequence([seed,sid,center]));control=int(rng.choice(controls))
            rows.append([[lookup[(sid,a,center)],lookup[(sid,a,control)]] for a in selected])
            leads.append((int(onset)-np.array(selected))*.75)
            records.append(dict(source=int(sid),center=center,control_center=control,onset_frame=int(onset),origins_frames=selected))
    if not rows:raise ValueError('No common archived event/control cohort')
    return dict(records=records,rows=np.array(rows),leads=np.array(leads),sources=np.array([r['source'] for r in records]))


def selected_examples(matched,seed,count=3):
    rng=np.random.default_rng(seed);chosen=[];seen=set()
    for i in rng.permutation(len(matched['records'])):
        sid=matched['records'][i]['source']
        if sid in seen:continue
        chosen.append(int(i));seen.add(sid)
        if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError('Insufficient distinct-source examples')
    return chosen


def replay(config,root,plan,specs,predictions,matched,examples):
    """Exact saved checkpoints; physical forecasts only, no model fitting."""
    selected_rows=matched['rows'][examples,0,0]
    ids=predictions['relaxed-ar_mse']['test_indices'][selected_rows].tolist()
    output={};reference=None;maps={};rng=np.random.default_rng(config['seed'])
    for domain in DOMAINS:
        spec=specs[f'{domain}-ar_mse'];data=ReusePaths(plan,spec,config['device'])
        if reference is None:
            reference=dict(ids=ids,cases=[dict(matched['records'][i],lead_ps=float(matched['leads'][i,0])) for i in examples])
            sampled=[]
            for source in data.sources:
                role=source.get('validation_role',source['split'])
                if role not in ('train','test'):continue
                group=[i for i in data.corpus.splits[role] if data.corpus.rows[i][0]==source['id']]
                sampled.extend(rng.choice(group,min(config['umap_per_source'],len(group)),replace=False).tolist())
            sampled=np.array(sampled);meta=np.array([data.corpus.rows[i] for i in sampled]);maps.update(indices=sampled,source=meta[:,0],temperature=meta[:,3],
                train=np.isin(sampled,data.corpus.splits['train']),event12=(data.event_bins(sampled)<16).cpu().numpy())
            truth=data.targets(ids,normalize=False)['state'].cpu().numpy();output['truth']=truth
        obs=data.observed(ids)
        for method in ('ar_mse','mixture'):
            key=f'{domain}-{method}';spec=specs[key];path=resolve_path(config['input'])/'technical/runs'/spec['name']/'best.pt'
            saved=torch.load(path,map_location=data.device,weights_only=False)
            if saved['plan_identity']!=plan['identity']:raise ValueError('Forecast checkpoint belongs to another plan')
            torch.testing.assert_close(saved['mean'],data.mean,rtol=0,atol=0)
            torch.testing.assert_close(saved['scale'],data.scale,rtol=0,atol=0)
            model=make_model(saved['spec']).to(data.device).eval();model.load_state_dict(saved['model'],strict=True)
            torch.manual_seed(config['seed'])
            with torch.no_grad():paths,cdf=model.forecast(obs,samples=config['samples'])
            original=predictions[key]['test_cdf'][selected_rows]
            error=float(np.max(np.abs(cdf.cpu().numpy()-original)))
            if error>2e-5:raise ValueError(f'Archived forecast replay differs: {key}: {error}')
            output[key]=(paths*data.scale+data.mean).cpu().numpy()
            reference[key]=dict(checkpoint_sha256=file_hash(path),cdf_replay_max_error=error)
            del model
        encoded=[]
        for start in range(0,len(sampled),128):encoded.append(data.observed(sampled[start:start+128])['features'][:,-25].cpu().numpy())
        maps[domain]=np.concatenate(encoded)
        del data;torch.cuda.empty_cache()
    np.savez_compressed(root/'technical/trajectory-examples.npz',**output)
    save_json(root/'technical/trajectory-examples.json',reference)
    from sklearn.preprocessing import StandardScaler
    import umap
    import joblib
    for domain in DOMAINS:
        scaler=StandardScaler().fit(maps[domain][maps['train']]);z=scaler.transform(maps[domain])
        mapper=umap.UMAP(n_neighbors=30,min_dist=.15,random_state=config['seed'],n_jobs=1)
        xy=np.empty((len(z),2));xy[maps['train']]=mapper.fit_transform(z[maps['train']]);xy[~maps['train']]=mapper.transform(z[~maps['train']])
        maps[domain+'_xy']=xy
        joblib.dump(dict(scaler=scaler,mapper=mapper),root/'technical'/f'umap-{domain}.joblib')
    np.savez_compressed(root/'technical/umap.npz',**maps)


def paired_clouds(config,root,plan,corpus):
    """Three label-selected real cells; identical IDs, centered charts and view."""
    from src.research.relaxed_encoder.prepare import paired_clouds as extract_pair
    sources={s['id']:s for s in plan['sources']};chosen={}
    for key,record in sorted(plan['archive_cells'].items(),key=lambda item:(item[1]['source'],item[1]['frame'])):
        sid=record['source'];source=sources[sid]
        if source.get('validation_role',source['split'])!='test':continue
        frame=record['frame'];a=corpus.arrays[sid]
        for ci,onset in enumerate(a['onset']):
            label=int(a['labels'][ci,frame]);delay=(onset-frame)*.75
            kind=('crystalline' if label in (1,2,3) and onset<=frame else
                  'before_onset' if label==0 and 0<delay<=12 else
                  'liquid' if label==0 and delay>96 else None)
            if kind and kind not in chosen:chosen[kind]=dict(source=sid,frame=frame,center=ci,atom_id=source['center_atom_ids'][ci],raw_ptm=label,delay_ps=float(delay),record=record)
        if len(chosen)==3:break
    if len(chosen)!=3:raise ValueError(f'Insufficient real illustration classes: {chosen.keys()}')
    arrays={};metadata={}
    for kind,item in chosen.items():
        source=sources[item['source']];frame=item['frame'];raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
        if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Illustration raw source changed')
        cold,box,ids,provenance=archived_positions(plan,item['record'],source)
        np.testing.assert_array_equal(ids,raw.atom_ids)
        center=int(np.searchsorted(raw.atom_ids,item['atom_id']))
        hot_cloud,cold_cloud,neighbor=extract_pair(raw.positions[frame].astype(float),cold,box,np.array([center]))
        displacement=cold_cloud[0]-hot_cloud[0];displacement-=box*np.rint(displacement/box)
        arrays[kind+'_observed']=hot_cloud[0];arrays[kind+'_relaxed']=cold_cloud[0];arrays[kind+'_displacement']=displacement;arrays[kind+'_ids']=raw.atom_ids[neighbor[0]]
        metadata[kind]=dict(**item,provenance=provenance,relative_displacement_rms_A=float(np.sqrt(np.mean(np.sum(displacement**2,-1)))),
            meaning='Center-relative displacement between the same atom IDs; removes translation of the tracked center.')
        if kind=='before_onset':
            hot=np.mod(raw.positions[frame].astype(float),box)
            graph=cold_graphs(plan,source,frame,cold,box,raw);ci=item['center'];qids=np.searchsorted(raw.atom_ids,graph['query_atom_ids'][ci])
            h,c,nn=extract_pair(hot,cold,box,qids)
            for domain,points,clouds in [('observed',hot,h),('relaxed',cold,c)]:
                full=points-points[center];full-=box*np.rint(full/box)
                relative=points[qids]-points[center];relative-=box*np.rint(relative/box)
                arrays['context_'+domain]=full[np.linalg.norm(full,axis=1)<31].astype(np.float32)
                arrays['queries_'+domain]=relative.astype(np.float32);arrays['clouds_'+domain]=clouds
            from scipy.spatial import cKDTree
            tree=cKDTree(hot,boxsize=box);radius=8*plan['scale']/9.192189
            for slot,query in enumerate(qids):
                neighbors=tree.query_ball_point(hot[query],radius)
                x=hot[neighbors]-hot[query];x-=box*np.rint(x/box)
                arrays[f'context_observed_cloud_{slot}']=x.astype(np.float32)
            metadata['context']=dict(source=item['source'],frame=frame,atom_id=item['atom_id'],query_atom_ids=raw.atom_ids[qids].tolist(),radius_A=8*plan['scale']/9.192189)
    np.savez_compressed(root/'technical/paired-clouds.npz',**arrays);save_json(root/'technical/paired-clouds.json',metadata)


def run(config):
    root=resolve_path(config['output']);base=resolve_path(config['input'])/'technical'
    for name in ('plots','tables','technical'):(root/name).mkdir(parents=True,exist_ok=True)
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard));torch.set_num_threads(1)
    plan=json.loads((base/'plan.json').read_text());corpus=Corpus(plan);restrict_corpus(corpus,plan['observed_histories'])
    producer_files=['src/research/'+p for p in ('structured_context/model.py','structured_context/reuse_data.py',
        'crystallization_paths/model.py','crystallization_paths/refined_model.py','context_night/context.py',
        'crystallization_transfer/attention.py','crystallization_transfer/model.py')]
    for name in producer_files:
        if file_hash(Path(name))!=file_hash(base/'code'/name):raise ValueError(f'Forecast producer changed: {name}')
    specs={};predictions={};metrics={};hashes={}
    for spec in json.loads((base/'queue.json').read_text()):
        key=spec['observation_domain']+'-'+spec['method'];folder=base/'runs'/spec['name']
        if json.loads((folder/'status.json').read_text())['state']!='complete':raise ValueError(f'Incomplete fit: {key}')
        specs[key]=spec;metrics[key]=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'predictions.npz') as a:predictions[key]={k:a[k] for k in a.files}
        hashes[key]={name:file_hash(folder/name) for name in ('predictions.npz','metrics.json','best.pt')}
    reference=predictions['relaxed-ar_mse'];ids=reference['test_indices'];source=corpus.source_ids[ids];weights=source_weights(source)
    for p in predictions.values():
        np.testing.assert_array_equal(p['test_indices'],ids);np.testing.assert_array_equal(p['test_event'],reference['test_event'])
        if not np.isfinite(p['test_cdf']).all() or np.any(np.diff(p['test_cdf'],axis=1)<-1e-6):raise ValueError('Invalid archived CDF')
    for key,p in predictions.items():
        # Replay the evaluator's hazard clipping, including tied diffusion risks.
        short=cumulative_risk(torch.tensor(short_logits(p['test_cdf']))).numpy()
        coarse=cumulative_risk(torch.tensor(coarse_logits(p['test_cdf'],plan['lags']))).numpy()
        p['evaluated_risk12']=short[:,-1]
        p['evaluated_horizon_risk']=np.c_[short,coarse[:,3:]]
        ap=average_precision_score(p['test_event']<16,p['evaluated_risk12'],sample_weight=weights)
        np.testing.assert_allclose(ap,metrics[key]['short_horizon']['classification']['12.0']['average_precision'],rtol=1e-12,err_msg=key)
    matched=matched_events(corpus,ids,config['offsets_ps'],config['offset_bin_width_ps'],config['seed'])
    summary,arrays=event_evaluate(predictions,matched,config['offsets_ps'],draws=config['bootstrap_draws'],seed=config['seed'])
    examples=selected_examples(matched,config['seed'])
    save_json(root/'technical/matched-events.json',dict(records=matched['records'],examples=examples,offsets_ps=config['offsets_ps'],bin_width_ps=config['offset_bin_width_ps']))
    save_json(root/'technical/event-offset-metrics.json',summary)
    np.savez_compressed(root/'technical/event-offset-arrays.npz',rows=matched['rows'],leads=matched['leads'],sources=matched['sources'],**arrays)
    write_metric_table(summary,root,family='relaxed_reuse_figures',name='event-offset-skill')
    paired_clouds(config,root,plan,corpus)
    replay(config,root,plan,specs,predictions,matched,examples)
    from .reuse_render import render
    render(config,root,plan,corpus,predictions,metrics,matched,summary,examples)
    save_json(root/'technical/validation.json',dict(passed=True,plan_identity=plan['identity'],files=hashes,
        plot_files=[p.name for p in sorted((root/'plots').glob('*.png'))],matched_events=len(matched['records']),matched_sources=len(np.unique(matched['sources'])),
        same_test_population=True,ap_replay=True,new_simulations=0))
    print(json.dumps(dict(output=str(root),plots=len(list((root/'plots').glob('*.png'))),matched_events=len(matched['records']))),flush=True)


def render_cached(config):
    """Restyle verified analysis artifacts without UMAP fitting or GPU replay."""
    root=resolve_path(config['output']);base=resolve_path(config['input'])/'technical'
    validation=json.loads((root/'technical/validation.json').read_text())
    if not validation['passed']:raise ValueError('No completed analysis to render')
    plan=json.loads((base/'plan.json').read_text())
    if validation['plan_identity']!=plan['identity']:raise ValueError('Analysis plan changed')
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    corpus=Corpus(plan);restrict_corpus(corpus,plan['observed_histories']);predictions={};metrics={}
    for spec in json.loads((base/'queue.json').read_text()):
        key=spec['observation_domain']+'-'+spec['method'];folder=base/'runs'/spec['name']
        for name,expected in validation['files'][key].items():
            if file_hash(folder/name)!=expected:raise ValueError(f'Analysis input changed: {folder/name}')
        metrics[key]=json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'predictions.npz') as a:predictions[key]={k:a[k] for k in a.files}
        p=predictions[key];short=cumulative_risk(torch.tensor(short_logits(p['test_cdf']))).numpy()
        coarse=cumulative_risk(torch.tensor(coarse_logits(p['test_cdf'],plan['lags']))).numpy()
        p['evaluated_risk12']=short[:,-1];p['evaluated_horizon_risk']=np.c_[short,coarse[:,3:]]
    metadata=json.loads((root/'technical/matched-events.json').read_text())
    if metadata['offsets_ps']!=config['offsets_ps'] or metadata['bin_width_ps']!=config['offset_bin_width_ps']:
        raise ValueError('Cached event lead definitions differ')
    with np.load(root/'technical/event-offset-arrays.npz') as a:
        matched={k:a[k] for k in ('rows','leads','sources')}
    matched['records']=metadata['records'];summary=json.loads((root/'technical/event-offset-metrics.json').read_text())
    from .reuse_render import render
    render(config,root,plan,corpus,predictions,metrics,matched,summary,metadata['examples'])
    write_metric_table(summary,root,family='relaxed_reuse_figures',name='event-offset-skill')
    validation['rendering_files']={name:file_hash(Path(name)) for name in ('src/research/structured_context/reuse_figures.py','src/research/structured_context/reuse_render.py')}
    validation['plot_sha256']={p.name:file_hash(p) for p in (root/'plots').glob('*.png')}
    save_json(root/'technical/validation.json',validation)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--render-only',action='store_true');a=p.parse_args()
    config=json.loads(resolve_path(a.config).read_text())
    if a.render_only:render_cached(config)
    else:run(config)

if __name__=='__main__':main()
