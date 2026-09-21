"""Updated frozen-encoder forecast figures, exact replay and fixed-event curves."""
import argparse
import json
import resource
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path, dataset_path
from src.research.crystallization_transfer.data import Corpus
from src.research.crystallization_paths.figures import digest, save_json, choose_examples, aggregate
from src.research.crystallization_paths.runtime import make_model
from src.research.context_night.context import LOCAL_COLUMNS, information_values
from src.experiment_runner.metric_docs import write_metric_table
from .event_offsets import cohort, evaluate


def folders(root):
    for name in ('technical', 'tables', 'plots'):
        (root/name).mkdir(parents=True, exist_ok=True)


class Replay:
    """Lazy small-batch implementation of StructuredPaths.observed/state."""
    def __init__(self, plan, corpus, encoder):
        self.plan = plan; self.corpus = corpus; self.encoder = encoder
        self.sources = {s['id']:s for s in plan['sources']}
        self.cache = resolve_path(plan['structured_config']['context_cache'])
        original = json.loads(resolve_path(plan['config']['reuse_plan']).read_text())
        self.assay = resolve_path(original['config']['assay_cache'])
        self.loaded = {}; self.states = {}; self.info = {}

    def arrays(self, sid):
        if sid not in self.loaded:
            folder = self.cache/str(sid); receipt = json.loads((folder/'complete.json').read_text())
            if receipt['identity'] != self.plan['structured_identity']:
                raise ValueError(f'Structured identity changed: {sid}')
            arrays = {}
            for name in ('relative', self.encoder+'_features', self.encoder+'_center'):
                path = folder/(name+'.npy')
                if digest(path) != receipt['files'][path.name]:
                    raise ValueError(f'Structured cache checksum changed: {path}')
                arrays[name] = np.load(path, mmap_mode='r')
            self.loaded[sid] = arrays
        return self.loaded[sid]

    def state(self, sid):
        if sid not in self.states:
            a = self.corpus.arrays[sid]; z = self.arrays(sid)[self.encoder+'_center']
            self.states[sid] = np.concatenate((z, a['packet'][:, ::4].transpose(1,0,2)[:199],
                a['order'][:, ::4].transpose(1,0,2)[:199],
                np.isin(a['labels'][:, ::4], [1,2,3]).T[:199,:,None]), -1).astype(np.float32)
        return self.states[sid]

    def observed(self, indices, spec):
        if spec['history_ps'] != 48 or spec['context_layout'] != 'cuboctahedral_v1':
            raise ValueError('This analysis requires the completed four-frame structured experiment')
        features = []; geometry = []; conditions = []; info = []
        offsets = np.array([-64,-16,-4,0])
        for index in indices:
            sid, ai, ci, temp = self.corpus.rows[index]; frame = self.plan['anchors'][ai]
            arrays = self.arrays(sid); frames = (frame+offsets)//4
            features.append(arrays[self.encoder+'_features'][frames,ci].reshape(100,128))
            r = arrays['relative'][frames,ci]; dt = np.broadcast_to(offsets[:,None,None]*.75, (4,25,1))
            geometry.append(np.concatenate((r,dt),-1).reshape(100,4))
            conditions.append([*[float(temp == t) for t in (400,450,500,510,520)], frame*.75/600, (frame*.75/600)**2])
            if sid not in self.info:
                source = self.sources[sid]; path = self.assay/source['shard']
                if digest(path) != source['shard_sha256']:
                    raise ValueError(f'Changed observed assay: {sid}')
                with np.load(path) as a:
                    np.testing.assert_array_equal(a['atom_ids'], source['center_atom_ids'])
                    local = np.concatenate((a['packet'],a['order']),-1)[...,LOCAL_COLUMNS]
                    shell = a['shell'][...,[0,1,6,7]]
                    self.info[sid] = torch.tensor(np.concatenate((local,shell),-1)[:,0:665:4].transpose(1,0,2)[None])
            info.append(torch.cat((information_values(self.info[sid],torch.tensor([[0,frame,ci,0]])),torch.zeros(1,128)),-1))
        return dict(features=torch.tensor(np.stack(features),dtype=torch.float32),
            geometry=torch.tensor(np.stack(geometry),dtype=torch.float32),
            condition=torch.tensor(conditions,dtype=torch.float32), information=torch.cat(info))


def producer_hashes(base):
    paths = ['src/research/structured_context/'+n+'.py' for n in ('model','data','geometry')]
    paths += ['src/research/crystallization_paths/'+n+'.py' for n in ('model','refined_model','runtime')]
    paths += ['src/research/crystallization_transfer/'+n+'.py' for n in ('attention','model','data')]
    paths += ['src/research/context_night/context.py']
    result = {}
    for path in paths:
        result[path] = digest(path)
        if result[path] != digest(base/'code'/path):
            raise ValueError(f'Changed forecast producer: {path}; replay frozen implementation')
    return result


def embedding_map(config, replay, cases, forecasts, root):
    from sklearn.preprocessing import StandardScaler
    import umap
    import joblib
    train = []; test = []; metadata = []; rng = np.random.default_rng(config['seed'])
    for s in replay.plan['sources']:
        role = s.get('validation_role',s['split'])
        if role not in ('train','test'):
            continue
        sid = s['id']; a = replay.corpus.arrays[sid]
        flat = np.sort(rng.choice(199*16, config['umap_per_source'], replace=False)); fi,ci = flat//16,flat%16
        z = np.array(replay.arrays(sid)[replay.encoder+'_center'][fi,ci])
        if role == 'train':
            train.append(z)
        else:
            test.append(z); metadata.extend(zip([sid]*len(fi),fi*3,ci,[s['temperature_K']]*len(fi),
                a['order'][ci,fi*4,1],np.isin(a['labels'][ci,fi*4],[1,2,3]).astype(int)))
    train = np.concatenate(train); test = np.concatenate(test)
    scaler = StandardScaler().fit(train)
    mapper = umap.UMAP(n_neighbors=config['umap_neighbors'],min_dist=config['umap_min_dist'],
        random_state=config['seed'],transform_seed=config['seed'],n_jobs=1,metric='euclidean')
    mapper.fit(scaler.transform(train))
    arrays = dict(test_xy=mapper.transform(scaler.transform(test)),metadata=np.array(metadata),
                  train_mean=scaler.mean_,train_scale=scaler.scale_)
    all_z = []; sizes = []
    for i,case in enumerate(cases):
        f = case['frame']//4; actual = replay.state(case['source'])[f-16:f+33,case['center'],:128]
        all_z.append(actual); sizes.append((f'actual_{i}',len(actual)))
        for name,value in forecasts.items():
            z = value['paths'][i].mean(0)[:,:128]; all_z.append(z); sizes.append((f'{name}_{i}',len(z)))
    xy = mapper.transform(scaler.transform(np.concatenate(all_z))); start=0
    for key,n in sizes:
        arrays[key] = xy[start:start+n]; start += n
    np.savez_compressed(root/'technical/umap.npz',**arrays)
    joblib.dump(dict(scaler=scaler,umap=mapper),root/'technical/umap.joblib')
    save_json(root/'technical/umap-method.json',dict(training_rows=len(train),test_rows=len(test),
        seed=config['seed'],neighbors=config['umap_neighbors'],min_dist=config['umap_min_dist'],
        fitting='Independent map per backbone, training sources only; identical sampled state identities'))


def context_clouds(plan, replay, case, root):
    from scipy.spatial import cKDTree
    from src.data.trajectories.shooting import ShootingBinaryTrajectory
    from src.data.structural_pretraining.support import REFERENCE_RADIUS
    from .geometry import stencil, representatives
    source = replay.sources[case['source']]; frame=case['frame']
    raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if digest(raw.root/'manifest.json') != source['manifest_sha256']:
        raise ValueError('Changed spatial illustration source')
    center_id = source['center_atom_ids'][case['center']]; center=int(np.searchsorted(raw.atom_ids,center_id))
    box = (raw.box_high[frame]-raw.box_low[frame]).astype(float)
    points = np.mod(raw.positions[frame].astype(float),box); tree=cKDTree(points,boxsize=box)
    query = stencil(plan['structured_config']['shell_radii_A'])
    ids, relative = representatives(points,center,tree,box,query,plan['structured_config']['max_query_offset_A'])
    np.testing.assert_allclose(relative,replay.arrays(source['id'])['relative'][frame//4,case['center']],atol=1e-6)
    factor=REFERENCE_RADIUS/plan['scale']; radii=dict(mace=8/factor)
    keep=tree.query_ball_point(points[center],24+max(radii.values())); full=points[keep]-points[center];full-=box*np.round(full/box)
    clouds=dict(full=full,queries=query,representatives=relative,atom_ids=raw.atom_ids[ids])
    counts={}
    for name,radius in radii.items():
        counts[name]=[]
        for slot in range(len(query)):
            keep=tree.query_ball_point(points[ids[slot]],radius); x=points[keep]-points[ids[slot]];x-=box*np.round(x/box)
            x=x[np.linalg.norm(x,axis=1)<radius];clouds[f'{name}_{slot}']=x;counts[name].append(len(x))
    np.savez_compressed(root/'technical/context-clouds.npz',**clouds)
    save_json(root/'technical/context-method.json',dict(source=source['id'],frame=frame,time_ps=frame*.75,
        tracked_atom_id=int(center_id),radii_A=radii,local_atom_counts=counts,query_slots=list(range(len(query))),
        max_assignment_offset_A=float(np.linalg.norm(relative-query,axis=1).max()),raw_manifest=str(raw.root/'manifest.json')))


def prepare(config):
    root=resolve_path(config['output']);folders(root);tech=root/'technical'
    if (tech/'prepared.json').exists():
        if json.loads((tech/'prepared.json').read_text())['config'] != config:
            raise ValueError('Analysis config changed; choose a new output')
        return
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,8192),hard),hard))
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    base=resolve_path(config['input'])/'technical'; hashes=producer_hashes(base)
    plan=json.loads((base/'plan.json').read_text());corpus=Corpus(plan)
    predictions={}; identities={}
    keys=('test_indices','test_event','test_cdf','test_path_scores','test_persistence_scores',
          'calibration_indices','calibration_event','calibration_cdf')
    for encoder in config['encoders']:
        for method in config['methods']:
            key=encoder+'-'+method;folder=base/'runs'/(key+'-symmetric-E36')
            if json.loads((folder/'status.json').read_text())['state'] != 'complete':
                raise ValueError(f'Incomplete predictor: {key}')
            with np.load(folder/'predictions.npz') as a:
                predictions[key]={k:a[k] for k in keys}
            identities[key]=dict(checkpoint=digest(folder/'best.pt'),predictions=digest(folder/'predictions.npz'))
    reference=predictions['mace-direct']
    for p in predictions.values():
        for k in ('test_indices','test_event','calibration_indices','calibration_event'):
            np.testing.assert_array_equal(reference[k],p[k])
    cases=choose_examples(corpus,reference);save_json(tech/'examples.json',cases)
    matched=cohort(corpus,reference['test_indices'],config['offsets_ps'],config['seed'])
    save_json(tech/'event-cohort.json',dict(records=matched['records'],excluded=matched['excluded'],offsets_ps=config['offsets_ps']))
    print('Fixed event cohort:',len(matched['records']),'events;',len(np.unique(matched['sources'])),'sources;',matched['excluded'],flush=True)
    # Verify label and time alignment directly against each archived dense target.
    positive=reference['test_event'][matched['rows'][:,:,0]]
    np.testing.assert_allclose((positive+1)*.75,matched['leads'])
    assert np.all(reference['test_event'][matched['rows'][:,:,1]]>positive)
    event_metrics,event_arrays=evaluate(predictions,matched,config['offsets_ps'],draws=config['bootstrap_draws'],seed=config['seed'])
    np.savez_compressed(tech/'event-offsets.npz',**event_arrays,rows=matched['rows'],leads=matched['leads'],sources=matched['sources'])
    save_json(tech/'event-metrics.json',event_metrics)
    write_metric_table(event_metrics,root,family='structured_figures',name='event_offsets')
    # Score-independent illustrative events: source medians, then three onset quantiles.
    medians=[]
    for sid in np.unique(matched['sources']):
        ids=np.flatnonzero(matched['sources']==sid)
        ids=sorted(ids,key=lambda i:(matched['records'][i]['onset_frame'],matched['records'][i]['center']))
        medians.append(ids[len(ids)//2])
    medians.sort(key=lambda i:(matched['records'][i]['onset_frame'],matched['records'][i]['source']))
    illustrated=np.asarray([medians[int(q*(len(medians)-1))] for q in (.2,.5,.8)])
    examples={key:p['test_cdf'][matched['rows'][illustrated,:,0]] for key,p in predictions.items()}
    np.savez_compressed(tech/'event-examples.npz',**examples,indices=illustrated,leads=matched['leads'][illustrated])
    checks={}
    for encoder in config['encoders']:
        out=root/encoder;folders(out);replay=Replay(plan,corpus,encoder)
        p={method:predictions[encoder+'-'+method] for method in config['methods']}
        curves,metrics=aggregate(config,corpus,p);np.savez_compressed(out/'technical/aggregate.npz',**curves)
        write_metric_table(metrics,out,family='structured_figures',name='forecast_curves')
        observed={f'state_{i}':replay.state(c['source'])[c['frame']//4-16:c['frame']//4+33,c['center']] for i,c in enumerate(cases)}
        np.savez_compressed(out/'technical/observed.npz',**observed)
        forecasts={}
        for method in config['methods']:
            key=encoder+'-'+method;folder=base/'runs'/(key+'-symmetric-E36')
            saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            model=make_model(saved['spec']).to(config['device']).eval();model.load_state_dict(saved['model'],strict=True)
            with np.load(folder/'sample-trajectories.npz') as a:
                check_ids=a['indices'][:2].tolist();target=a['target'][:2];saved_paths=a['paths'][:2].astype(np.float32)
            ids=[c['index'] for c in cases]+check_ids
            inputs={k:v.to(config['device']) for k,v in replay.observed(ids,saved['spec']).items()}
            torch.manual_seed(config['seed'])
            with torch.no_grad():
                paths,cdf=model.forecast(inputs,samples=config['samples'],diffusion_steps=saved['spec']['diffusion_steps'])
            paths=paths.cpu().numpy();cdf=cdf.cpu().numpy();mean=saved['mean'].numpy();scale=saved['scale'].numpy()
            target_replay=[]
            for index in check_ids:
                sid,ai,ci,_=corpus.rows[index];f=plan['anchors'][ai]//4
                target_replay.append((replay.state(sid)[f+1:f+33,ci]-mean)/scale)
            np.testing.assert_allclose(target_replay,target,atol=1e-6,rtol=1e-5)
            if method in ('direct','ar_mse'):
                np.testing.assert_allclose(paths[-2:],saved_paths,atol=.001,rtol=.002)
                expected=p[method]['test_cdf'][[c['test_row'] for c in cases]]
                np.testing.assert_allclose(cdf[:4],expected,atol=3e-5,rtol=1e-4)
                checks[key]=dict(cdf_max_abs_error=float(abs(cdf[:4]-expected).max()),saved_path_max_error=float(abs(paths[-2:]-saved_paths).max()))
            value=dict(paths=paths[:4]*scale+mean,cdf=p[method]['test_cdf'][[c['test_row'] for c in cases]],history_ps=np.array(48))
            forecasts[method]=value;np.savez_compressed(out/'technical'/f'{method}-examples.npz',**value)
            del model
            print('Verified/replayed',key,flush=True)
        embedding_map(config,replay,cases,forecasts,out)
        if encoder=='mace':context_clouds(plan,replay,cases[0],root)
        print('UMAP complete:',encoder,flush=True)
    save_json(tech/'replay-checks.json',checks)
    save_json(tech/'prepared.json',dict(config=config,producer_hashes=hashes,inputs=identities,
        plan_sha256=digest(base/'plan.json'),test_windows=len(reference['test_indices']),test_sources=30,
        matched_events=len(matched['records']),matched_sources=len(np.unique(matched['sources']))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('all','prepare','render'),default='all');args=parser.parse_args()
    config=json.loads(resolve_path(args.config).read_text())
    if args.stage in ('all','prepare'):prepare(config)
    if args.stage in ('all','render'):
        from .figure_render import render
        render(config)


if __name__=='__main__':main()
