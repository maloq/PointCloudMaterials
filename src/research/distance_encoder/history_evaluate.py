"""Prepare actual tracked-atom MD histories and score unchanged spatial scan paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.fixed_cohort.dataset import read_release
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.crystallization_origin import ancestry
from src.research.spatial_approach.evaluate import csv_rows
from src.research.spatial_distance.confidence import tables, RADII
from src.research.spatial_distance.train import distance_metrics
from src.research.spatial_distance.model import log_likelihood, cdf, capped_mean, capped_median
from src.research.encoder_context.geometry import graph
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.supervised_onset.tracking import update_training_summary
from .history import HistoryDistanceEncoder
from .model import DistanceEncoder


def inputs(config):
    fixed, plan = read_release(config['fixed_dataset']['root'])
    if plan['identity'] != config['fixed_dataset']['identity']:
        raise ValueError('Wrong fixed Al64 release')
    parent = resolve_path(config['history_evaluation']['parent_run'])
    audit_path = resolve_path(config['labels']['native_config'])
    binding = dict(fixed_identity=plan['identity'], population_sha256=sha(fixed/'benchmark/population.npz'),
        parent_identity_sha256=sha(parent/'technical/identity.json'),
        offsets_ps=config['history']['offsets_ps'],
        native_audit_config_sha256=sha(audit_path), producer_sha256=sha(__file__))
    return fixed, plan, parent, json.loads(audit_path.read_text()), digest(binding)


def prepare_source(config_path, sid):
    c = json.loads(Path(config_path).read_text())
    fixed, plan, parent, audit_config, identity = inputs(c)
    item = next(p for p in plan['sources'] if p['id'] == sid)
    if item['role'] not in ('selection', 'calibration', 'test'):
        raise ValueError('Evaluation histories only use declared held-out roles')
    dest = resolve_path(c['history_evaluation']['geometry_cache'])/str(sid)
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest/'complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or any(sha(dest/n) != h for n,h in saved['files'].items()):
            raise ValueError(f'Changed history evaluation cache: {sid}')
        return saved
    with np.load(fixed/'benchmark/population.npz') as a:
        rows = np.flatnonzero(a['source'] == sid)
        pop = {k:a[k][rows] for k in ('source','role','atom','frame','sample_id','legacy_row')}
    origin = parent/'technical/sources'/str(sid)
    old = json.loads((origin/'complete.json').read_text())
    for name, checksum in old['files'].items():
        if sha(origin/name) != checksum:
            raise ValueError(f'Changed scan/label producer: {origin/name}')
    records = json.loads((origin/'paths.json').read_text())['paths']
    with np.load(origin/'labels.npz') as a:
        np.testing.assert_array_equal(a['rows'], rows)
        labels = {k:a[k] for k in ('distance','visible_local','ptm_local')}
    folder = resolve_path(audit_config['output'])/'technical/sources'/str(sid)
    if sha(folder/'graph.npz') != json.loads((folder/'graph-complete.json').read_text())['sha256']:
        raise ValueError(f'Changed crystal component graph: {sid}')
    ptm = json.loads((folder/'ptm-complete.json').read_text())
    with np.load(folder/'graph.npz') as a:
        g = {k:a[k] for k in a.files}
    events, roots, _ = ancestry.establish(g, audit_config['lineage']['thresholds'][0])
    access = ancestry.GeometryAccess(audit_config, item, folder, g)
    output = {k:v for k,v in pop.items()}
    output['distance'] = labels['distance']
    offsets = np.asarray(c['history']['offsets_ps'])
    frame_offsets = np.rint(offsets/.75).astype(int)
    if offsets[-1]!=0 or np.any(np.diff(offsets)<=0):raise ValueError('Expected strictly causal, ordered offsets ending at zero')
    np.testing.assert_allclose(frame_offsets*.75,offsets,rtol=0,atol=1e-8)
    frames = len(offsets)
    output['positions'] = np.empty((len(rows),frames,80,3), np.float32)
    output['visible_frames'] = np.empty((len(rows),frames), bool)
    output['ptm_frames'] = np.empty((len(rows),frames), bool)
    verified = set()
    for frame in np.unique(pop['frame']):
        frame = int(frame); take = np.flatnonzero(pop['frame'] == frame)
        paths = [r for r in records if r['frame'] == frame]
        arrays = []
        for record in paths:
            with np.load(origin/record['file']) as a:
                arrays.append({k:a[k] for k in ('atom','distance','travel_A','positions','inverse','visible_local','ptm_local')})
        atoms = np.unique(np.concatenate([pop['atom'][take], *[a['atom'] for a in arrays]]))
        atom_rows = np.searchsorted(access.raw.atom_ids, atoms)
        np.testing.assert_array_equal(access.raw.atom_ids[atom_rows], atoms)
        xyz, reference_visible, ptm_visible = [], [], []
        for delta in frame_offsets:
            past = frame+int(delta)
            if past < 0:
                raise ValueError(f'Fixed evaluation row lacks declared MD history: {sid}/{frame}, offsets={offsets}')
            chunk = audit_config['ptm']['chunk_frames']; start = past//chunk*chunk
            name = f'ptm-{start:04d}-{min(start+chunk,item["frame_count"]):04d}.npz'
            if name not in verified:
                if sha(folder/name) != ptm['files'][name]:
                    raise ValueError(f'Changed PTM labels: {folder/name}')
                verified.add(name)
            points, box, dense = access.frame(past)
            tree = cKDTree(points, boxsize=box)
            neighbors = tree.query(points[atom_rows], k=80, workers=1)[1]
            np.testing.assert_array_equal(neighbors[:,0], atom_rows)
            p = points[neighbors]-points[atom_rows,None]
            p -= box*np.rint(p/box)
            valid = np.linalg.norm(p,axis=-1)<8.
            nodes = np.flatnonzero((g['frame']==past)&(g['size']>=64))
            known = [n for n in nodes if any(events[r-1]['confirmation_frame']<=past for r in roots[n])]
            solid = np.isin(dense,known) if known else np.zeros(len(points),bool)
            lab = access.labels[past-access.chunk_start]
            xyz.append(p.astype(np.float32))
            reference_visible.append((solid[neighbors]&valid).any(1))
            ptm_visible.append((np.isin(lab[neighbors],[1,2,3])&valid).any(1))
            if delta==0:
                distances = cKDTree(points[solid],boxsize=box).query(points[atom_rows])[0] if solid.any() else np.full(len(atoms),np.inf)
        xyz = np.stack(xyz,1); visible = np.stack(reference_visible,1); crystalline = np.stack(ptm_visible,1)
        indices = np.searchsorted(atoms,pop['atom'][take])
        np.testing.assert_allclose(distances[indices],labels['distance'][take],rtol=2e-6,atol=2e-5)
        np.testing.assert_array_equal(visible[indices,-1],labels['visible_local'][take])
        np.testing.assert_array_equal(crystalline[indices,-1],labels['ptm_local'][take])
        output['positions'][take] = xyz[indices]
        output['visible_frames'][take] = visible[indices]
        output['ptm_frames'][take] = crystalline[indices]
        for record,a in zip(paths,arrays):
            indices = np.searchsorted(atoms,a['atom'])
            np.testing.assert_allclose(xyz[indices,-1],a['positions'][a['inverse'][:,0]],rtol=0,atol=1e-6)
            np.testing.assert_allclose(distances[indices],a['distance'],rtol=2e-6,atol=2e-5)
            np.savez(dest/record['file'],positions=xyz[indices],visible_frames=visible[indices],ptm_frames=crystalline[indices],
                **{k:a[k] for k in ('atom','distance','travel_A')})
    np.savez(dest/'fixed.npz',**output)
    write_json(dest/'paths.json',records)
    saved = dict(identity=identity, source=sid, role=item['role'], rows=len(rows), paths=len(records),
        source_manifest_sha256=item['manifest_sha256'], graph_sha256=sha(folder/'graph.npz'),
        ptm_receipt_sha256=sha(folder/'ptm-complete.json'), files={p.name:sha(p) for p in dest.glob('*.npz')})
    saved['files']['paths.json'] = sha(dest/'paths.json')
    write_json(receipt,saved)
    return saved


def prepare(config_path):
    c=json.loads(Path(config_path).read_text());_,plan,_,_,identity=inputs(c)
    root=resolve_path(c['history_evaluation']['geometry_cache']);root.mkdir(parents=True,exist_ok=True)
    items=[s for s in plan['sources'] if s['role'] in ('selection','calibration','test')]
    results=[]
    with ProcessPoolExecutor(max_workers=c['history_evaluation']['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(prepare_source,str(Path(config_path).resolve()),s['id']) for s in items]
        for job in as_completed(jobs):
            result=job.result();results.append(result)
            progress=dict(completed=len(results),total=len(items),source=result['source'])
            write_json(root/'state.json',dict(state='preparing',**progress));print(json.dumps(progress),flush=True)
    write_json(root/'manifest.json',dict(identity=identity,sources=sorted(results,key=lambda s:s['source'])))
    write_json(root/'state.json',dict(state='complete',sources=len(results)))


@torch.no_grad()
def predict(model, x, distance, c, baseline=False):
    chunks=[];size=c['history_evaluation']['batch_size']
    for start in range(0,len(x),size):
        coords=torch.as_tensor(x[start:start+size],device='cuda',dtype=torch.float32)
        if baseline or c['history']['mode']=='repeated_current':coords=coords[:,-1:]
        target=torch.as_tensor(distance[start:start+size],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            parts=model(graph(coords.reshape(-1,80,3),model.encoder))
        chunks.append(dict(log_likelihood=log_likelihood(parts,target,64).cpu().numpy(),
            cdf=cdf(parts,target.new_tensor(RADII)).cpu().numpy(),
            mean_A=capped_mean(parts,64).cpu().numpy(),median_A=capped_median(parts,64).cpu().numpy()))
    return {k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]}


def run(config_path, checkpoint, output, baseline=False):
    c=json.loads(Path(config_path).read_text());root=result_folders(resolve_path(output));tech=root/'technical'
    checkpoint=resolve_path(checkpoint);saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if saved['epoch']<12:raise ValueError('Only completed 12-epoch checkpoints enter final history evaluation')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    model=(DistanceEncoder(saved['encoder_config']) if baseline else HistoryDistanceEncoder(saved['encoder_config'],c['history'])).cuda()
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    cache=resolve_path(c['history_evaluation']['geometry_cache']);manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['identity']!=inputs(c)[-1]:raise ValueError('History evaluation geometry contract changed')
    binding=dict(config=c,checkpoint_sha256=sha(checkpoint),geometry_manifest_sha256=sha(cache/'manifest.json'),baseline=baseline)
    identity=digest(binding);complete=tech/'complete.json'
    if complete.exists():
        prior=json.loads(complete.read_text())
        if prior['identity']!=identity or any(sha(tech/k)!=v for k,v in prior['files'].items()):raise ValueError('Changed completed evaluation')
        return
    write_json(tech/'identity.json',binding)
    mode='current' if baseline or c['history']['mode']=='repeated_current' else 'history'
    write_json(tech/'prediction-context.json',dict(encoder='shared geometry-only trainable during fitting; frozen for evaluation',
        history_offsets_ps=c['history']['offsets_ps'] if mode=='history' else [0],same_tracked_atom=True,
        predictor='ordered local embeddings; no surrounding patch features or explicit time/temperature inputs',
        visibility='union of local observations actually supplied to the model',baseline=baseline))
    fixed_parts=[];scan_parts=[];records=[];compiled=False
    for record in manifest['sources']:
        folder=cache/str(record['source'])
        for name,checksum in record['files'].items():
            if sha(folder/name)!=checksum:raise ValueError(f'Changed MD history geometry: {folder/name}')
        paths=json.loads((folder/'paths.json').read_text())
        for path in [None,*paths]:
            with np.load(folder/('fixed.npz' if path is None else path['file'])) as a:
                x=a['positions'];distance=a['distance']
                if not compiled:
                    example=x[:c['history_evaluation']['batch_size']]
                    if mode=='current':example=example[:,-1:]
                    compile_spatial_encoder(model.encoder,graph(torch.as_tensor(example.reshape(-1,80,3),device='cuda'),model.encoder))
                    compiled=True
                values=predict(model,x,distance,c,baseline)
                values.update({k:a[k] for k in a.files if k not in ('positions','visible_frames','ptm_frames')})
                for key,field in [('visible','visible_frames'),('ptm','ptm_frames')]:
                    actual=a[field].any(1) if mode=='history' else a[field][:,-1]
                    values[key+'_local']=actual;values[key+'_context']=actual
                    values[key+'_current']=a[field][:,-1]
                if path is None:fixed_parts.append(values)
                else:scan_parts.append(values);records.append(path)
        print(json.dumps(dict(stage='history-evaluation',source=record['source'],sources_done=len(fixed_parts))),flush=True)
    fixed={k:np.concatenate([r[k] for r in fixed_parts]) for k in fixed_parts[0]}
    scans={k:np.concatenate([r[k] for r in scan_parts]) for k in scan_parts[0]}
    scans['offsets']=np.r_[0,np.cumsum([len(r['distance']) for r in scan_parts])]
    np.savez_compressed(tech/'predictions.npz',**fixed);np.savez_compressed(tech/'path-predictions.npz',**scans)
    write_json(tech/'paths.json',records)
    analysis=root/'analyses/front-v1';snapshot_metric_docs(analysis,c.get('metric_family','distance_encoder_history'))
    metrics=[]
    for role in ('selection','calibration','test'):
        ids=np.flatnonzero(fixed['role']==role)
        metrics.append(dict(population='fixed_at_risk',role=role,**distance_metrics({k:fixed[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},fixed['distance'][ids],fixed['source'][ids],64)))
    ids=np.concatenate([np.arange(scans['offsets'][i],scans['offsets'][i+1]) for i,r in enumerate(records) if r['role']=='test'])
    sources=np.concatenate([np.full(r['rows'],r['source']) for r in records if r['role']=='test'])
    metrics.append(dict(population='controlled_scan',role='test',**distance_metrics({k:scans[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},scans['distance'][ids],sources,64)))
    alarms,paths,reliability=tables('mace_local',fixed,scans,records,fixed['cdf'],scans['cdf'])
    name='CD-MACE128 snapshot' if baseline else c['encoder_name']
    for rows in (alarms,paths,reliability):
        for row in rows:row['model']=name
    for filename,rows in [('distance',metrics),('alarms',alarms),('paths',paths),('confidence-reliability',reliability)]:
        csv_rows(analysis/'tables'/f'{filename}.csv',rows)
    write_json(tech/'metrics.json',metrics)
    if not baseline:
        training=resolve_path(c['output']);identity_training=json.loads((training/'technical/complete.json').read_text())['identity']
        tracking=SimpleNamespace(config=c,root=training,technical=training/'technical',identity=identity_training)
        fields={f"front/{row['population']}/{row['role']}/{k}":v
            for row in metrics for k,v in row.items() if k not in ('population','role')}
        for row in alarms:
            if row['radius_A']==20 and row['consecutive']==2:
                for k in ('conditional_median_warning_A','misses','recall_at12A','away_false_alarm_rate'):
                    fields[f"front/sustained_within20A/p{row['threshold']}/{k}"]=row[k]
        update_training_summary(tracking,'joint-mace-distance',fields,evaluation='front')
    write_json(complete,dict(identity=identity,files={n:sha(tech/n) for n in ('predictions.npz','path-predictions.npz','paths.json','metrics.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--baseline',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.config)
    else:run(a.config,a.checkpoint,a.output,a.baseline)
