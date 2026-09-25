"""Current-frame spatial observations, with a frozen shared MACE per domain."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path, dataset_path
from src.research.structural_state.common import sha, digest, write_json
from src.research.structured_context.geometry import stencil, representatives
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from .model import context_fields
from .features import FeatureExtractor, prepare_graphs, prepared_frames


def population(config):
    with np.load(resolve_path(config['population_cache'])/'population.npz') as a:
        # Deliberately exclude condition, temperature, age and descriptor arrays.
        fields=('source','role','event','frame','atom')
        if 'fixed_dataset' in config:fields+=('sample_id',)
        return {k:a[k] for k in fields}


def parent_plan(config):
    return json.loads(resolve_path(config['parent_plan']).read_text())


@lru_cache(maxsize=32)
def resolved_directory(value):
    """Resolve shared archive roots once, instead of reparsing aliases per cell."""
    return resolve_path(value)


def cell_archive(parent,source,frame):
    path=resolved_directory(parent['config']['cache'])/'cells'/f'{source["id"]}-{frame}'/'complete.json'
    receipt=json.loads(path.read_text())
    if receipt['identity']!=parent['identity']:raise ValueError(f'Changed parent cell identity: {path}')
    saved=Path(receipt['archive'])
    archive=resolved_directory(str(saved.parent))/saved.name
    meta=json.loads((archive/'metadata.json').read_text())
    if (meta['source_manifest_sha256']!=source['manifest_sha256'] or meta['source_frame']!=frame
        or meta['fmax_eV_per_A']>.01
        or set(meta['potential_checksums'].values())!=set(parent['config']['potential_sha256'])):
        raise ValueError(f'Cell ancestry, potential, frame or convergence mismatch: {archive}')
    return archive


def inventory(config):
    """Read-only availability/provenance audit; no simulation or feature fitting."""
    pop=population(config);parent=parent_plan(config);sources={s['id']:s for s in parent['sources']}
    if 'fixed_dataset' in config:
        from src.data.fixed_cohort.dataset import read_release
        from src.data.fixed_cohort.protocol import assert_prediction_rows
        root,fixed=read_release(config['fixed_dataset']['root'])
        if (fixed['identity']!=config['fixed_dataset']['identity'] or
            fixed['parent_sha256']!=sha(resolve_path(config['parent_plan']))):
            raise ValueError('Context ancestry differs from the pinned Al64 release')
        with np.load(root/'benchmark/population.npz') as original:
            assert_prediction_rows(original['sample_id'],pop['sample_id'])
            for key in ('source','role','event','frame','atom'):
                np.testing.assert_array_equal(original[key],pop[key])
    def audit_source(sid):
        source=sources[int(sid)]
        root=dataset_path(source['dataset'])/source['relative_trajectory_path']
        if sha(root/'manifest.json')!=source['manifest_sha256']:raise ValueError(f'Raw source changed: {root}')
        roles=np.unique(pop['role'][pop['source']==sid])
        if len(roles)!=1 or roles[0]!=source.get('validation_role',source['split']):
            raise ValueError(f'Source split leakage or changed role: {sid}')
        cells=[]
        for frame in np.unique(pop['frame'][pop['source']==sid]):
            archive=cell_archive(parent,source,int(frame))
            binary=archive/'relaxed_binary_float16'
            cells.append(dict(frame=int(frame),archive=str(archive),
                metadata_sha256=sha(archive/'metadata.json'),manifest_sha256=sha(binary/'manifest.json')))
        return dict(id=int(sid),role=str(roles[0]),manifest_sha256=source['manifest_sha256'],cells=cells)
    with ThreadPoolExecutor(max_workers=8) as pool:
        result=list(pool.map(audit_source,np.unique(pop['source'])))
    return dict(rows=len(pop['event']),sources=result,source_count=len(result),
                frames=sum(len(s['cells']) for s in result),parent_identity=parent['identity'],
                population_sha256=sha(resolve_path(config['population_cache'])/'population.npz'),
                potential_sha256=parent['config']['potential_sha256'])


def paired_frame(source,frame,atoms,parent,raw=None):
    """Actual representatives and patch membership selected once from observed IDs."""
    if raw is None:raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    box=raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float)
    hot=np.mod(raw.positions[frame].astype(float),box)
    archive=cell_archive(parent,source,int(frame))
    relaxed=TemporalLAMMPSBinaryTrajectory.load(archive/'relaxed_binary_float16')
    relaxed.verify_checksums()
    np.testing.assert_array_equal(relaxed.atom_ids,raw.atom_ids)
    np.testing.assert_array_equal(relaxed.timesteps[0],raw.timesteps[frame])
    np.testing.assert_allclose(relaxed.box_high[0]-relaxed.box_low[0],box,rtol=0,atol=1e-5)
    cold=np.mod(relaxed.positions[0].astype(float),box)
    centers=np.searchsorted(raw.atom_ids,atoms)
    np.testing.assert_array_equal(raw.atom_ids[centers],atoms)
    queries=stencil();tree=cKDTree(hot,boxsize=box)
    selected=np.stack([representatives(hot,int(c),tree,box,queries)[0] for c in centers])
    unique,inverse=np.unique(selected,return_inverse=True)
    neighbors=tree.query(hot[unique],k=80,workers=1)[1]
    np.testing.assert_array_equal(neighbors[:,0],unique)
    output={}
    for name,points in (('hot',hot),('cold',cold)):
        clouds=points[neighbors]-points[unique,None]
        clouds-=box*np.rint(clouds/box)
        patches=[p[np.linalg.norm(p,axis=1)<8.].astype(np.float32) for p in clouds]
        actual=points[selected]-points[centers,None]
        actual-=box*np.rint(actual/box)
        output[name]=dict(patches=patches,actual=actual.astype(np.float32))
    return output,inverse.reshape(selected.shape),raw.atom_ids[selected]


def patch_features(encoder,patches,device,chunk=256):
    """One-shot eager export; campaigns reuse a compiled FeatureExtractor."""
    arrays=prepare_graphs(patches,encoder.cutoff,pin_memory=torch.device(device).type=='cuda')
    return FeatureExtractor(encoder,device,chunk,compile=False)(arrays)


def extract(study,domain,checkpoint,device,deadline):
    from src.research.supervised_onset.model import Model
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    if saved['arm']['input']!=domain or saved['config']['objective']!='hazard_nll':
        raise ValueError('Extraction requires a matched NLL-trained domain encoder')
    model=Model(saved['encoder_config'],saved['arm']).to(device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    model.requires_grad_(False)
    checksum=sha(checkpoint);root=study.cache/domain;root.mkdir(parents=True,exist_ok=True)
    prepared=json.loads((study.technical/'plan.json').read_text())
    if sha(study.technical/'inventory.json')!=prepared['inventory_sha256']:
        raise ValueError('Prepared archive inventory changed')
    frozen={s['id']:s for s in json.loads((study.technical/'inventory.json').read_text())['sources']}
    pop=population(study.config);parent=parent_plan(study.config)
    sources={s['id']:s for s in parent['sources']};receipts={}
    runtime=study.config['extraction']
    extractor=FeatureExtractor(model.encoder,device,runtime['chunk'],compile=runtime['compile'])
    timing_path=study.technical/f'extraction-{domain}-timings.jsonl'
    for sid in np.unique(pop['source']):
        dest=root/f'{sid}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['identity']!=study.identity or record['encoder_sha256']!=checksum or record['sha256']!=sha(dest):
                raise ValueError(f'Changed context feature shard: {dest}')
            receipts[str(sid)]=record;continue
        rows=np.flatnonzero(pop['source']==sid);parts=[];source=sources[int(sid)]
        raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
        if sha(raw.root/'manifest.json')!=frozen[int(sid)]['manifest_sha256']:
            raise ValueError(f'Observed source changed after preparation: {sid}')
        cells={c['frame']:c for c in frozen[int(sid)]['cells']}
        def prepare(frame):
            start=time.perf_counter()
            if time.time()>deadline-120:raise TimeoutError('Extraction checkpointed at completed source; resume this stage')
            ids=rows[pop['frame'][rows]==frame]
            cell=cells[int(frame)];saved_archive=Path(cell['archive'])
            archive=resolved_directory(str(saved_archive.parent))/saved_archive.name
            if (sha(archive/'metadata.json')!=cell['metadata_sha256'] or
                sha(archive/'relaxed_binary_float16/manifest.json')!=cell['manifest_sha256']):
                raise ValueError(f'Relaxed source changed after preparation: {sid}/{frame}')
            views,inverse,atoms=paired_frame(source,int(frame),pop['atom'][ids],parent,raw)
            read_geometry_done=time.perf_counter()
            arrays=prepare_graphs(views[domain]['patches'],model.encoder.cutoff,
                                  pin_memory=torch.device(device).type=='cuda')
            return dict(frame=int(frame),ids=ids,inverse=inverse,atoms=atoms,actual=views[domain]['actual'],
                        arrays=arrays,prepare_s=time.perf_counter()-start,
                        read_membership_s=read_geometry_done-start)
        with prepared_frames(np.unique(pop['frame'][rows]),prepare,
                workers=runtime['workers'],capacity=runtime['prefetch']) as frames:
            previous=time.perf_counter()
            for frame in frames:
                waited=time.perf_counter()-previous
                if time.time()>deadline-120:raise TimeoutError('Extraction checkpointed at completed source; resume this stage')
                values=extractor(frame['arrays'])
                parts.append(dict(rows=frame['ids'],query_atom_ids=frame['atoms'],actual=frame['actual'],
                                  **{k:v[frame['inverse']] for k,v in values.items()}))
                timing=dict(source=int(sid),frame=frame['frame'],domain=domain,
                    prepare_s=frame['prepare_s'],read_membership_s=frame['read_membership_s'],
                    consumer_wait_s=waited,**extractor.last_timing)
                with timing_path.open('a') as stream:stream.write(json.dumps(timing)+'\n')
                previous=time.perf_counter()
        arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order=np.argsort(arrays['rows']);arrays={k:v[order] for k,v in arrays.items()}
        np.testing.assert_array_equal(arrays['rows'],rows)
        temporary=dest.with_suffix('.tmp.npz');np.savez(temporary,**arrays);temporary.replace(dest)
        record=dict(identity=study.identity,encoder_sha256=checksum,sha256=sha(dest),rows=len(rows))
        write_json(receipt,record);receipts[str(sid)]=record
        print(json.dumps(dict(stage='extract',domain=domain,source=int(sid),rows=len(rows))),flush=True)
    write_json(root/'manifest.json',dict(identity=study.identity,encoder_sha256=checksum,shards=receipts))


def normalize(features,fit,source):
    """Source-weighted train statistics; equivariant fields get channel RMS only."""
    from src.research.local_predictability.metrics import source_weights
    w=source_weights(source[fit]);stats={};normalized={}
    for name,values in features.items():
        if name=='actual':normalized[name]=values;continue
        x=values[fit]
        if name=='z':
            mean=np.einsum('b,bnd->d',w,x)/x.shape[1]
            scale=np.sqrt(np.einsum('b,bnd->d',w,(x-mean)**2)/x.shape[1]).clip(1e-5)
        else:
            mean=0.
            scale=np.sqrt(np.einsum('b,bncm->c',w,x*x)/(x.shape[1]*x.shape[3])).clip(1e-5)[:,None]
        stats[name]=dict(mean=np.asarray(mean).tolist(),scale=scale.tolist())
        normalized[name]=((values-mean)/scale).astype(np.float32)
    return normalized,stats


class ContextCorpus:
    def __init__(self,study,domain,variant,device):
        self.required_fields=context_fields(variant)
        self.pop=population(study.config)
        self.split={r:np.flatnonzero(self.pop['role']==r) for r in ('train','selection','calibration','test')}
        root=study.cache/domain;manifest=json.loads((root/'manifest.json').read_text())
        if manifest['identity']!=study.identity:raise ValueError(f'Changed cache: {root}')
        from .normalization import Moments,apply
        from src.research.local_predictability.metrics import source_weights
        n=len(self.pop['event']);seen=np.zeros(n,dtype=bool)
        fit=self.split['train'];weights=np.zeros(n);weights[fit]=source_weights(self.pop['source'][fit])
        moments={k:Moments(k) for k in self.required_fields if k!='actual'}
        schemas={}
        # First pass: validate every shard, accumulating only training statistics.
        for sid,record in manifest['shards'].items():
            path=root/f'{sid}.npz'
            if sha(path)!=record['sha256']:raise ValueError(f'Feature checksum failed: {path}')
            with np.load(path) as a:
                rows=a['rows']
                if seen[rows].any() or not np.all(self.pop['source'][rows]==int(sid)):
                    raise ValueError(f'Invalid feature row alignment: {path}')
                seen[rows]=True
                for k in self.required_fields:
                    field=a[k]
                    if k in schemas and schemas[k]!=field.shape[1:]:
                        raise ValueError(f'Inconsistent {k} shape in {path}')
                    schemas[k]=field.shape[1:]
                    if not np.isfinite(field).all():raise FloatingPointError(f'Nonfinite context field: {path}/{k}')
                    if k!='actual':
                        local=np.flatnonzero(weights[rows]>0)
                        for start in range(0,len(local),256):
                            part=local[start:start+256]
                            moments[k].add(field[part],weights[rows[part]])
        if not seen.all():raise ValueError('Incomplete population coverage')
        self.scalers={k:moment.result() for k,moment in moments.items()}
        self.features={k:torch.empty((n,)+shape,device=device,dtype=torch.float32) for k,shape in schemas.items()}
        # One field of one source at a time; never materialize a full host corpus.
        for sid in manifest['shards']:
            with np.load(root/f'{sid}.npz') as a:
                rows=a['rows']
                for k in self.required_fields:
                    field=a[k]
                    for start in range(0,len(rows),256):
                        part=slice(start,start+256)
                        values=apply(field[part],self.scalers.get(k))
                        self.features[k][torch.as_tensor(rows[part],device=device)]=torch.as_tensor(values,device=device)
        self.nominal=torch.as_tensor(stencil(),device=device)
        self.events=torch.as_tensor(self.pop['event'],device=device)
        self.cache_identity=digest(manifest)

    def batch(self,ids):
        return {k:v[ids] for k,v in self.features.items()}|{'nominal':self.nominal[None].expand(len(ids),-1,-1)}
