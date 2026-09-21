"""Frozen snapshot readouts on original MD onset, never on quenched labels."""
import json
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,geometry_packet
from src.analysis.liquid_structure import persistence_image
from src.training_methods.neighborhood_jepa.regularization.data import order_targets
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.research.crystallization_information.data import event_bins
from src.research.crystallization_information.runtime import fit,standardize
from .prepare import graph_arrays,target_cloud,arm_for_name


def prepare(plan):
    c=plan['config'];root=resolve_path(c['output'])/'technical/assay';root.mkdir(parents=True,exist_ok=True)
    if (root/'ready.json').exists():return True
    cache=resolve_path(c['cache']);original=json.loads(resolve_path(c['assay_plan']).read_text())
    for s in plan['sources']:
        for f in c['frames']:
            if not (cache/'cells'/f'{s["id"]}-{f}'/'complete.json').exists():return False
    with np.load(resolve_path(c['population'])) as p:
        anchor=np.array(original['anchors'])[p['rows'][:,1]];keep=np.isin(anchor,c['frames'])
        pop={k:p[k][keep] for k in ('source','rows','condition','role')}
        pop['original_geometry']=p['descriptor'][keep];frames=anchor[keep]
    n=len(frames);pop.update(graph=np.empty(n,np.int64),event=np.empty(n,np.int64),delay=np.empty(n,np.float32),temperature=np.empty(n,np.float32))
    descriptors={domain:np.empty((n,237),np.float32) for domain in ('hot','cold')}
    for s in plan['sources']:
        sid=s['id'];ix=np.flatnonzero(pop['source']==sid);ci=pop['rows'][ix,2]
        onset=np.load(resolve_path(original['config']['cache'])/str(sid)/'onset.npy')[ci]
        pop['event'][ix]=event_bins(onset,frames[ix]);pop['delay'][ix]=(onset-frames[ix])*.75;pop['temperature'][ix]=s['temperature_K']
        for domain in descriptors:
            clouds=[]
            for f in c['frames']:
                with np.load(cache/'cells'/f'{sid}-{f}'/'clouds.npz') as a:
                    rows=np.searchsorted(a['query_atom_ids'][:,0],s['center_atom_ids'])
                    np.testing.assert_array_equal(a['query_atom_ids'][rows,0],s['center_atom_ids'])
                    clouds.extend(a[domain][rows,0])
            clouds=np.array(clouds);arrays,_=graph_arrays(clouds,c['scale'])
            folder=cache/'assay'/domain/str(sid);folder.mkdir(parents=True,exist_ok=True)
            for name,values in arrays.items():np.save(folder/f'{name}.npy',values)
            save_json(folder/'complete.json',dict(identity=plan['identity'],hashes={name:file_hash(folder/f'{name}.npy') for name in arrays}))
            graph=np.array([c['frames'].index(int(f))*16+int(i) for f,i in zip(frames[ix],ci,strict=True)])
            pop['graph'][ix]=graph
            unique=np.unique(graph);cropped={int(g):target_cloud(clouds[g],c['scale']) for g in unique}
            d={g:np.r_[geometry_packet(x),persistence_image(x),order_targets(x*REFERENCE_RADIUS/c['scale'],c['scale'])] for g,x in cropped.items()}
            descriptors[domain][ix]=np.array([d[int(g)] for g in graph])
    for domain,values in descriptors.items():
        if not np.isfinite(values).all():raise FloatingPointError(f'Nonfinite {domain} descriptors')
        np.save(root/f'{domain}-descriptors.npy',values)
        save_json(root/f'{domain}-plan.json',dict(identity=plan['identity'],scale=c['scale'],sources=plan['sources'],config=dict(cache=str(cache/'assay'/domain))))
    np.savez(root/'population.npz',**pop)
    save_json(root/'ready.json',dict(identity=plan['identity'],rows=n,counts={r:int((pop['role']==r).sum()) for r in np.unique(pop['role'])},
        events={r:int(((pop['event']<5)&(pop['role']==r)).sum()) for r in np.unique(pop['role'])},
        frames=c['frames'],horizons_ps=[.75,3,6,9,12],labels='Original MD first sustained local crystalline onset; relaxation never changes labels'))
    return True


def extract(plan,name):
    from src.training_methods.neighborhood_jepa.v2.extract import run
    c=plan['config'];root=resolve_path(c['output'])/'technical';a=root/'assay'
    domain='cold' if name=='parent_cold' or (not name.startswith('parent_') and arm_for_name(c,name)=='relaxed_to_relaxed') else 'hot'
    parent=name.startswith('parent_');checkpoint=resolve_path(c['warm_checkpoint']) if parent else root/'runs'/name/'best.pt'
    folder=a/name;folder.mkdir(exist_ok=True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent_manifest=json.loads(resolve_path(c['normalization_manifest']).read_text())
    seen=set(parent_manifest['train_roots']+parent_manifest['selection_roots'])
    # All fitted ancestors are development-only; frozen readouts retain the
    # original calibration/test split, including for parent controls.
    protected={s['lineage'] for s in plan['sources'] if s.get('validation_role',s['split']) in ('test','calibration')}
    if seen&protected:raise ValueError(f'Pretrained ancestry overlaps held-out sources: {seen&protected}')
    record=dict(name=name,kind='v2_large' if parent else 'regularization',checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),
        directory=str(folder),assay_plan=str(a/f'{domain}-plan.json'),assay_cache=str(resolve_path(c['cache'])/'assay'/domain),
        population=str(a/'population.npz'),population_sha256=file_hash(a/'population.npz'),protected_overlap=[],input=domain,
        parent_data_identity=parent_manifest['identity'],checkpoint_data_identity=saved['manifest']['data_identity'])
    if parent and saved['manifest']['data_identity']!=parent_manifest['identity']:raise ValueError('Parent normalization/ancestry release does not match checkpoint')
    save_json(folder/'record.json',record);run(folder/'record.json')
    pop=dict(np.load(a/'population.npz'));z=np.empty((len(pop['source']),128),np.float32)
    for sid in np.unique(pop['source']):z[pop['source']==sid]=np.load(folder/'features'/f'{sid}.npy')
    np.save(folder/'features.npy',z);save_json(folder/'complete.json',dict(checkpoint_sha256=file_hash(checkpoint),feature_sha256=file_hash(folder/'features.npy')))


def probes(plan,name):
    c=plan['config'];root=resolve_path(c['output']);a=root/'technical/assay';pop=dict(np.load(a/'population.npz'));train=np.flatnonzero(pop['role']=='train');normalizers={}
    x=np.zeros((len(pop['source']),128+237+7),np.float32)
    # Equal neural readout capacity for all representations and descriptors.
    if name in ('geometry_hot','geometry_cold','original_geometry','conditions'):
        if name=='original_geometry':values=pop['original_geometry']
        elif name!='conditions':values=np.load(a/f'{name.removeprefix("geometry_")}-descriptors.npy')
        if name!='conditions':
            values,mean,std=standardize(values,train,pop['source']);x[:,128:128+values.shape[1]]=values;normalizers.update(mean=mean,std=std)
    else:
        values=np.load(a/name/'features.npy');x[:,:128],mean,std=standardize(values,train,pop['source']);normalizers.update(mean=mean,std=std)
    # The original condition vector is predetermined temperature only.
    x[:,-7:]=pop['condition'];torch.set_num_threads(c['probe']['threads'])
    dest=a/name;dest.mkdir(exist_ok=True);np.savez(dest/'normalizer.npz',**normalizers)
    for kind in ('linear','mlp'):
        fit(c['probe'],dict(encoder=name,variant='snapshot',readout=kind,task='hazard'),pop,x,pop['event'],root/'readouts')
    save_json(dest/'probe-complete.json',dict(state='complete'))
