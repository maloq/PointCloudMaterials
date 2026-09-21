"""Reuse audited features; obtain exact short-horizon labels from saved onset frames."""
import json
from pathlib import Path
import numpy as np
from src.data.structural_pretraining.prepare import file_hash, save_json, digest
from src.project_runtime.paths import resolve_path
from src.research.forecast_crystallization.local_metrics import first_sustained_onset

HORIZONS = np.array([.75, 3., 6., 9., 12.])
GEOMETRY = np.r_[0:80,112:117]
MOTION = np.r_[80:112,117:128]
ORDER_NAMES = ['q4','q6','w4','w6','qbar6','mean_q6_coherence','density_r12','smooth_coordination']
BLOCKS = {'radial_pair':np.r_[0:64,112:117], 'angular':np.arange(64,80),
          'order':np.arange(128,134), 'density':np.arange(134,136),
          'motion':MOTION, 'outer_geometry':np.array([136,137,142,143]),
          'outer_motion':np.array([138,139,140,141,144,145,146,147]),
          'history':np.arange(148,427)}
GROUPS = list(BLOCKS)

def event_bins(onset, anchor):
    delay=np.asarray(onset)-np.asarray(anchor)
    if np.any(delay<=0):raise ValueError('Diagnostic population must be strictly before first sustained onset')
    return np.searchsorted((HORIZONS/.75).astype(int),delay).astype(np.int64)

def observed_features(packet,order,shell,centers,frames):
    """Current descriptors and geometry/order changes over past 3,6,12 ps only."""
    if np.min(frames)<16:raise ValueError('Insufficient observed history')
    current=np.c_[packet[centers,frames],order[centers,frames]]
    columns=np.r_[GEOMETRY,128:136]
    history=[]
    for offset in (4,8,16):
        past=np.c_[packet[centers,frames-offset],order[centers,frames-offset]]
        history.append(current[:,columns]-past[:,columns])
    return np.c_[current,shell[centers,frames],*history].astype(np.float32)

def prepare(config):
    root=resolve_path(config['output'])/'technical';root.mkdir(parents=True,exist_ok=True)
    dest=root/'data';dest.mkdir(exist_ok=True)
    identity=digest(config)
    if (dest/'complete.json').exists():
        if json.loads((dest/'complete.json').read_text())['identity']!=identity:raise ValueError('Diagnostic inputs changed')
        return
    plan_path=resolve_path(config['plan']);plan=json.loads(plan_path.read_text())
    pop_path=resolve_path(config['population']);pop_hash=file_hash(pop_path);pop=np.load(pop_path)
    roles={s['id']:s.get('validation_role',s['split']) for s in plan['sources']}
    ancestry=[s['lineage'] for s in plan['sources']]
    if len(set(ancestry))!=len(ancestry):raise ValueError('Source ancestry is not independent')
    n=len(pop['source']);observed=np.empty((n,427),np.float32);events=np.empty(n,np.int64)
    delays=np.empty(n,np.float32);temps=np.empty(n,np.float32)
    manifests=[];cache=resolve_path(plan['config']['cache']);assay=resolve_path(plan['config']['assay_cache'])
    for source in plan['sources']:
        sid=source['id'];ix=np.flatnonzero(pop['source']==sid)
        rows=pop['rows'][ix];np.testing.assert_array_equal(rows[:,0],sid)
        ci=rows[:,2];frames=np.array(plan['anchors'])[rows[:,1]]
        np.testing.assert_array_equal(pop['role'][ix],roles[sid])
        receipt=json.loads((cache/str(sid)/'complete.json').read_text())
        if receipt['identity']!=plan.get('cache_identity',plan['identity']):raise ValueError(f'Source identity changed: {sid}')
        path=assay/source['shard']
        if file_hash(path)!=source['shard_sha256']:raise ValueError(f'Assay checksum mismatch: {sid}')
        with np.load(path) as a:
            np.testing.assert_array_equal(a['atom_ids'],source['center_atom_ids'])
            np.testing.assert_allclose(a['times_ps'],np.arange(801)*.75,rtol=0,atol=1e-8)
            observed[ix]=observed_features(a['packet'],a['order'],a['shell'],ci,frames)
            verified_onset=first_sustained_onset(np.isin(a['labels'],[1,2,3]),3)[ci]
        np.testing.assert_array_equal(observed[ix,:136],pop['descriptor'][ix])
        onset=np.load(cache/str(sid)/'onset.npy')[ci]
        np.testing.assert_array_equal(onset,verified_onset)
        if np.any(frames+16+2>=source['frame_count']):raise ValueError('Insufficient sustained-event followup')
        events[ix]=event_bins(onset,frames);delays[ix]=(onset-frames)*.75;temps[ix]=source['temperature_K']
        manifests.append(dict(source=sid,lineage=source['lineage'],role=roles[sid],assay_sha256=source['shard_sha256']))
    if not np.isfinite(observed).all():raise FloatingPointError('Nonfinite diagnostic observations')
    np.save(dest/'observed.npy',observed)
    np.savez(dest/'population.npz',source=pop['source'],role=pop['role'],rows=pop['rows'],condition=pop['condition'],event=events,delay=delays,temperature=temps)
    records={}
    for name,folder_text in config['encoders'].items():
        folder=resolve_path(folder_text);record=json.loads((folder/'record.json').read_text())
        if record['population_sha256']!=pop_hash or record['protected_overlap']:raise ValueError(f'Encoder cohort/ancestry differs: {name}')
        if file_hash(Path(record['checkpoint']))!=record['checkpoint_sha256']:raise ValueError(f'Checkpoint changed: {name}')
        features=np.empty((n,128),np.float32);hashes={}
        for sid in np.unique(pop['source']):
            path=folder/'features'/f'{sid}.npy';info=json.loads(path.with_suffix('.json').read_text())
            sha=file_hash(path)
            if info['checkpoint_sha256']!=record['checkpoint_sha256'] or sha!=info['sha256']:raise ValueError(f'Frozen features changed: {name}/{sid}')
            values=np.load(path);ix=np.flatnonzero(pop['source']==sid)
            if values.shape!=(len(ix),128):raise ValueError((name,sid,values.shape,len(ix)))
            features[ix]=values;hashes[str(sid)]=sha
        if not np.isfinite(features).all():raise FloatingPointError(f'Nonfinite export: {name}')
        np.save(dest/f'{name}.npy',features)
        records[name]=dict(record_sha256=file_hash(folder/'record.json'),checkpoint_sha256=record['checkpoint_sha256'],feature_sha256=hashes)
    save_json(dest/'complete.json',dict(identity=identity,plan_sha256=file_hash(plan_path),population_sha256=pop_hash,rows=n,
        counts={r:int((pop['role']==r).sum()) for r in np.unique(pop['role'])},sources=manifests,encoders=records,
        observation='current descriptors and strictly past geometry/order differences; shell annuli 7-17 and 17-25 Angstrom; no PTM inputs',
        horizons_ps=HORIZONS.tolist(),blocks={k:v.tolist() for k,v in BLOCKS.items()}))

def variants():
    geometrical=['radial_pair','angular','order','density']
    return {'z':[],**{'z+'+k:[k] for k in GROUPS},'z+geometry':geometrical,
            'z+current':geometrical+['motion'],'z+all':GROUPS,'z+shuffled':GROUPS,
            'conditions':[],'geometry':geometrical,'current':geometrical+['motion'],'all':GROUPS}

def permutation_control(observed,role,temperature,seed):
    # Negative control only: no outcomes, no train/held-out mixing. Each fit sees
    # the same fixed within-role/temperature permutation, independent of its label.
    rng=np.random.default_rng(seed);result=np.empty_like(observed)
    for r in np.unique(role):
        for t in np.unique(temperature):
            ix=np.flatnonzero((role==r)&(temperature==t))
            result[ix]=observed[rng.permutation(ix)]
    return result
