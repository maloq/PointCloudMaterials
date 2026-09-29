"""Dense, outcome-blind candidates from existing frames; retain invisible contexts."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry
from .data import context_atoms
from .interface import interface_mask,nearest_target


def make_plan(config_path):
    c=json.loads(Path(config_path).read_text());e=c['expansion']
    base=resolve_path(e['parent']);manifest=json.loads((base/'manifest.json').read_text())
    if sha(base/'manifest.json')!=e['parent_manifest_sha256'] or manifest['state']!='complete':raise ValueError('Changed expansion parent')
    parent=json.loads((base/'plan.json').read_text())
    if sha(base/'plan.json')!=manifest['plan_sha256']:raise ValueError('Changed parent plan')
    root=resolve_path(c['dataset']['root']);root.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol=c['protocol'],expansion=e,target=c['target'],dataset=c['dataset'],
        sources=parent['sources'],paths=parent['paths'],fixed_identity=parent['fixed_identity'],
        implementation={p.name:sha(p) for p in [Path(__file__),Path(ancestry.__file__),Path(__file__).with_name('interface.py'),Path(__file__).with_name('data.py')]})
    plan['identity']=digest(plan);dest=root/'plan.json'
    if dest.exists() and json.loads(dest.read_text())!=plan:raise ValueError('Expansion producer changed; use a new dataset root')
    write_json(dest,plan);return root


def prepare_source(config_path,sid):
    c=json.loads(Path(config_path).read_text());e=c['expansion'];root=resolve_path(c['dataset']['root'])
    plan=json.loads((root/'plan.json').read_text());folder=root/'sources'/str(sid);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        r=json.loads((folder/'complete.json').read_text())
        if r['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in r['files'].items()):raise ValueError(f'Changed expanded source {sid}')
        return r
    item=next(s for s in plan['sources'] if s['id']==sid)
    base=resolve_path(e['parent'])/'sources'/str(sid);receipt=json.loads((base/'complete.json').read_text())
    if any(sha(base/n)!=h for n,h in receipt['files'].items()):raise ValueError(f'Changed parent {sid}')
    with np.load(base/'rows.npz') as a:old={k:a[k] for k in a.files}
    old['interface_parent_index']=np.arange(len(old['atom']))
    old['expanded_candidate']=np.zeros(len(old['atom']),bool)
    values=[old];banks=[];offset=receipt['patches'];audits=[]
    audit=resolve_path(c['dataset']['audit'])/'technical/sources'/str(sid)
    if sha(audit/'graph.npz')!=json.loads((audit/'graph-complete.json').read_text())['sha256']:raise ValueError('Changed crystal graph')
    with np.load(audit/'graph.npz') as a:g={k:a[k] for k in a.files}
    settings=json.loads(resolve_path(c['dataset']['audit_config']).read_text())
    events,roots,_=ancestry.establish(g,settings['lineage']['thresholds'][0]);access=ancestry.GeometryAccess(settings,item,audit,g)
    ptm=json.loads((audit/'ptm-complete.json').read_text());checked=set()
    frames=np.unique(np.rint(np.linspace(0,item['frame_count']-1,e['frames_per_source'])).astype(int))
    if len(frames)!=e['frames_per_source']:raise ValueError('Insufficient distinct source frames')
    rng=np.random.default_rng(np.random.SeedSequence([e['seed'],sid]));candidate_total=0
    for frame in frames:
        frame=int(frame);chunk=settings['ptm']['chunk_frames'];start=frame//chunk*chunk;stop=min(item['frame_count'],start+chunk)
        name=f'ptm-{start:04d}-{stop:04d}.npz'
        if name not in checked:
            if sha(audit/name)!=ptm['files'][name]:raise ValueError(f'Changed PTM {sid}/{frame}')
            checked.add(name)
        points,box,dense=access.frame(frame);tree=cKDTree(points,boxsize=box)
        # Exclude only previously sampled uniform IDs, independent of their labels.
        existing=old['atom'][(old['frame']==frame)&(old['kind']==1)]
        candidates=np.flatnonzero(~np.isin(access.raw.atom_ids,existing))
        atoms=rng.choice(candidates,e['centers_per_frame'],replace=False);candidate_total+=len(atoms)
        nodes=np.flatnonzero((g['frame']==frame)&(g['size']>=64))
        known=[n for n in nodes if any(events[r-1]['confirmation_frame']<=frame for r in roots[n])]
        solid=np.isin(dense,known) if known else np.zeros(len(points),bool)
        boundary,_=interface_mask(points,box,solid,c['target']['neighbor_cutoff_A'],c['target']['minimum_disordered_component_atoms'])
        labels=nearest_target(points,box,atoms,boundary,c['dataset']['tie_tolerance_A'])
        chosen=[context_atoms(points,box,tree,int(a),c['dataset']['shells']) for a in atoms]
        queries=np.stack([v[0] for v in chosen]);actual=np.stack([v[1] for v in chosen])
        patches,inverse=np.unique(queries,return_inverse=True)
        distances,neighbors=tree.query(points[patches],k=80,workers=1)
        patch_visible=(boundary[neighbors]&(distances<8)).any(1)
        visible=patch_visible[inverse].reshape(queries.shape)
        keep=~visible.any(1)
        audits.append(dict(frame=frame,candidates=len(atoms),retained=int(keep.sum()),
            near20_retained=int((keep&(labels['distance']<=20)).sum()),near32_retained=int((keep&(labels['distance']<=32)).sum()),
            finite_retained=int((keep&np.isfinite(labels['distance'])).sum()),
            crystal_interior_retained=int((keep&solid[atoms]).sum())))
        if not keep.any():continue
        # Cache geometry only for the retained new queries. Original benchmark
        # rows, including visible ones, remain unchanged for paired secondary exports.
        queries=queries[keep];actual=actual[keep];atoms=atoms[keep]
        unique,inverse=np.unique(queries,return_inverse=True);which=np.searchsorted(patches,unique)
        nn=neighbors[which];xyz=points[nn]-points[unique,None];xyz-=box*np.rint(xyz/box)
        crystal_visible=(solid[nn]&(np.linalg.norm(xyz,axis=-1)<8)).any(1)[inverse].reshape(queries.shape)
        crystal=nearest_target(points,box,atoms,solid,c['dataset']['tie_tolerance_A'])
        n=len(atoms)
        record=dict(indices=inverse.reshape(queries.shape).astype(np.int64)+offset,actual=actual,
            **{k:v[keep] for k,v in labels.items()},visible_local=np.zeros(n,bool),visible_context=np.zeros(n,bool),
            source=np.full(n,sid,np.int32),frame=np.full(n,frame,np.int32),atom=access.raw.atom_ids[atoms],
            role=np.full(n,item['role']),row=np.full(n,-1),kind=np.ones(n,int),path=np.full(n,-1),travel=np.zeros(n),
            parent_index=np.full(n,-1),crystal_distance=crystal['distance'],
            crystal_visible_local=crystal_visible[:,0],crystal_visible_context=crystal_visible.any(1),
            inside_crystal=solid[atoms],interface_member=boundary[atoms],interface_exists=np.full(n,boundary.any()),
            interface_parent_index=np.full(n,-1),expanded_candidate=np.ones(n,bool))
        if old.keys()!=record.keys():raise ValueError('Unexpected expansion row schema')
        if record['interface_member'].any():raise ValueError('Interface member cannot be invisible')
        values.append(record);banks.append(xyz.astype(np.float32));offset+=len(xyz)
    original=np.load(base/'positions.npy',mmap_mode='r')
    bank=np.lib.format.open_memmap(folder/'positions.npy',mode='w+',dtype=np.float32,shape=(offset,80,3))
    bank[:len(original)]=original;begin=len(original)
    for xyz in banks:bank[begin:begin+len(xyz)]=xyz;begin+=len(xyz)
    bank.flush();del bank
    fields={k:np.concatenate([v[k] for v in values]) for k in old}
    for k in old:
        if not np.array_equal(fields[k][:len(old['atom'])],old[k]):raise ValueError(f'Changed parent {k}')
    np.savez(folder/'rows.npz',**fields);write_json(folder/'candidate-counts.json',audits)
    # These pre-exclusion denominators preserve the conditional original sampling
    # population; otherwise sources with very few invisible candidates get overweighted.
    population_counts={str(k):int((old['kind']==k).sum())+(candidate_total if k==1 else 0) for k in (0,1,2)}
    r=dict(identity=plan['identity'],source=sid,rows=len(fields['atom']),patches=offset,
        original_interface_rows=len(old['atom']),candidate_uniform_added=candidate_total,
        new_invisible_rows=int(fields['expanded_candidate'].sum()),population_counts=population_counts,
        near20_new=int((fields['expanded_candidate']&(fields['distance']<=20)).sum()),
        files={n:sha(folder/n) for n in ('positions.npy','rows.npz','candidate-counts.json')})
    write_json(folder/'complete.json',r)
    print(json.dumps(dict(stage='expanded',source=sid,retained=r['new_invisible_rows'],near20=r['near20_new'])),flush=True)
    return r


def seal(config_path):
    c=json.loads(Path(config_path).read_text());root=resolve_path(c['dataset']['root']);plan=json.loads((root/'plan.json').read_text())
    records=[];counts={role:dict(rows=0,eligible=0,near20=0,new_candidates=0,new_eligible=0) for role in ('train','selection','calibration','test')}
    for item in plan['sources']:
        p=root/'sources'/str(item['id']);r=json.loads((p/'complete.json').read_text())
        if r['identity']!=plan['identity'] or any(sha(p/n)!=h for n,h in r['files'].items()):raise ValueError(f'Changed/unsealed {item["id"]}')
        records.append(r)
        with np.load(p/'rows.npz') as a:
            mask=(a['kind']<2);clear=mask&~a['visible_context'];s=counts[item['role']]
            s['rows']+=int(mask.sum());s['eligible']+=int(clear.sum());s['near20']+=int((clear&(a['distance']<=20)).sum())
            s['new_candidates']+=r['candidate_uniform_added'];s['new_eligible']+=r['new_invisible_rows']
    if counts['train']['near20']<=71:raise ValueError('Expansion did not increase requested invisible near-interface examples')
    write_json(root/'manifest.json',dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),sources=records))
    write_json(root/'state.json',dict(state='complete',counts=counts,total_rows=sum(r['rows'] for r in records),
        patches=sum(r['patches'] for r in records),coordinate_GiB=sum(r['patches'] for r in records)*80*3*4/2**30))
