"""Frozen local80 response encoder on the retained comparison observations."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(config):
    c = read(config)
    # All model imports belong to the exact recorded training source tree.
    sys.path[:] = [c['producer']] + [p for p in sys.path if p != str(Path(__file__).parent)]
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
    from src.research.local_response.model import Predictor
    from src.postprocessing.protocols.extract import raw_source
    from src.postprocessing.io import frame_geometry
    from src.data.fixed_cohort.protocol import centered

    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name, expected in c['inference_dependencies'].items():
        if sha(Path(c['producer'])/name) != expected:
            raise ValueError('Changed frozen producer: '+name)
    if sha(c['checkpoint']) != c['checkpoint_sha256']:
        raise ValueError('Changed selected checkpoint')
    saved = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
    if saved['epoch'] != c['model']['epoch'] or saved['selection_nll'] != c['model']['selection_nll']:
        raise ValueError('Checkpoint differs from recorded validation selection')
    model = Predictor(c['training_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    started = time.monotonic()
    out = Path(c['output']); data = out/'data'; data.mkdir(parents=True, exist_ok=True)
    assay = Path(c['assay'])
    plan = read(assay.parent/'plan.json')
    if plan['fixed_identity'] != c['fixed_identity']:
        raise ValueError('Different fixed Al64 release')
    manifest = read(assay.parent/'manifest.json')
    for name in ('parents', 'role', 'uniform', 'source', 'frame', 'atom'):
        if sha(assay/(name+'.npy')) != manifest['files']['assay/'+name+'.npy']:
            raise ValueError('Changed fixed assay '+name)
    parents = np.load(assay/'parents.npy', mmap_mode='r')
    arrays = {k: np.load(assay/(k+'.npy')) for k in ('role','uniform','source','frame','atom')}
    batch = c['batch_size']

    def features(patches, rows):
        with torch.inference_mode():
            for start in range(0, len(rows), batch):
                ids = rows[start:start+batch]
                x = np.array(patches[ids, :80], dtype=np.float32, copy=True)
                if x.shape[1:] != (80,3) or np.any(x[:,0]) or not np.isfinite(x).all():
                    raise ValueError('Invalid center-relative nearest80 geometry')
                z = model.encode(torch.from_numpy(x).cuda()).float().cpu().numpy()
                if z.shape != (len(ids),128) or not np.isfinite(z).all():
                    raise ValueError('Invalid native response embedding')
                yield start, ids, z

    def assign(z):
        return _labels_inertia_threadpool_limit(np.ascontiguousarray(z, dtype=np.float32),
            np.ones(len(z), np.float32), centers, n_threads=1, return_inertia=False).astype(np.uint8)

    fit_rows = np.flatnonzero((arrays['role']=='train') & arrays['uniform'])
    cluster_path = data/'clusters.npz'
    if not cluster_path.exists():
        values = np.empty((len(fit_rows),128), np.float32)
        for start, ids, z in features(parents, fit_rows):
            values[start:start+len(ids)] = z
            if start % (batch*32) == 0:
                print(f'Training observations {start}/{len(fit_rows)}; {time.monotonic()-started:.1f}s',flush=True)
        km = MiniBatchKMeans(n_clusters=7, batch_size=4096, n_init=3, max_iter=200,
            random_state=20260929, reassignment_ratio=0).fit(values)
        np.savez_compressed(cluster_path, centers=km.cluster_centers_, fit_rows=fit_rows)
        del values
    with np.load(cluster_path) as z:
        centers = z['centers']
        if not np.array_equal(z['fit_rows'],fit_rows):raise ValueError('Changed cluster fitting rows')

    pages = {kind:read(Path(ds['reference'])/'index.json')['data'] for kind,ds in c['datasets'].items()}
    page = pages['matched']
    test_rows = np.flatnonzero((arrays['role']=='test') & arrays['uniform'])
    for field in ('source','frame','atom'):
        if not np.array_equal(arrays[field][test_rows],page[field]):
            raise ValueError('Changed held-out '+field)
    held = data/'heldout.npz'
    if not held.exists():
        values = np.concatenate([z for _,_,z in features(parents,test_rows)])
        np.savez_compressed(held,z=values,labels=assign(values),assay_row=test_rows,
            **{k:arrays[k][test_rows] for k in ('source','frame','atom')})
    with np.load(held) as z: held_labels=z['labels']
    raw = raw_source(next(s for s in plan['sources'] if s['id']==908))
    static_parts=[]
    for kind, ds in c['datasets'].items():
        reference = Path(ds['reference']); page=pages[kind]
        for snap in page['md']['snapshots']:
            key=snap['key']; folder=data/kind/key; folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                rec=read(folder/'complete.json')
                for name,expected in rec['files'].items():
                    if sha(folder/name)!=expected:raise ValueError('Changed inference '+key+'/'+name)
                if kind=='static':
                    with np.load(folder/'sampled.npz') as z:static_parts.append(dict(z))
                continue
            md=read((reference/'interactive'/snap['asset']).with_suffix('.json'))['data']
            display=np.column_stack([md[k] for k in ('x','y','z')])
            geometry=read((reference/'interactive'/page['travel'][key]['geometry']['asset']).with_suffix('.json'))['data']
            travel=np.asarray(geometry['rows'],dtype=int)
            chosen=np.flatnonzero(np.asarray(page['frame'])==snap['frame'])
            if kind=='matched':
                points,box=frame_geometry(raw,snap['frame']);atoms=raw.atom_ids
                if np.max(np.abs(points-display))>5.1e-5:raise ValueError('MD coordinate mismatch '+key)
                neighbors=cKDTree(points,boxsize=box).query(points,k=80,workers=4)[1]
                patches=centered(points,box,np.arange(len(points)),neighbors)
                original=np.flatnonzero((arrays['source']==908)&(arrays['frame']==snap['frame']))
                old_rows=np.searchsorted(atoms,arrays['atom'][original])
                if not np.array_equal(atoms[old_rows],arrays['atom'][original]):raise ValueError('Assay atom identity')
                patches[old_rows]=parents[original,:80]
                chosen=chosen[np.asarray(page['source'])[chosen]==908]
                observed=np.searchsorted(atoms,np.asarray(page['atom'])[chosen])
            else:
                source=Path(c['static_data'])/(key+'.npy')
                expected=c['static_sources'][key]
                if sha(source)!=expected['source_sha256']:raise ValueError('Changed static source '+key)
                points=np.asarray(np.load(source),np.float64);tree=cKDTree(points)
                error,atoms=tree.query(display,k=1,workers=4)
                if error.max()>1e-6 or len(np.unique(atoms))!=snap['count']:
                    raise ValueError('Cannot recover exact static grid centers '+key)
                neighbors=tree.query(points[atoms],k=80,workers=4)[1]
                patches=(points[neighbors]-points[atoms,None]).astype(np.float32)
                lookup={int(atom):i for i,atom in enumerate(atoms)}
                observed=np.array([lookup[int(atom)] for atom in np.asarray(page['atom'])[chosen]])
            if len(patches)!=snap['count'] or not np.array_equal(atoms[travel],geometry['atoms']):
                raise ValueError('Dense/travel identities differ '+key)
            # Audit exact original patch reconstruction before running the encoder.
            temporary=folder/'patches-replay.npy';np.save(temporary,patches)
            patch_sha=sha(temporary);temporary.unlink()
            if patch_sha!=c['patch_sha256'][kind][key]:
                raise ValueError('Reconstructed nearest80 patches differ from original '+key)
            retained=np.unique(np.r_[travel,observed]);values=np.empty((len(retained),128),np.float32)
            labels=np.empty(len(patches),np.uint8)
            for start, ids, z in features(patches,np.arange(len(patches))):
                labels[ids]=assign(z)
                take=(retained>=start)&(retained<start+len(ids))
                values[take]=z[retained[take]-start]
                if start%(batch*128)==0:print(f'{kind}/{key} {start}/{len(patches)}; {time.monotonic()-started:.1f}s',flush=True)
            if kind=='matched' and not np.array_equal(labels[observed],held_labels[chosen]):
                raise ValueError('Sampled/dense assignment replay differs '+key)
            np.savez_compressed(folder/'labels.npz',encoder=labels)
            travel_z=values[np.searchsorted(retained,travel)]
            tensor=torch.as_tensor(travel_z,device='cuda');dist=torch.cdist(tensor,tensor);dist.fill_diagonal_(float('inf'))
            np.savez_compressed(folder/'travel.npz',z=travel_z,rows=travel,
                neighbors=dist.topk(16,largest=False).indices.cpu().numpy(),labels=labels[travel])
            sample_rows=[];counts=[]
            for cluster in range(7):
                members=np.flatnonzero(labels==cluster);counts.append(len(members))
                seed=int.from_bytes(hashlib.sha256(f'20260929/{key}/neural/{c["model"]["id"]}/{cluster}'.encode()).digest()[:8],'little')
                sample_rows.extend(np.random.default_rng(seed).choice(members,min(5,len(members)),replace=False))
            sample_rows=np.asarray(sample_rows,dtype=int)
            np.savez_compressed(folder/'samples.npz',rows=sample_rows,patches=patches[sample_rows],
                atoms=atoms[sample_rows],clusters=labels[sample_rows],counts=np.asarray(counts))
            files=['labels.npz','travel.npz','samples.npz']
            if kind=='static':
                part=dict(z=values[np.searchsorted(retained,observed)],labels=labels[observed],
                    frame=np.full(len(observed),snap['frame']),atom=atoms[observed])
                np.savez_compressed(folder/'sampled.npz',**part);static_parts.append(part);files.append('sampled.npz')
            save(folder/'complete.json',dict(rows=len(patches),checkpoint_sha256=c['checkpoint_sha256'],
                patch_sha256=patch_sha,patch_reconstruction_exact=True,files={name:sha(folder/name) for name in files}))
            del patches,values,neighbors
            print('Completed '+kind+'/'+key,flush=True)
    np.savez_compressed(data/'static.npz',**{k:np.concatenate([p[k] for p in static_parts]) for k in static_parts[0]})
    save(out/'technical/inference.json',dict(checkpoint_sha256=c['checkpoint_sha256'],model=c['model'],
        fixed_release_identity=c['fixed_identity'],track='all64',input_record=c['input_record'],
        precision='float32; TF32 disabled; no autocast',exported_state='native normalized scalar z128 before predictor head',
        batch_size=batch,neural_training=False,descriptor_refit=False,patch_reconstruction_exact=True,
        device=torch.cuda.get_device_name(),slurm_allocation=os.environ.get('SLURM_JOB_ID'),seconds=time.monotonic()-started))


if __name__=='__main__':
    main(sys.argv[1])
