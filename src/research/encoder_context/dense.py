"""All 64 fixed test centers over the complete observed 0.75 ps trajectories."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import centered
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path
from src.research.structural_state.common import sha,write_json


def prepare(study):
    root,plan=read_release(study.config['fixed_dataset']['root'])
    cache=study.cache/'dense-observed';cache.mkdir(parents=True,exist_ok=True)
    def source(s):
        dest=cache/str(s['id']);dest.mkdir(exist_ok=True);receipt=dest/'complete.json'
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['release_identity']!=plan['identity']:raise ValueError('Dense dataset identity changed')
            for name,checksum in record['files'].items():
                if sha(dest/name)!=checksum:raise ValueError(f'Changed dense shard {dest/name}')
            return record
        raw=ShootingBinaryTrajectory.load(dataset_path(s['dataset'])/s['relative_trajectory_path'])
        if sha(raw.root/'manifest.json')!=s['manifest_sha256']:raise ValueError('Dense source changed')
        atoms=np.asarray(s['center_atom_ids']);rows=np.searchsorted(raw.atom_ids,atoms)
        np.testing.assert_array_equal(raw.atom_ids[rows],atoms)
        n=s['frame_count'];values=np.lib.format.open_memmap(dest/'positions.npy',mode='w+',dtype='float32',shape=(n,64,80,3))
        for frame in range(n):
            box=raw.box_high[frame].astype(float)-raw.box_low[frame].astype(float)
            points=np.mod(raw.positions[frame].astype(float),box)
            tree=cKDTree(points,boxsize=box);neighbors=tree.query(points[rows],k=80,workers=1)[1]
            np.testing.assert_array_equal(neighbors[:,0],rows)
            values[frame]=centered(points,box,rows,neighbors)
        values.flush();del values
        labels=np.load(root/'benchmark/sources'/str(s['id'])/'labels.npy')
        if labels.shape!=(64,n):raise ValueError('Dense labels/time coverage differs')
        np.savez(dest/'observations.npz',atom=np.tile(atoms,n),frame=np.repeat(np.arange(n),64),
            source=np.full(n*64,s['id']),labels=labels.T.flatten())
        record=dict(source=s['id'],role='test',release_identity=plan['identity'],rows=n*64,
            files={p.name:sha(p) for p in (dest/'positions.npy',dest/'observations.npz')})
        write_json(receipt,record);print(json.dumps(dict(stage='dense-source',**record)),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(source,[s for s in plan['sources'] if s['role']=='test']))
    record=dict(release_identity=plan['identity'],sources=records,rows=sum(r['rows'] for r in records),
        cadence_ps=.75,domain='observed',selection='all frames and all 64 fixed centers; no event filtering')
    write_json(cache/'manifest.json',record);return record
