"""Expand cached static sampling centers to complete atomic receptive fields."""
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree
from src.data_utils.temporal_campaign import write_json


def attach_atomic_context(dataset,config,sample_cache_dir):
    """Keep the existing grid/atom centers; replace only their input context.

    This workflow uses physical coordinates and interior static centers. It does
    not invent periodic box lengths from coordinate extrema or reflect padding.
    The cached result is directly consumed by PointCloudDataset's batch reader.
    """
    if dataset.normalize or not dataset.return_coords:
        raise ValueError('atomic_context requires normalize=false and return_coords=true')
    reference=Path(sample_cache_dir);metadata=json.loads((reference/'metadata.json').read_text())
    directory=Path(config['cache_dir']);directory.mkdir(parents=True,exist_ok=True)
    p=int(config['points']);halo=float(config['required_radius_A'])
    sources={s['name']:s for s in metadata['request']['sources']}
    arrays=[]
    for si,shard in enumerate(metadata['shards']):
        source=sources[shard['source']];source_file=Path(source['root'])/shard['file']
        signature=dict(reference_fingerprint=metadata['fingerprint'],source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),points=p,required_radius_A=halo,count=shard['count'])
        path=directory/(shard['file']+'.context.npy');record=directory/(shard['file']+'.json')
        if record.exists():
            saved=json.loads(record.read_text())
            if saved['signature']!=signature:raise ValueError(f'Atomic context signature mismatch: {record}; use a new cache directory')
        else:
            points=np.load(source_file).astype(np.float64);coords=dataset._cache_coord_arrays[si]
            margin=np.minimum(coords-points.min(0),points.max(0)-coords).min()
            if margin<=halo:raise ValueError(f'{source_file}: static boundary margin {margin} A cannot support {halo} A; select interior centers')
            tree=cKDTree(points);result=open_memmap(path,mode='w+',dtype='float16',shape=(len(coords),p,3));excluded=float('inf')
            for start in range(0,len(coords),2048):
                centers=coords[start:start+2048]
                distances,ids=tree.query(centers,k=p+1,workers=4)
                np.testing.assert_allclose(distances[:,0],0.,rtol=0,atol=1e-7)
                excluded=min(excluded,float(distances[:,-1].min()))
                if excluded<=halo:raise ValueError(f'{source_file}: {p} neighbors truncate {halo} A halo; excluded radius {excluded} A')
                result[start:start+len(centers)]=points[ids[:,:p]]-centers[:,None]
            result.flush();write_json(record,dict(signature=signature,minimum_excluded_radius_A=excluded,boundary_margin_A=float(margin)))
            print(f'[atomic_context] {source_file.name}: {len(coords)} unchanged centers, {p} points, excluded radius >= {excluded:.4f} A',flush=True)
        arrays.append(np.load(path,mmap_mode='r'))
    dataset._cache_sample_arrays=arrays
    dataset.num_points=p
