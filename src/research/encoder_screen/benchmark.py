"""Explicit CPU metric benchmark; no training or model-selection side effects."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from src.research.geoframe_evolution.evaluate import frame_metrics
from .common import load_config,write


def difference(a,b,path=''):
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Metric fields differ')
        return [v for k in a for v in difference(a[k],b[k],path+'.'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Metric lengths differ')
        return [v for i,(x,y) in enumerate(zip(a,b)) for v in difference(x,y,path+f'[{i}]')]
    if a is None:
        if b is not None:raise ValueError('Eligibility changed')
        return []
    return [(path,abs(float(a)-float(b)))]


def run(config):
    root=Path(config['output']);ref=Path(config['reference']);reuse=Path(config['reuse_geoframe'])
    manifest=json.loads((ref/'manifest.json').read_text());a=dict(np.load(ref/'frame-02.npz'))
    z=np.load(reuse/'technical/evaluations/epoch-034/frame-02.npz')['embeddings'][:,0]
    result={};values={};clusters={};pools=threadpool_info()
    for name,limit in [('unrestricted',None),('bounded',1)]:
        timings=[]
        with threadpool_limits(limits=limit):
            for _ in range(3):
                start=time.perf_counter();values[name],clusters[name]=frame_metrics(z,a,manifest['frames'][2]);timings.append(time.perf_counter()-start)
        result[name]=dict(seconds=timings,median_seconds=float(np.median(timings)))
    delta=difference(values['unrestricted'],values['bounded'])
    scalar=[(p,d) for p,d in delta if not ('confusion[' in p or 'counts[' in p)]
    max_delta=max(d for _,d in scalar)
    if max_delta>.002:raise ValueError(f'Threading changes a score beyond0.002: {sorted(scalar,key=lambda x:-x[1])[:10]}')
    np.testing.assert_array_equal(clusters['unrestricted'],clusters['bounded'])
    result.update(threadpools=pools,speedup=result['unrestricted']['median_seconds']/result['bounded']['median_seconds'],
        max_scalar_difference=max_delta,largest_differences=sorted(delta,key=lambda x:-x[1])[:12],
        cluster_assignments_exact=True,scope='CPU frame metrics only; fixed Al177 encoder checkpoint; same rows/probes/KMeans; no UMAP or inference',
        historical_exports_modified=False)
    write(root/'technical/benchmark.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);run(load_config(p.parse_args().config))
