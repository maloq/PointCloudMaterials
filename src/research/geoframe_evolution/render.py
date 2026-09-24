"""Redraw saved spatial fields with discrete named legends; verify cluster counts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from sklearn.cluster import KMeans

from .reference import write_json


def run(output):
    root=Path(output);ref=root/'technical/reference'
    manifest=json.loads((ref/'manifest.json').read_text())
    records=[]
    for path in sorted((root/'plots').glob('*-frame-*-spatial.png')):
        name,index=path.stem.rsplit('-frame-',1);i=int(index.split('-')[0])
        record=manifest['frames'][i];a=np.load(ref/f'frame-{i:02d}.npz');n=record['anchor_count']
        folder=root/'technical/evaluations'/name
        z=np.load(folder/f'frame-{i:02d}.npz')['embeddings'][:n,1]
        clusters=KMeans(n_clusters=7,n_init=5,random_state=20260923).fit(z[a['split'][:n]==0]).predict(z)
        test=a['split'][:n]==1
        count=[[int(((clusters==k)&(a['context'][:n,1]==j)&test).sum()) for j in range(7)] for k in range(7)]
        old=json.loads((folder/'metrics.json').read_text())[f'frame_{i:02d}_{record["material"]}_projector']['cluster_context_counts']
        np.testing.assert_array_equal(count,old)
        xyz=a['coords'][:n];slab=np.abs(xyz[:,2]-np.median(xyz[:,2]))<np.ptp(xyz[:,2])*.12
        ki=0 if record['material']=='Al' else 1
        fields=[a['ptm'][:n,1],a['context'][:n,1],clusters,a['order'][:n,ki,4]]
        labels=[['Other','FCC','HCP','BCC','ICO'],
                ['Liquid other','Crystal interior','Mixed boundary','Al planar fault','Internal defect','Five-fold proxy','Ordered liquid'],
                [f'C{k+1}' for k in range(7)],None]
        palettes=[['#bbbbbb','#e7b800','#8a55be','#187bc4','#cc5cba'],
                  ['#bbbbbb','#e7b800','#00a39b','#8a55be','#735240','#cc5cba','#187bc4'],
                  list(plt.get_cmap('tab10').colors[:7]),None]
        fig,axes=plt.subplots(2,2,figsize=(14,11),constrained_layout=True)
        for ax,values,names,colors,title in zip(axes.flat,fields,labels,palettes,
                    ['PTM motif','Independent context proxy','Embedding K=7 (arbitrary IDs)','Averaged bond order']):
            style=dict(cmap='viridis') if names is None else dict(cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(len(names)+1)-.5,len(names)))
            sc=ax.scatter(xyz[slab,0],xyz[slab,1],c=values[slab],s=7,**style)
            ax.set(title=title,xlabel='x (Å)',ylabel='y (Å)',aspect='equal')
            cb=fig.colorbar(sc,ax=ax,shrink=.75)
            if names is not None:cb.set_ticks(range(len(names)),labels=names)
        fig.suptitle(f'{name}: {record["material"]} {Path(record["file"]).stem}\nFixed sampled slab; candidate labels do not establish future nucleation')
        fig.savefig(path,dpi=150);plt.close(fig)
        records.append(dict(file=path.name,cluster_counts_exactly_match_export=True))
    write_json(root/'technical/spatial-rendering.json',dict(
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=records,
        note='Readability-only redraw: explicit category legends; cluster contingency verified unchanged. Metric exports preserved.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)
