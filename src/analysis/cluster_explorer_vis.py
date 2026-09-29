"""Offline linked explorer and print figures in the established cluster palette."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.result_records import file_hash, location
from .cluster_colors import _cluster_palette
from .cluster_order_vis import radial_colors, display_geometry
from .cluster_geometry import _draw_edges, _set_equal_axes_3d


def packed(values):
    array=np.asarray(values)
    integer=array.dtype.kind in 'iub'
    array=array.astype('<i4' if integer else '<f4')
    return dict(dtype='i4' if integer else 'f4',shape=list(array.shape),
        base64=base64.b64encode(array.tobytes()).decode('ascii'))


def geometry(data, index):
    points=data['points'][index]
    radii=np.sort(np.linalg.norm(points,axis=1))
    return display_geometry(dict(points_A=points,center_cutoff_A=float(1.2*np.median(radii[1:13]))),2)


def draw_patch(ax, data, index, colors):
    points,edges,_=geometry(data,index);cid=int(data['clusters'][index])
    shade=radial_colors(points,colors[cid]);ax.set_proj_type('ortho');ax.view_init(elev=22,azim=38)
    _draw_edges(ax,points,edges,point_colors=shade,edge_alpha=.8,edge_linewidth=.75)
    ax.scatter(*points.T,c=shade,s=18,edgecolors='#242424',linewidths=.35,depthshade=False)
    _set_equal_axes_3d(ax,points);ax.set_axis_off()


def print_figures(root, data, atlas, profiles, colors, c):
    ids={int(v):i for i,v in enumerate(data['sample_index'])}
    folder=root/'data/atlas';folder.mkdir(parents=True,exist_ok=True)
    for cid in sorted({r['cluster'] for r in atlas}):
        rows=[r for r in atlas if r['cluster']==cid]
        fig=plt.figure(figsize=(10,8.5),dpi=180,facecolor='white')
        for j,record in enumerate(rows):
            ax=fig.add_subplot(3,4,j+1,projection='3d')
            index=ids[record['sample_index']];draw_patch(ax,data,index,colors)
            ax.set_title(f"{record['band'].capitalize()} · alignment {record['mean_q6_coherence']:.2f}",fontsize=9,color=colors[cid],pad=0)
            ax.text2D(.5,-.015,f"{record['frame']} · #{record['sample_index']}",transform=ax.transAxes,ha='center',fontsize=7,color='#555555')
        fig.suptitle(f"{c['title']}\nC{cid+1} · variation within the cluster",fontsize=13,color=colors[cid])
        fig.text(.5,.017,'Rank thirds of neighbor q6 alignment · four samples per band · pale center → dark outer atoms',ha='center',fontsize=8)
        fig.subplots_adjust(top=.89,bottom=.065,left=.015,right=.985,wspace=.01,hspace=.14)
        fig.savefig(folder/f'C{cid+1}_order_atlas.png',bbox_inches='tight')
        if c['svg']:fig.savefig(folder/f'C{cid+1}_order_atlas.svg',bbox_inches='tight')
        plt.close(fig)
    folder=root/'data/radial';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),dpi=190)
    for ax,measure,title in zip(axes,['crystalline_fraction','q6_alignment'],['Crystal fraction around the center','Orientational coherence with the center'],strict=True):
        for cid in sorted(profiles.cluster.unique()):
            rows=profiles[(profiles.cluster==cid)&(profiles.measure==measure)]
            x=(rows.inner_A+rows.outer_A)/2
            ax.fill_between(x,rows.q25,rows.q75,color=colors[cid],alpha=.10,linewidth=0)
            ax.plot(x,rows['mean'],color=colors[cid],marker='o',markersize=3,lw=1.6,label=f'C{cid+1}')
        ax.set(title=title,xlabel='Distance from focal atom (Å)',ylim=(-.12,1.05))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#eeeeee')
    axes[0].set_ylabel('FCC + HCP + BCC fraction');axes[1].set_ylabel('Mean normalized q6 inner product')
    axes[0].legend(frameon=False,ncol=4,fontsize=8)
    fig.suptitle(c['title'],fontsize=12)
    fig.text(.5,.015,'Full-source shells · equal weight per sampled neighborhood · bands: neighborhood IQR, not confidence intervals',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(folder/'radial_order_profiles.png')
    if c['svg']:fig.savefig(folder/'radial_order_profiles.svg')
    plt.close(fig)


def render(c):
    run=Path(c['source_run']);root=run/'analyses'/c['analysis_name']
    protocol=json.loads((root/'technical/protocol.json').read_text())
    if file_hash(root/'data/inspection.npz')!=protocol['inspection_sha256']:raise ValueError('Changed inspection data')
    with np.load(root/'data/inspection.npz') as a:data={k:a[k] for k in a.files}
    atlas=json.loads((root/'data/atlas.json').read_text());profiles=pd.read_csv(root/'tables/radial_profiles.csv')
    retrieval=pd.read_csv(root/'tables/embedding_neighbors.csv',dtype={'frame':str})
    colors=_cluster_palette(int(data['clusters'].max())+1)
    print_figures(root,data,atlas,profiles,colors,c)
    ids={int(v):i for i,v in enumerate(data['sample_index'])}
    important={r['sample_index'] for r in atlas}|set(retrieval.sample_index)
    # PCA is for display only, never retrieval or physical measurements.
    oriented={str(ids[int(i)]):geometry(data,ids[int(i)])[0].round(5).tolist() for i in important}
    payload=dict(title=c['title'],colors=colors,frames=protocol['frames'],atlas=atlas,
        retrieval=json.loads(retrieval.to_json(orient='records')),profiles=json.loads(profiles.to_json(orient='records')),
        metrics=json.loads((root/'data/metrics.json').read_text()),oriented=oriented,
        arrays={k:packed(v) for k,v in data.items()},checkpoint=protocol['checkpoint_sha256'])
    template=Path(__file__).with_name('cluster_explorer.html').read_text()
    rendered=template.replace('__TITLE__',html.escape(c['title'])).replace('__PLOTLY__',get_plotlyjs()).replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')))
    page=root/'data/explorer.html';page.write_text(rendered)
    write_json(root/'technical/display.json',dict(producer_sha256=file_hash(__file__),template_sha256=file_hash(Path(__file__).with_name('cluster_explorer.html')),
        offline=True,svg=c['svg'],colors=colors,hull=False,connections='Adaptive cutoff 1.2×median focal 12-neighbor distance; all edges inside focal shell and mutual-2NN outer edges. Display only.',
        orientation='PCA for atlas/retrieval cards; source orientation for arbitrary clicked samples.',
        radial_color='pale center to dark outer atoms; physical distance normalized by 95th percentile, gamma 1.4'))
    from .publication import publish_bundle
    analysis=publish_bundle(root/'data',run,name=c['analysis_name'],title=c['title']+' — linked exploration',
        numerical_file='metrics.json',checkpoint_sha256=protocol['checkpoint_sha256'],refresh=True,
        context='Offline linked spatial views, order-versus-template scatter, 12-neighborhood cluster atlases, radial order profiles, and exact native-128D embedding neighbors with coarse-order-matched controls. Open explorer.html to interact.',
        metadata=dict(protocol='static-cluster-exploration-v1',inputs=protocol['inputs'],
            population=dict(frames=protocol['frames'],sampling=protocol['sampling'],source_analysis_id=protocol['source_analysis_id'],order_analysis_id=protocol['order_analysis_id']),
            selection=dict(atlas=protocol['atlas'],controls=protocol['controls'])),
        stages=dict(exploration=dict(state='complete',evidence=location(page))))
    entry=run/'index.html';text=entry.read_text();marker=f'<!-- {c["analysis_name"]} -->'
    if marker not in text:
        link=marker+f'<p><a href="analyses/{c["analysis_name"]}/data/explorer.html"><strong>Open linked structure explorer</strong></a> · spatial slices, order, atlas, radial profiles and embedding neighbors</p>'
        text=text.replace('</body>',link+'</body>') if '</body>' in text else text+link
        entry.write_text(text)
    print(json.dumps(dict(page=str(page),analysis_id=analysis['id']),indent=2))
