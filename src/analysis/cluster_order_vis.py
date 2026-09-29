"""Three coordinated figures for saved-cluster order diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.result_records import file_hash, location
from .cluster_colors import _build_cluster_color_map, _cluster_label_color
from .cluster_geometry import _draw_edges, _set_equal_axes_3d
from .representative_style import radial_colors, sparse_geometry

MOTIFS=['FCC','HCP','BCC','ICO','Other']
MARKERS={0:'o',1:'o',2:'^',3:'s',4:'D'}
PLOTLY_MARKERS={0:'circle',1:'circle',2:'diamond',3:'square',4:'diamond'}


def display_geometry(record, mutual_neighbors):
    return sparse_geometry(record['points_A'],record['center_cutoff_A'],mutual_neighbors=mutual_neighbors)


def add_lines(figure, points, edges, colors, row, col, opacity=.78, width=2.6):
    if not edges:
        return
    xyz=[];line_colors=[]
    for i,j in edges:
        xyz.extend([points[i].tolist(),points[j].tolist(),[None,None,None]])
        line_colors.extend([mcolors.to_hex(.78*.5*(colors[i]+colors[j]))]*3)
    x,y,z=zip(*xyz)
    figure.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',showlegend=False,
        line=dict(color=line_colors,width=width),opacity=opacity,hoverinfo='skip'),row=row,col=col)


def representative_figure(root, records, colors, config, *, highlight):
    fig=plt.figure(figsize=(10.35,10.7),dpi=220,facecolor='white')
    interactive=make_subplots(rows=3,cols=3,
        specs=[[{'type':'scene'}]*3,[{'type':'scene'}]*3,[{'type':'scene'},None,None]],
        subplot_titles=[f'C{i+1}' for i in range(7)],horizontal_spacing=.025,vertical_spacing=.065)
    edge_receipts=[]
    for pos,record in enumerate(records):
        cid=record['cluster_id'];base=colors[cid]
        points,edges,receipt=display_geometry(record,config['context_mutual_neighbors'])
        edge_receipts.append(dict(cluster=cid,**receipt))
        point_colors=radial_colors(points,base)
        types=np.asarray(record['ptm_type']);matched=types!=0
        ax=fig.add_subplot(3,3,pos+1,projection='3d',facecolor='white')
        ax.set_proj_type('ortho');ax.view_init(elev=22,azim=38)
        row,col=pos//3+1,pos%3+1
        if highlight:
            emphasized=[(i,j) for i,j in edges if matched[i] and matched[j]]
            context=[(i,j) for i,j in edges if not (matched[i] and matched[j])]
            _draw_edges(ax,points,context,point_colors=point_colors,edge_alpha=.20,edge_linewidth=.75)
            _draw_edges(ax,points,emphasized,point_colors=point_colors,edge_alpha=.92,edge_linewidth=1.35)
            add_lines(interactive,points,context,point_colors,row,col,.20,1.7)
            add_lines(interactive,points,emphasized,point_colors,row,col,.92,3.2)
            counts={name:int(np.sum(types==i)) for i,name in enumerate(['Other','FCC','HCP','BCC','ICO'])}
            caption=' · '.join(f'{name} {counts[name]}' for name in MOTIFS[:-1] if counts[name]) or 'No template matches'
        else:
            _draw_edges(ax,points,edges,point_colors=point_colors,edge_alpha=.80,edge_linewidth=1.12)
            add_lines(interactive,points,edges,point_colors,row,col,.80,2.7)
            caption=''
        groups=list(np.unique(types)) if highlight else [None]
        for code in groups:
            mask=types==code if code is not None else np.ones(len(types),dtype=bool)
            is_order=highlight and code!=0
            alpha=.98 if not highlight or is_order else .25
            sizes=np.full(mask.sum(),64 if is_order else 58,dtype=float)
            lw=np.full(mask.sum(),1.1 if is_order else .36)
            if not highlight:
                sizes[0]=72;lw[0]=.8
            ax.scatter(*points[mask].T,c=point_colors[mask],s=sizes,alpha=alpha,
                edgecolors='#171717' if is_order else '#222222',linewidths=lw,
                marker=MARKERS[code] if code is not None else 'o',depthshade=False)
            atom_rows=np.asarray(record['source_atom_rows'])[mask]
            rmsd=np.asarray(record['ptm_rmsd'])[mask]
            names=np.array(['Other','FCC','HCP','BCC','ICO'])[types[mask]]
            interactive.add_trace(go.Scatter3d(x=points[mask,0],y=points[mask,1],z=points[mask,2],
                mode='markers',showlegend=False,
                marker=dict(color=[mcolors.to_hex(c) for c in point_colors[mask]],
                    size=7 if is_order else 6.3,opacity=alpha,
                    symbol=PLOTLY_MARKERS[code] if code is not None else 'circle',
                    line=dict(color='#222222',width=1.7 if is_order else .8)),
                customdata=np.column_stack((atom_rows,names,rmsd,np.linalg.norm(points[mask],axis=1))),
                hovertemplate=f'C{cid+1} · {record["frame"]} · sample {record["sample_index"]}<br>'
                    'atom=%{customdata[0]}<br>PTM=%{customdata[1]}<br>RMSD=%{customdata[2]}<br>r=%{customdata[3]} Å<extra></extra>'),row=row,col=col)
        _set_equal_axes_3d(ax,points);ax.set_axis_off()
        title_color=_cluster_label_color(base,darken_factor=.58)
        ax.set_title(f'C{cid+1}',fontsize=12,color=title_color,pad=2,fontweight='bold')
        if caption:
            ax.text2D(.5,.035,caption,ha='center',transform=ax.transAxes,fontsize=8.5,color=title_color)
        elev,azim=np.deg2rad([22,38])
        interactive.update_scenes(dict(xaxis=dict(visible=False,range=list(ax.get_xlim())),
            yaxis=dict(visible=False,range=list(ax.get_ylim())),zaxis=dict(visible=False,range=list(ax.get_zlim())),
            aspectmode='cube',bgcolor='white',camera=dict(projection=dict(type='orthographic'),
            eye=dict(x=1.8*np.cos(elev)*np.cos(azim),y=1.8*np.cos(elev)*np.sin(azim),z=1.8*np.sin(elev)))),row=row,col=col)
        interactive.layout.annotations[pos].update(text=f'<b>C{cid+1}</b>'+('<br>'+caption if caption else ''),
            font=dict(size=16,color=title_color))
    title='Detected local order' if highlight else 'Cluster representatives'
    basename='02_detected_local_order' if highlight else '01_cluster_representatives'
    fig.suptitle(title,fontsize=12,fontweight='bold',y=.975)
    if highlight:
        handles=[Line2D([],[],marker=MARKERS[i],color='none',markerfacecolor='white',
            markeredgecolor='#222222',markeredgewidth=1.2,label=name,markersize=7)
            for i,name in enumerate(['Other','FCC','HCP','BCC','ICO']) if i]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=4,frameon=False,fontsize=9)
        fig.text(.5,.025,'PTM on full snapshots · RMSD ≤ 0.10 · outlined atoms match a template; faint atoms do not',
            ha='center',fontsize=8.5)
    else:
        fig.text(.5,.025,'Distance from focal atom: pale center → dark outer atoms · sparse connections for display',
            ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.02,right=.988,bottom=.055,top=.895,wspace=.03,hspace=.09)
    folder=root/'data/representatives';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{basename}.png',bbox_inches='tight');plt.close(fig)
    interactive.update_layout(title=dict(text='<b>'+title+'</b>',x=.5,font=dict(size=22,color='black')),
        paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Arial, sans-serif'),height=1200,
        margin=dict(t=105,l=20,r=20,b=50))
    interactive.write_html(folder/f'{basename}.html',include_plotlyjs=True,
        config=dict(displaylogo=False,scrollZoom=True))
    return edge_receipts


def order_summary(root, colors, config):
    pooled=pd.read_csv(root/'tables/ptm_by_cluster.csv')
    frames=pd.read_csv(root/'tables/ptm_by_frame.csv')
    samples=pd.read_csv(root/'tables/local_order_samples.csv')
    frames=frames[np.isclose(frames.ptm_rmsd_cutoff,config['ptm_rmsd_cutoff'])]
    names=list(frames.frame.unique());cid_list=sorted(colors)
    fig,axes=plt.subplots(2,2,figsize=(12,9.2),dpi=200)
    fig.patch.set_facecolor('white')
    panel=make_subplots(rows=2,cols=2,subplot_titles=['PTM across all saved centers','Crystal-like centers by snapshot',
        'Local bond order: stratified sample','Neighbor bond-orientational coherence'],vertical_spacing=.15,horizontal_spacing=.12)
    ax=axes[0,0]
    for cid in cid_list:
        cells=pooled[pooled.cluster==cid].set_index('motif').loc[MOTIFS]
        for j,value in enumerate(cells.fraction):
            color=(1-value*.88)*np.ones(3)+value*.88*np.array(mcolors.to_rgb(colors[cid]))
            ax.add_patch(Rectangle((j-.5,cid-.5),1,1,facecolor=color,edgecolor='white',linewidth=2))
            ax.text(j,cid,f'{value*100:.1f}%',ha='center',va='center',fontsize=9,
                color='white' if value>.70 else '#222222')
        panel.add_trace(go.Heatmap(x=MOTIFS,y=[f'C{cid+1}'],z=[cells.fraction.to_list()],zmin=0,zmax=1,
            colorscale=[[0,'white'],[1,colors[cid]]],showscale=False,text=[[f'{v*100:.1f}%' for v in cells.fraction]],
            texttemplate='%{text}',xgap=2,ygap=2,hovertemplate='%{y} · %{x}: %{z:.2%}<extra></extra>'),row=1,col=1)
    ax.set(xlim=(-.5,4.5),ylim=(6.5,-.5),xticks=range(5),xticklabels=MOTIFS,
        yticks=cid_list,yticklabels=[f'C{i+1}' for i in cid_list],title='PTM across all saved centers')
    ax.tick_params(length=0);ax.spines[:].set_visible(False)
    for tick,cid in zip(ax.get_yticklabels(),cid_list,strict=True):tick.set_color(colors[cid]);tick.set_fontweight('bold')
    ax=axes[0,1]
    for cid in cid_list:
        subset=frames[(frames.cluster==cid)&frames.motif.isin(['FCC','HCP','BCC'])]
        fraction=subset.groupby('frame',sort=False).fraction.sum(min_count=1).reindex(names)
        totals=frames[(frames.cluster==cid)&(frames.motif=='Other')].set_index('frame').total.reindex(names)
        x=np.arange(len(names));ax.plot(x,fraction,color=colors[cid],marker='o',markersize=4,lw=1.6,label=f'C{cid+1}')
        small=(totals>0)&(totals<20)
        ax.scatter(x[small],fraction[small],s=40,facecolors='white',edgecolors=colors[cid],zorder=5)
        panel.add_trace(go.Scatter(x=names,y=fraction,mode='lines+markers',name=f'C{cid+1}',
            line=dict(color=colors[cid]),marker=dict(symbol=['circle-open' if v else 'circle' for v in small]),
            customdata=totals,hovertemplate='%{x}: %{y:.2%}<br>n=%{customdata}<extra>%{fullData.name}</extra>'),row=1,col=2)
    ax.set(xticks=range(len(names)),xticklabels=[n.replace('ps','') for n in names],ylim=(-.025,1.025),
        xlabel='Snapshot (ps)',ylabel='FCC + HCP + BCC fraction',title='Crystal-like centers by snapshot')
    ax.legend(ncol=4,fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(0,.90))
    ax.text(.98,.13,'Open markers: fewer than 20 centers',transform=ax.transAxes,ha='right',fontsize=8,color='#555555')
    ax=axes[1,0]
    for cid in cid_list:
        part=samples[samples.cluster==cid]
        ax.scatter(part.q4,part.q6,s=9,alpha=.38,color=colors[cid],linewidths=0)
        panel.add_trace(go.Scattergl(x=part.q4,y=part.q6,mode='markers',name=f'C{cid+1}',showlegend=False,
            marker=dict(size=4,color=colors[cid],opacity=.4),customdata=part[['frame','sample_index','ptm_type']],
            hovertemplate='q4=%{x:.3f}, q6=%{y:.3f}<br>%{customdata[0]} · sample %{customdata[1]} · %{customdata[2]}<extra></extra>'),row=2,col=1)
    ax.set(xlabel=r'$q_4$',ylabel=r'$q_6$',title=f'Local bond order: {len(samples):,} sampled neighborhoods')
    ax=axes[1,1]
    values=[samples.loc[samples.cluster==cid,'mean_q6_coherence'] for cid in cid_list]
    boxes=ax.boxplot(values,positions=cid_list,patch_artist=True,showfliers=False,widths=.52,
        medianprops=dict(color='#222222',lw=1.4),whiskerprops=dict(color='#777777'),capprops=dict(color='#777777'))
    for patch,cid,part in zip(boxes['boxes'],cid_list,values,strict=True):
        patch.set(facecolor=mcolors.to_rgba(colors[cid],.45),edgecolor=colors[cid],linewidth=1.2)
        panel.add_trace(go.Box(y=part,name=f'C{cid+1}',marker_color=colors[cid],boxpoints=False,showlegend=False),row=2,col=2)
    ax.set(xticks=cid_list,xticklabels=[f'C{cid+1}\nn={len(v)}' for cid,v in zip(cid_list,values,strict=True)],
        ylabel=r'Mean neighbor $q_6$ alignment',title='Neighbor bond-orientational coherence',ylim=(-.05,1.05))
    for tick,cid in zip(ax.get_xticklabels(),cid_list,strict=True):tick.set_color(colors[cid])
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    total=int(pooled[pooled.motif=='Other'].total.sum())
    fig.suptitle('Order across clusters and samples',fontsize=14,fontweight='bold',y=.98)
    fig.text(.5,.015,f'PTM: all {total:,} centers, RMSD ≤ 0.10. Local order: up to 64 samples per cluster and snapshot.\n'
        'Other = no accepted template; ICO = local fivefold order. Related neighborhoods; descriptive distributions.',
        ha='center',fontsize=9,linespacing=1.5)
    fig.tight_layout(rect=(0,.06,1,.945),h_pad=3,w_pad=3)
    folder=root/'data/order';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'03_order_across_samples.png');plt.close(fig)
    panel.update_layout(title=dict(text='<b>Order across clusters and samples</b>',x=.5),
        template='plotly_white',height=1000,font=dict(family='Arial, sans-serif'),
        legend=dict(orientation='h',y=1.1),margin=dict(t=120,b=65))
    panel.update_yaxes(autorange='reversed',row=1,col=1)
    panel.update_yaxes(title_text='FCC + HCP + BCC fraction',range=[-.025,1.025],row=1,col=2)
    panel.update_xaxes(title_text='q4',row=2,col=1);panel.update_yaxes(title_text='q6',row=2,col=1)
    panel.update_yaxes(title_text='Mean neighbor q6 alignment',range=[-.05,1.05],row=2,col=2)
    panel.write_html(folder/'03_order_across_samples.html',include_plotlyjs=True,config=dict(displaylogo=False))


def render(config):
    run=Path(config['source_run']);root=run/'analyses'/config['analysis_name']
    records=json.loads((root/'data/representatives.json').read_text())
    colors=_build_cluster_color_map(np.array([r['cluster_id'] for r in records]))
    edges=representative_figure(root,records,colors,config,highlight=False)
    representative_figure(root,records,colors,config,highlight=True)
    order_summary(root,colors,config)
    write_json(root/'technical/display.json',dict(producer_sha256=file_hash(__file__),
        context_mutual_neighbors=config['context_mutual_neighbors'],edges=edges,
        colors=colors,radial_color=dict(center_lighten=.96,edge_darken=.24,gamma=1.4,origin='focal atom'),
        point_selection='same 64 nearest source atoms in PNG and HTML',hull=False,
        highlighting='Full-source PTM atom matches; no claim of connected crystallite or orientation agreement'))
    from .publication import publish_bundle
    previous=json.loads((run/'run.json').read_text())
    protocol=json.loads((root/'technical/protocol.json').read_text())
    source_analysis=next(a for a in previous['analyses'] if a['id']==protocol['source_analysis_id'])
    metrics=json.loads((root/'data/metrics.json').read_text())
    analysis=publish_bundle(root/'data',run,name=config['analysis_name'],
        title=config.get('title','MACE EPI epoch 12')+' — local order',numerical_file='metrics.json',
        checkpoint_sha256=protocol['checkpoint_sha256'],refresh=True,
        context=f'Three views: sparse representatives, detected local order, and order across {metrics["total_centers"]:,} saved centers with {metrics["local_order_samples"]:,} stratified local-order samples. PTM uses full source snapshots. Descriptive analysis of related relaxed Al observations; Other is not a liquid label.',
        metadata=dict(protocol='saved-cluster-order-v1',inputs=protocol['inputs'],
            population=dict(centers=metrics['total_centers'],local_order_samples=metrics['local_order_samples'],
                source_analysis_id=protocol['source_analysis_id'],frames=protocol['frames']),
            selection=dict(checkpoint=source_analysis['selection'],
                representatives='Original saved representatives; unchanged selection',local_order=protocol['sampling'])),
        stages=dict(full_source_ptm=dict(state='complete',evidence=location(root/'data/metrics.json')),
            sampled_local_order=dict(state='complete',evidence=location(root/'tables/local_order_samples.csv'))))
    entry=run/'index.html'
    page=entry.read_text();marker=f'<!-- {config["analysis_name"]} -->'
    if marker not in page:
        page+=marker+f'<p><a href="analyses/{config["analysis_name"]}/index.html"><strong>New: representatives and detailed local-order analysis</strong></a></p>'
        entry.write_text(page)
    # Keep one current representative view in the galleries, while retaining the
    # old images at their recorded producer paths and with their original hashes.
    record=json.loads((run/'run.json').read_text())
    for bundle in record['analyses']:
        if bundle['id'] != protocol['source_analysis_id']:
            continue
        for artifact in bundle['artifacts']:
            path=Path(artifact['relative'])
            if (path.parent == Path('real_md/representatives') and
                path.stem == f'04_cluster_representatives_k{len(records)}' and
                path.suffix in ('.png','.html')):
                artifact['superseded_by']=location(root/'data/representatives'/f'01_cluster_representatives{path.suffix}')
    write_json(run/'run.json',record)
    from .publication import refresh_publication_record
    refresh_publication_record(run/'run.json')
    print(json.dumps(dict(page=str(root/'index.html'),edges=edges,analysis_id=analysis['id']),indent=2))
