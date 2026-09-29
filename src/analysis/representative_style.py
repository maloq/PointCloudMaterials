"""Shared sparse, hull-free representative style for static and interactive views."""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cluster_colors import _cluster_label_color
from .cluster_geometry import _draw_edges, _orient_points_for_crystal_view, _set_equal_axes_3d


def radial_colors(points, base):
    radius=np.linalg.norm(points,axis=1)
    t=np.clip(radius/np.percentile(radius,95),0,1)**1.4
    rgb=np.asarray(mcolors.to_rgb(base))
    pale=rgb+.96*(1-rgb);dark=rgb*.76
    return pale[None,:]*(1-t[:,None])+dark[None,:]*t[:,None]


def sparse_geometry(points, cutoff, *, mutual_neighbors=2, orientation='pca'):
    points=np.asarray(points)
    distance=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    valid=(distance<=cutoff)&(distance>1e-9)
    np.fill_diagonal(distance,np.inf)
    order=np.argsort(distance,axis=1)[:,:mutual_neighbors]
    directed=np.zeros_like(valid)
    directed[np.arange(len(points))[:,None],order]=True
    core=np.linalg.norm(points,axis=1)<=cutoff
    keep=valid & ((core[:,None]&core[None,:]) | (directed&directed.T))
    edges=[tuple(map(int,pair)) for pair in np.argwhere(np.triu(keep,1))]
    oriented,basis=_orient_points_for_crystal_view(points,method=orientation)
    return oriented,edges,dict(edges=len(edges),previous_full_cutoff_edges=int(np.triu(valid,1).sum()),
        core_atoms=int(core.sum()),orientation=basis)


def render_representatives(records, out_file, *, title='Cluster representatives',
                           orientation='pca', view_elev=22., view_azim=38., projection='ortho'):
    """One PNG/HTML pair, identical points/edges/camera, actual cluster IDs."""
    if not records:raise ValueError('No representative records to render')
    if projection not in ('ortho','persp'):raise ValueError(f'Unknown projection: {projection}')
    columns=min(3,len(records));rows=(len(records)+columns-1)//columns
    specs=[[{'type':'scene'} if r*columns+c<len(records) else None for c in range(columns)] for r in range(rows)]
    interactive=make_subplots(rows=rows,cols=columns,specs=specs,
        subplot_titles=[f'C{r["cluster_id"]+1}' for r in records],horizontal_spacing=.025,vertical_spacing=.065)
    figure=plt.figure(figsize=(3.45*columns,3.5*rows+.45),dpi=220,facecolor='white');receipts=[]
    for pos,record in enumerate(records):
        cid=int(record['cluster_id']);base=record['base_color'];row,col=pos//columns+1,pos%columns+1
        points,edges,receipt=sparse_geometry(record['points'],record['cutoff'],orientation=orientation)
        colors=radial_colors(points,base);radius=np.linalg.norm(points,axis=1)
        ax=figure.add_subplot(rows,columns,pos+1,projection='3d',facecolor='white')
        ax.set_proj_type(projection);ax.view_init(elev=view_elev,azim=view_azim)
        _draw_edges(ax,points,edges,point_colors=colors,edge_alpha=.80,edge_linewidth=1.12)
        sizes=np.full(len(points),58.);sizes[0]=72
        widths=np.full(len(points),.36);widths[0]=.8
        ax.scatter(*points.T,c=colors,s=sizes,alpha=.98,edgecolors='#222222',linewidths=widths,depthshade=False)
        _set_equal_axes_3d(ax,points);ax.set_axis_off()
        label_color=_cluster_label_color(base,darken_factor=.58)
        ax.set_title(f'C{cid+1}',fontsize=12,color=label_color,pad=2,fontweight='bold')
        caption=f'sample {record["sample_index"]}'
        ax.text2D(.5,.0,caption,ha='center',transform=ax.transAxes,fontsize=8,color='#666666')
        if edges:
            xyz=[];line_colors=[]
            for i,j in edges:
                xyz.extend([points[i].tolist(),points[j].tolist(),[None,None,None]])
                line_colors.extend([mcolors.to_hex(.78*.5*(colors[i]+colors[j]))]*3)
            x,y,z=zip(*xyz)
            interactive.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='lines',showlegend=False,
                line=dict(color=line_colors,width=2.7),opacity=.80,hoverinfo='skip'),row=row,col=col)
        interactive.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',showlegend=False,
            marker=dict(color=[mcolors.to_hex(c) for c in colors],size=[7.3]+[6.3]*(len(points)-1),
                line=dict(color='#222222',width=.8)),customdata=np.column_stack((np.arange(len(points)),radius)),
            hovertemplate=f'C{cid+1} · {caption}<br>display atom=%{{customdata[0]}}<br>'
                f'r=%{{customdata[1]:.3f}} {record["units"]}<extra></extra>'),row=row,col=col)
        elev,azim=np.deg2rad([view_elev,view_azim])
        interactive.update_scenes(dict(xaxis=dict(visible=False,range=list(ax.get_xlim())),
            yaxis=dict(visible=False,range=list(ax.get_ylim())),zaxis=dict(visible=False,range=list(ax.get_zlim())),
            aspectmode='cube',bgcolor='white',camera=dict(projection=dict(type='orthographic' if projection=='ortho' else 'perspective'),
                eye=dict(x=1.8*np.cos(elev)*np.cos(azim),y=1.8*np.cos(elev)*np.sin(azim),z=1.8*np.sin(elev)))),row=row,col=col)
        interactive.layout.annotations[pos].update(text=f'<b>C{cid+1}</b><br><span style="font-size:11px">{caption}</span>',
            font=dict(size=16,color=label_color))
        receipts.append(dict(cluster_id=cid,sample_index=int(record['sample_index']),num_points_plotted=len(points),
            cutoff=float(record['cutoff']),cutoff_source=record['cutoff_source'],units=record['units'],
            edge_info=receipt,orientation=receipt['orientation'],edge_indices=edges,
            plotted_points=points.tolist(),point_colors=[mcolors.to_hex(c) for c in colors]))
    figure.suptitle(title,fontsize=12,fontweight='bold',y=.975)
    figure.text(.5,.025,'Distance from focal atom: pale center → dark outer atoms · sparse connections for display',ha='center',fontsize=8.5)
    figure.subplots_adjust(left=.02,right=.988,bottom=.07,top=.895,wspace=.03,hspace=.09)
    out_file=Path(out_file);out_file.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(out_file,bbox_inches='tight');plt.close(figure)
    interactive.update_layout(title=dict(text='<b>'+title+'</b>',x=.5,font=dict(size=22,color='black')),
        paper_bgcolor='white',plot_bgcolor='white',font=dict(family='Arial, sans-serif'),height=380*rows+60,
        margin=dict(t=110,l=20,r=20,b=50))
    interactive.write_html(out_file.with_suffix('.html'),include_plotlyjs=True,config=dict(displaylogo=False,scrollZoom=True))
    return dict(out_file=str(out_file),interactive_file=str(out_file.with_suffix('.html')),
        variant_name='sparse_first_shell_mutual2',edge_method='first_shell_cutoff_and_outer_mutual_2nn',
        orientation_method=orientation,view_elev=view_elev,view_azim=view_azim,projection=projection,
        hull=False,representatives=receipts)
