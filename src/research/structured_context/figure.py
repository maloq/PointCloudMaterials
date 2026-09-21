"""PNG illustration of the query geometry and actual MD atom assignments."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.project_runtime.paths import resolve_path,dataset_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from .geometry import stencil


def render(plan,source):
    config=plan['structured_config'];root=resolve_path(config['output']);folder=root/'plots';folder.mkdir(exist_ok=True)
    q=stencil(config['shell_radii_A'])
    r=np.load(resolve_path(config['context_cache'])/str(source['id'])/'relative.npy')[0,0]
    raw=ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    row=np.searchsorted(raw.atom_ids,source['center_atom_ids'][0]);box=raw.box_high[0]-raw.box_low[0]
    x=raw.positions[0].astype(float)-raw.positions[0,row].astype(float);x-=box*np.round(x/box)
    x=x[np.linalg.norm(x,axis=1)<25]
    colors=['#252525']+['#0072B2']*12+['#D55E00']*12
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':300})
    fig=plt.figure(figsize=(7.1,3.5),layout='constrained')
    for i,title in enumerate(('Symmetric queries','Real atomic neighborhoods')):
        ax=fig.add_subplot(1,2,i+1,projection='3d');ax.set_title(title,pad=0)
        if i:
            ax.scatter(*x.T,s=1.5,c='#AAAAAA',alpha=.14,depthshade=False,rasterized=True)
            for a,b in zip(q,r,strict=True):ax.plot(*np.stack((a,b)).T,c='#333333',lw=.5,alpha=.7)
            ax.scatter(*q.T,s=25,facecolors='none',edgecolors=colors,linewidths=.65,depthshade=False)
            ax.scatter(*r.T,s=18,c=colors,depthshade=False)
        else:
            ax.scatter(*q.T,s=24,c=colors,depthshade=False)
            for shell in (q[1:13],q[13:]):
                for a in range(12):
                    for b in range(a):
                        if np.isclose(np.linalg.norm(shell[a]-shell[b]),np.linalg.norm(shell[a]),rtol=1e-5):
                            ax.plot(*shell[[a,b]].T,c='#777777',lw=.55,alpha=.55)
        ax.scatter(0,0,0,c='black',s=55,marker='*',depthshade=False)
        ax.view_init(elev=19,azim=31);ax.set_box_aspect((1,1,1));ax.set_axis_off()
        ax.set_xlim(-25,25);ax.set_ylim(-25,25);ax.set_zlim(-25,25)
    fig.savefig(folder/'symmetric-context.png',facecolor='white');plt.close(fig)
    (folder/'README.md').write_text(
        '# Symmetric context illustration\n\n'
        f'Existing Al source {source["id"]}, frame 0, tracked atom {source["center_atom_ids"][0]}. '
        'Blue/orange: 12 query directions at 10/20 Å. Black star: tracked center. '
        'Left: exactly symmetric cuboctahedral queries. Right: the same queries (open markers), '
        'assigned real atom centers (filled markers), assignment offsets (short lines), '
        'and actual nearby MD atoms (gray). The encoder uses full local neighborhoods around '
        'these real centers; the faint 25 Å cloud is an illustration of context locations, '
        'not a truncation of the encoder input. Nominal queries and actual offsets both enter '
        'the predictor. The figure is a perspective projection of three-dimensional geometry.\n')
