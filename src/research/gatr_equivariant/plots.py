"""Standalone scientific figures and offline 3D direction explorer."""
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import direction_pair, unit_vectors
from .report import PRIMARY, get_vectors

DISPLAY = {
    'block1/point_numerator':'Block 1 · point triplet',
    'block2_mlp_input/point_numerator':'Before last MLP · point triplet',
    'block2_mlp_input/ideal_bivector':'Before last MLP · ideal bivector',
    'block2_mlp_input/axial_bivector':'Before last MLP · axial bivector',
    'centroid_7A':'Density dipole · 7 Å',
    'centroid_support':'Density dipole · full support',
    'shape_axis_7A':'Shape principal axis · 7 Å',
}
COLORS = ['#4e5ce6','#00998c','#c363d4','#dc7e36','#555555','#999999']


def lookup(table, field, metric):
    row = table[(table.field==field)&(table.metric==metric)]
    if len(row)!=1:
        raise ValueError(f'Expected one summary row: {field}, {metric}')
    return row.iloc[0]


def save(fig,root,name):
    fig.savefig(root/'plots'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(root/'plots'/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def make_plots(root,tables,items,fields):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    ts = pd.DataFrame(tables['temporal_summary']); lag = pd.DataFrame(tables['lag_summary'])
    spatial = pd.DataFrame(tables['spatial_summary'])
    selected = list(DISPLAY)[:6]
    color_by_field = dict(zip(selected,COLORS,strict=True))
    fig,axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for i,field in enumerate(selected):
        row = lookup(ts,field,'mean_angle_deg')
        axes[0,0].barh(i,row['mean'],color=COLORS[i],alpha=.85)
        axes[0,0].errorbar(row['mean'],i,xerr=[[row['mean']-row.low],[row.high-row['mean']]],color='#222',capsize=3)
        curve = lag[(lag.field==field)&(lag.metric=='p1')].sort_values('lag_ps')
        axes[0,1].plot(curve.lag_ps,curve['mean'],'o-',ms=3,color=COLORS[i],label=DISPLAY[field])
        axes[0,1].fill_between(curve.lag_ps,curve.low,curve.high,color=COLORS[i],alpha=.1)
    axes[0,0].set_yticks(range(len(selected)),[DISPLAY[f] for f in selected]); axes[0,0].invert_yaxis()
    axes[0,0].axvline(90,color='black',ls=':',lw=1); axes[0,0].set_xlim(0,100)
    axes[0,0].set(title='Direction changes at 0.75 ps',xlabel='Mean turn (degrees); 90° = independent isotropic directions')
    axes[0,1].set_xscale('log'); axes[0,1].axhline(0,color='#888',lw=.8)
    axes[0,1].set(title='Directional memory',xlabel='Time separation (ps)',ylabel='P1 = mean direction dot product',ylim=(-.15,1))
    for i,field in enumerate([PRIMARY,'block1/point_numerator','centroid_7A','centroid_support']):
        for ax,metric in [(axes[1,0],'excess_p1'),(axes[1,1],'excess_p2')]:
            curve = spatial[(spatial.field==field)&(spatial.metric==metric)].sort_values('distance_lo_A')
            x = (curve.distance_lo_A+curve.distance_hi_A)/2
            ax.plot(x,curve['mean'],'o-',ms=4,color=color_by_field[field],label=DISPLAY[field])
            ax.fill_between(x,curve.low,curve.high,color=color_by_field[field],alpha=.13)
            ax.axhline(0,color='#888',lw=.7); ax.set_xlabel('Periodic center separation (Å)')
    axes[1,0].set(title='Neighbor direction alignment',ylabel='P1 − phase-matched shuffle expectation')
    axes[1,1].set(title='Neighbor axis alignment (sign ignored)',ylabel='P2 − phase-matched shuffle expectation')
    axes[1,0].legend(fontsize=8,loc='best')
    fig.suptitle('Frozen GATr: rotation-covariant directions along Al trajectories\n40 tracks · 10 held-out sources · 70 spatial snapshots · A100 / node07',fontsize=15)
    save(fig,root,'overview')

    examples = [next((s,a) for s,a in items if s['split']=='test' and s['temperature_K']==t) for t in (400,450,520)]
    fig,axes = plt.subplots(4,3,figsize=(15,10),sharex='col',layout='constrained')
    for col,(source,a) in enumerate(examples):
        v = get_vectors(a,fields[PRIMARY]).reshape(len(a['frames']),len(a['centers']),3)[:,0]
        u,n,valid = unit_vectors(v,fields[PRIMARY]['threshold']); u[~valid] = np.nan
        t = a['times_ps']; t = t.reshape(len(a['frames']),-1)[:,0] if t.ndim>1 else t
        for k,label in enumerate('xyz'):
            axes[0,col].plot(t,u[:,k],lw=.7,alpha=.8,label=label)
        axes[0,col].set_title(f'{source["temperature_K"]:g} K · source {source["id"]} · atom {a["centers"][0]}')
        axes[0,col].set_ylim(-1.1,1.1)
        axes[1,col].plot(t,n/fields[PRIMARY]['rms'],lw=.8,color=COLORS[0])
        axes[1,col].axhline(.1,color='#999',ls=':')
        _,angle,_ = direction_pair(v[:-1],v[1:],fields[PRIMARY]['threshold'])
        axes[2,col].plot(t[1:],angle,lw=.65,color=COLORS[3]); axes[2,col].axhline(90,color='#999',ls=':')
        order = a['order'].reshape(len(a['frames']),len(a['centers']),-1)
        axes[3,col].plot(t,order[:,0,4],color=COLORS[1],lw=1)
        axes[3,col].set_xlabel('Time (ps)')
    axes[0,0].legend(ncol=3,fontsize=8); axes[0,0].set_ylabel('Unit direction components')
    axes[1,0].set_ylabel('Norm / training RMS'); axes[2,0].set_ylabel('Adjacent turn (degrees)')
    axes[3,0].set_ylabel('Physical order q̄6')
    fig.suptitle(f'Unsmoothed direction traces · before final MLP · point triplet, channel {fields[PRIMARY]["channel"]}\nFirst preselected track per shown temperature; gaps mark weak directions',fontsize=14)
    save(fig,root,'trajectories')

    fig,axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    sensitivity = pd.DataFrame(tables['sensitivity_summary'])
    for ax,metric,title in zip(axes,('coverage','mean_angle_deg','flip90_fraction'),
        ('Valid adjacent pairs','Mean turn (degrees)','Turns above 90°'),strict=True):
        sub = sensitivity[sensitivity.metric==metric]
        ax.plot(sub.threshold_fraction,sub['mean'],'o-',color=COLORS[0])
        ax.fill_between(sub.threshold_fraction,sub.low,sub.high,color=COLORS[0],alpha=.15)
        ax.set(xlabel='Minimum norm / training RMS',title=title)
    fig.suptitle('Are angular jumps explained by weak vectors?',fontsize=14)
    save(fig,root,'norm-sensitivity')

    # Real-space 3D panels with fixed-length arrows: amplitude is shown separately.
    source = examples[1][0]
    fig = plt.figure(figsize=(13,5),layout='constrained')
    for i,frame in enumerate((0,800)):
        a = dict(np.load(root/'technical/spatial'/f'{source["id"]}-{frame:04d}.npz'))
        take = a['patch']==0
        p = a['positions'][take]-a['patch_anchors'][0]; p -= a['box']*np.round(p/a['box'])
        u,n,ok = unit_vectors(get_vectors(a,fields[PRIMARY])[take],fields[PRIMARY]['threshold'])
        ax = fig.add_subplot(1,2,i+1,projection='3d')
        ax.scatter(*p.T,c=a['order'][take,4],cmap='viridis',vmin=0,vmax=.6,s=10,alpha=.6)
        ax.quiver(*p[ok].T,*u[ok].T,length=2,color='#3d52b9',linewidth=.6,arrow_length_ratio=.35)
        ax.set(xlabel='x (Å)',ylabel='y (Å)',zlabel='z (Å)',title=f'450 K · source {source["id"]} · {float(a["time_ps"]):g} ps')
        ax.set_box_aspect((1,1,1))
    fig.suptitle('Neighboring local directions · fixed-length arrows; atom color = q̄6',fontsize=14)
    save(fig,root,'spatial-directions')


def make_explorer(root,items,fields,config):
    from plotly.offline import get_plotlyjs
    payload = dict(field=PRIMARY,channel=fields[PRIMARY]['channel'],tracks=[],panels=[])
    for source,a in items:
        if source['split']!='test': continue
        nf,nc = len(a['frames']),len(a['centers'])
        v = get_vectors(a,fields[PRIMARY]).reshape(nf,nc,3)
        u,n,ok = unit_vectors(v,fields[PRIMARY]['threshold'])
        for c in range(nc):
            _,angle,valid = direction_pair(v[:-1,c],v[1:,c],fields[PRIMARY]['threshold'])
            payload['tracks'].append(dict(name=f'{source["temperature_K"]:g} K / source {source["id"]} / atom {a["centers"][c]}',
                time=a['times_ps'].tolist(),u=np.round(u[:,c],5).tolist(),valid=ok[:,c].tolist(),
                norm=np.round(n[:,c]/fields[PRIMARY]['rms'],5).tolist(),
                angle=[None]+[round(float(x),3) if valid[k] else None for k,x in enumerate(angle)],
                q6=np.round(a['order'].reshape(nf,nc,-1)[:,c,4],5).tolist()))
        for frame in config['spatial_frames']:
            a = dict(np.load(root/'technical/spatial'/f'{source["id"]}-{frame:04d}.npz'))
            for patch in (0,1):
                take = a['patch']==patch
                p = a['positions'][take]-a['patch_anchors'][patch]; p -= a['box']*np.round(p/a['box'])
                u,n,ok = unit_vectors(get_vectors(a,fields[PRIMARY])[take],fields[PRIMARY]['threshold'])
                payload['panels'].append(dict(name=f'{source["temperature_K"]:g} K / source {source["id"]} / {float(a["time_ps"]):g} ps / patch {patch}',
                    p=np.round(p,4).tolist(),u=np.round(u,5).tolist(),valid=ok.tolist(),
                    norm=np.round(n/fields[PRIMARY]['rms'],4).tolist(),q6=np.round(a['order'][take,4],5).tolist(),
                    ids=a['atom_ids'][take].tolist(),phase=a['labels'][take].tolist()))
    template = '''<!doctype html><html><head><meta charset="utf-8"><title>GATr directional audit</title>
<style>body{font:16px system-ui;margin:25px auto;max-width:1250px;color:#223}select{padding:8px;max-width:100%}.note{color:#556;line-height:1.6}.grid{display:grid;grid-template-columns:1fr 1fr}@media(max-width:800px){.grid{display:block}}h1{font-size:29px}</style>
<script>__PLOTLY__</script></head><body><h1>GATr: directions in time and space</h1>
<p class="note">Frozen checkpoint 1216 · A100 / node07. Geometric triplet before the final MLP,
channel __CHANNEL__, selected using training norms. Arrows show direction, not predicted velocity or crystal orientation.
Weak directions (&lt;0.1 × training RMS) are masked. This last-layer direction has no detectable effect on z under erasure;
the report also measures earlier layers. <a href="RESULTS.md">Findings</a> · <a href="plots/overview.png">Summary figure</a></p>
<label>Tracked atom <select id="track"></select></label><div id="temporal" style="height:670px"></div>
<div class="grid"><div><label>Spatial snapshot <select id="panel"></select></label><p class="note">Drag to rotate; scroll to zoom.
Each arrow is a separate atom-centered local encoding. Color = physical q̄6. Shared neighborhoods can create spatial alignment.</p>
<div id="space" style="height:570px"></div></div><div><h2>Direction on the unit sphere</h2><p class="note">Same temporal track; color follows time.
These are saved frames 0.75 ps apart. Connecting segments do not show sub-frame dynamics.</p><div id="sphere" style="height:570px"></div></div></div>
<script>const D=__DATA__;const col=(a,k)=>a.map(x=>x[k]);
function fill(id,rows){document.getElementById(id).innerHTML=rows.map((r,i)=>`<option value="${i}">${r.name}</option>`).join('')}
fill('track',D.tracks);fill('panel',D.panels);
function track(){const d=D.tracks[+document.getElementById('track').value],t=d.time;
let traces=['x','y','z'].map((n,k)=>({x:t,y:d.u.map((u,i)=>d.valid[i]?u[k]:null),name:n,mode:'lines',line:{width:1},xaxis:'x',yaxis:'y'}));
traces.push({x:t,y:d.norm,name:'norm / training RMS',xaxis:'x2',yaxis:'y2',line:{color:'#4e5ce6',width:1}});
traces.push({x:t,y:d.angle,name:'turn (°)',xaxis:'x3',yaxis:'y3',line:{color:'#cc7530',width:1}});
traces.push({x:t,y:d.q6,name:'q̄6',xaxis:'x4',yaxis:'y4',line:{color:'#00998c',width:1}});
Plotly.react('temporal',traces,{title:d.name,grid:{rows:4,columns:1,pattern:'independent'},margin:{t:45,l:65,r:20,b:45},height:670,
xaxis:{matches:'x4'},xaxis2:{matches:'x4'},xaxis3:{matches:'x4'},xaxis4:{title:'Time (ps)'},yaxis:{title:'Unit direction',range:[-1.1,1.1]},yaxis2:{title:'Norm'},yaxis3:{title:'Turn (°)'},yaxis4:{title:'q̄6'}},{responsive:true});
Plotly.react('sphere',[{type:'scatter3d',mode:'markers',x:d.u.map((u,i)=>d.valid[i]?u[0]:null),y:d.u.map((u,i)=>d.valid[i]?u[1]:null),z:d.u.map((u,i)=>d.valid[i]?u[2]:null),marker:{size:3,color:t,colorscale:'Viridis',colorbar:{title:'ps'}},text:t.map(x=>`${x} ps`)}],
{margin:{t:0,l:0,r:0,b:0},scene:{aspectmode:'cube',xaxis:{range:[-1,1]},yaxis:{range:[-1,1]},zaxis:{range:[-1,1]}}},{responsive:true});}
function panel(){const d=D.panels[+document.getElementById('panel').value];let p=[],u=[];d.p.forEach((v,i)=>{if(d.valid[i]){p.push(v);u.push(d.u[i])}});
Plotly.react('space',[{type:'scatter3d',mode:'markers',x:col(d.p,0),y:col(d.p,1),z:col(d.p,2),marker:{size:4,color:d.q6,cmin:0,cmax:.6,colorscale:'Viridis',colorbar:{title:'q̄6'}},text:d.ids.map((id,i)=>`atom ${id}<br>norm/RMS ${d.norm[i]}<br>PTM ${d.phase[i]}`),hoverinfo:'text'},
{type:'cone',x:col(p,0),y:col(p,1),z:col(p,2),u:col(u,0),v:col(u,1),w:col(u,2),sizemode:'absolute',sizeref:2,anchor:'tail',colorscale:[[0,'#4056bd'],[1,'#4056bd']],showscale:false,hoverinfo:'skip'}],
{margin:{t:0,l:0,r:0,b:0},scene:{aspectmode:'data',xaxis:{title:'x (Å)'},yaxis:{title:'y (Å)'},zaxis:{title:'z (Å)'}}},{responsive:true});}
document.getElementById('track').onchange=track;document.getElementById('panel').onchange=panel;track();panel();</script></body></html>'''
    text = template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload,allow_nan=False))
    (root/'explore.html').write_text(text.replace('__CHANNEL__',str(fields[PRIMARY]['channel'])))


def findings(root,tables,fields,config,parent):
    ts = pd.DataFrame(tables['temporal_summary']); ss = pd.DataFrame(tables['spatial_summary'])
    dg = pd.DataFrame(tables['diagnostics'])
    audit = json.loads((root/'technical/numerical-audit.json').read_text())
    maxerr = max(np.max(r['vector_relative_rms']) for r in audit['rotations'])
    def value(metric): return lookup(ts,PRIMARY,metric)['mean']
    lines = ['# GATr internal directions along Al trajectories', '',
        'Completed frozen-checkpoint evaluation on the A100 on node07. No training changes.', '',
        '## What was measured', '',
        f'Checkpoint 1216 (`{config["checkpoint_sha256"]}`), the same release used in the MACE/GATr jitter comparison. '
        '40 fixed-identity tracks, 32,040 test observations, 0–600 ps at 0.75 ps cadence; '
        f'{sum(r["centers"] for r in tables["sampling"]):,} spatial encodings across 70 snapshots. '
        'Training-only norm calibration: 420 observations. Ten held-out sources, two per temperature.', '',
        'The primary displayed feature is the point-numerator triplet before the last MLP, '
        f'channel {fields[PRIMARY]["channel"]}. Other stages, all eight channels and four triplet types are also exported. '
        'This is a rotation-covariant local vector, not a physical position, velocity or crystal axis.', '',
        '## Time: rapid reorientation at the saved cadence', '',
        f'Mean adjacent signed turn **{value("mean_angle_deg"):.1f}°**; '
        f'**{100*value("flip90_fraction"):.1f}%** exceed 90°, **{100*value("jump60_fraction"):.1f}%** exceed 60°. '
        f'Coverage is **{100*value("coverage"):.2f}%** after excluding weak vectors. '
        f'P1={value("p1"):.3f}, P2={value("p2"):.3f}. '
        f'Ignoring sign still gives mean axis turn **{value("mean_axis_angle_deg"):.1f}°**. '
        f'Correcting for the best-fit local cage rotation gives **{value("cage_mean_angle_deg"):.1f}°**.', '',
        'Across all eight point-triplet channels, the mean turn ranges from about 71–82° after block 1 '
        'and 69–78° before the final MLP; this behavior is not confined to the displayed channel.', '',
        'These are large frame-to-frame changes, not a stable orientation trajectory at 0.75 ps resolution. '
        'They do not prove mathematical discontinuities between frames; faster trajectory output would be needed to resolve that. '
        'Rotation equivariance is a symmetry property and does not imply temporal persistence.', '',
        '| Feature | Mean turn | Axis turn | P1 | P2 |', '|---|---:|---:|---:|---:|']
    for field in DISPLAY:
        vals = [lookup(ts,field,m)['mean'] for m in ('mean_angle_deg','mean_axis_angle_deg','p1','p2')]
        lines.append(f'| {DISPLAY[field]} | {vals[0]:.1f}° | {vals[1]:.1f}° | {vals[2]:.3f} | {vals[3]:.3f} |')
    lines += ['', 'Shape-axis sign is arbitrary; its signed turn/P1 are implementation diagnostics only. '
        'Compare its axis turn/P2. Tables retain source values and 95% source-bootstrap intervals.', '',
        'Among FCC→FCC adjacent pairs, the source-balanced mean turn is '
        f'**{pd.DataFrame(tables["phase_summary"]).query("field == @PRIMARY and phase == \'FCC\' and metric == \'mean_angle_deg\'")["mean"].iloc[0]:.1f}°**. '
        'Directional persistence is therefore particularly weak in the crystalline subset. '
        'This is consistent with a vector responding to fluctuating local asymmetry; it is not evidence of a persistent lattice axis.', '',
        '## Space: measured order relative to a phase-matched shuffle', '',
        '| Separation | P1 | Shuffle P1 | Excess P1 (95% interval) | Excess P2 | Nearest-80 overlap |',
        '|---|---:|---:|---:|---:|---:|']
    for lo,hi in zip(config['distance_edges_A'][:-1],config['distance_edges_A'][1:]):
        sub = ss[(ss.field==PRIMARY)&(ss.distance_lo_A==lo)].set_index('metric')
        ex = sub.loc['excess_p1']
        lines.append(f'| {lo}–{hi} Å | {sub.loc["p1","mean"]:.3f} | {sub.loc["shuffle_p1","mean"]:.3f} | '
            f'{ex["mean"]:.3f} [{ex.low:.3f}, {ex.high:.3f}] | {sub.loc["excess_p2","mean"]:.3f} | {sub.loc["overlap80","mean"]:.2f} |')
    lines += ['', 'Positive excess indicates more parallel directions (P1), or common axes irrespective of sign (P2), '
        'than the same snapshot and phase composition shuffled across positions. This does not establish crystal orientation order: '
        'the descriptors share atoms, and the sampling concentrates on two local patches. '
        'Observed directional order is weak below 4 Å and approximately absent at larger separation. '
        'Earlier block-1 point triplets have stronger nearest-neighbor alignment, still short-range. '
        'Separate FCC/FCC and unclassified/unclassified spatial tables are included. '
        'Compare density-dipole baselines and the 3D viewer.', '',
        '## What these directions encode', '']
    part = dg[dg.field==PRIMARY]
    lines += [f'The mean fraction of channel-axis energy in one direction is **{part.channel_axis_rank1_fraction.mean():.5f}** '
        '(1 means collinear axes across channels). '
        f'P2 alignment with the 7 Å density dipole is **{part.align_centroid_7A_p2.mean():.3f}**; '
        f'with the full-support dipole **{part.align_centroid_support_p2.mean():.3f}**; '
        f'with the local shape axis **{part.align_shape_axis_7A_p2.mean():.3f}**.', '',
        'Plane-normal triplets are exactly zero in the two earlier sampled stages. '
        'The small final plane-normal output is not used by the scalar readout. ', '',
        '## Numerical and readout controls', '',
        f'Five rigid rotations pass: maximum relative vector RMS error **{maxerr:.2g}**. '
        f'Repeated identical inference changes multivectors by **{audit["repeated_multivector_max_abs"]:g}**. '
        'Replacing the final multivector output with zeros leaves z128 exactly unchanged, as expected from the readout code.', '']
    interventions = json.loads((root/'technical/interventions.json').read_text())
    lines += ['| Intervention, 60 observations | Native BF16 z RMS change | Autocast disabled z RMS change |', '|---|---:|---:|']
    ir = pd.DataFrame(interventions['results']).pivot(index='intervention',columns='precision',values='rms')
    for name,row in ir.iterrows(): lines.append(f'| {name} | {row.bf16:.6g} | {row.float32:.6g} |')
    lines += ['', 'Earlier geometric streams do influence the output, but erasing all directional triplets just before the final MLP '
        'has no detectable effect for these inputs, even with autocast disabled. Do not equate a visually interesting last-layer '
        'direction with information actively used by the descriptor. Synthetic interventions are mechanistic probes, not plausible atomic motions.', '',
        'The trajectories originate from stored float16 positions. This audit measures the encoder on those actual inputs; '
        'it does not separately identify thermal motion, structural rearrangements and input-coordinate quantization as causes of angular changes.', '',
        '## Implications for training', '',
        'If the goal is a persistent orientational representation, first attach an explicitly used covariant readout to an earlier geometric '
        'stream and give it a physically meaningful target. Do not smooth the discarded final multivector output and assume z will improve. '
        'Use norm-aware temporal consistency, and verify that a nonzero feature remains sensitive to geometry. '
        'For crystal orientation, a vector alone cannot represent all symmetry-equivalent lattice axes; a symmetry-aware tensor or bond-order '
        'representation is a better target. Validate persistence, spatial alignment, physical sensitivity and collapse together. '
        'For the existing invariant z128, direct-z temporal regularization remains a separate objective.', '',
        '## Artifacts and reproduction', '',
        '- [Interactive 40-track / 140-patch explorer](explore.html)',
        '- [Summary figure](plots/overview.png), [time traces](plots/trajectories.png), [3D spatial arrows](plots/spatial-directions.png), [norm sensitivity](plots/norm-sensitivity.png)',
        '- [Exact metric definitions](tables/METRICS.md); all source, channel, phase, lag and distance tables in `tables/`.',
        '- Frozen checkpoint, extraction receipts, GPU identity and interventions in `technical/`.', '',
        '```bash',
        'conda run -n pointnet-torch214 python -m src.research.gatr_equivariant --config configs/analysis/gatr_equivariant.json',
        'conda run -n pointnet-torch214 python -m src.research.gatr_equivariant.controls --config configs/analysis/gatr_equivariant.json',
        '```', '',
        'Extraction requires node07/A100; run inside a valid allocation. The default all stage includes intervention controls and reporting. '
        'Separate stages temporal, spatial, then controls, then report are also available. Only this frozen Al checkpoint and cohort were tested. '
        'Two sources per temperature yield limited uncertainty estimates; weak-vector thresholds are diagnostics, not confidence calibration.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    (root/'README.md').write_text('# GATr equivariant audit\n\n[Results](RESULTS.md) · [Interactive explorer](explore.html) · [Metric definitions](tables/METRICS.md)\n')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GATr directional audit</title>'
        '<h1>GATr directional audit</h1><p><a href="explore.html">Open interactive explorer</a> · '
        '<a href="RESULTS.md">Scientific findings</a> · <a href="tables/METRICS.md">Metric definitions</a></p>'
        '<img src="plots/overview.png" style="max-width:100%" alt="Temporal and spatial directional statistics">')
