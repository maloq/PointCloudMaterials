"""Raster-only gallery for the completed structured forecast comparison."""
import html
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_table, snapshot_metric_docs
from src.research.crystallization_paths import figure_render as previous


def event_curves(root,config,metrics):
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.8),sharex=True)
    x=np.array(config['offsets_ps'])
    for row,encoder in enumerate(config['encoders']):
        for col,metric in enumerate(('matched_ap','restricted_timing_mae_ps')):
            ax=axes[row,col]
            for method in config['methods']:
                values=[metrics[encoder+'-'+method][str(o)][metric] for o in x]
                mean=np.array([v['value'] for v in values]);band=np.array([v['ci95'] for v in values])
                ax.plot(x,mean,color=previous.COLORS[method],marker='o',ms=3)
                ax.fill_between(x,band[:,0],band[:,1],color=previous.COLORS[method],alpha=.09,lw=0)
            if col==0:
                ax.axhline(.5,color='#999999',ls=':',lw=1);ax.set(ylim=(.45,1.01),ylabel='Matched average precision')
            else:ax.set(ylabel='Timing MAE (ps)',ylim=(0,None))
            previous.panel(ax,chr(97+row*2+col),encoder.upper())
            ax.set(xticks=[3,12,24,36,48],xlim=(2,49))
            if row==1:ax.set_xlabel('Nominal time before the same onset (ps)')
    fig.legend(handles=previous.handles(config['methods']),ncol=4,loc='upper center',bbox_to_anchor=(.52,1.01))
    fig.tight_layout(rect=(0,0,1,.95),w_pad=2.5,h_pad=2)
    previous.save(fig,root,'08_event_offset_skill',config['dpi'])
    # Probability and conditional timing make censoring/low predicted event mass visible.
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.8),sharex=True)
    for row,encoder in enumerate(config['encoders']):
        for method in config['methods']:
            values=metrics[encoder+'-'+method]
            for key,ls in (('case_probability','-'),('control_probability','--')):
                axes[row,0].plot(x,[values[str(o)][key]['value'] for o in x],ls=ls,color=previous.COLORS[method])
            y=[values[str(o)]['conditional_timing_mae_ps']['value'] for o in x]
            axes[row,1].plot(x,y,color=previous.COLORS[method],marker='o',ms=3)
        axes[row,0].set(ylabel='Probability by reference onset',ylim=(0,1))
        axes[row,1].set(ylabel='Conditional timing MAE (ps)',ylim=(0,None))
        for col in range(2):
            previous.panel(axes[row,col],chr(97+row*2+col),encoder.upper())
            axes[row,col].set(xticks=[3,12,24,36,48],xlim=(2,49))
            if row==1:axes[row,col].set_xlabel('Nominal time before the same onset (ps)')
    legends=previous.handles(config['methods'])+[Line2D([],[],c='#666666',label='Event'),Line2D([],[],c='#666666',ls='--',label='Control')]
    fig.legend(handles=legends,ncol=6,loc='upper center',bbox_to_anchor=(.52,1.01),fontsize=8)
    fig.tight_layout(rect=(0,0,1,.95),h_pad=2,w_pad=2.5)
    previous.save(fig,root,'09_event_offset_probability',config['dpi'])


def event_examples(root,config):
    a=np.load(root/'technical/event-examples.npz');fig,axes=plt.subplots(3,2,figsize=(9.5,8),sharex=True,sharey=True)
    offsets=[3,12,24,48];colors=['#d94b3d','#dfa346','#429f98','#4c64a4']
    for row in range(3):
        for col,key in enumerate(('mace-ar_mse','gatr-mixture')):
            ax=axes[row,col]
            for offset,color in zip(offsets,colors,strict=True):
                j=config['offsets_ps'].index(offset);lead=a['leads'][row,j]
                x=np.arange(129)*.75-lead;y=np.r_[0,a[key][row,j]]
                ax.plot(x,y,color=color);ax.scatter(-lead,0,s=13,color=color,zorder=3)
            ax.axvline(0,c=previous.INK,lw=1,ls='--');ax.set(xlim=(-52,48),ylim=(-.025,1.025),xticks=[-48,-24,0,24,48],yticks=[0,.5,1])
            previous.panel(ax,chr(97+2*row+col),f'Event {row+1} · '+('MACE autoregressive' if col==0 else 'GATr mixture'))
            if col==0:ax.set_ylabel('Onset probability')
            if row==2:ax.set_xlabel('Time relative to actual onset (ps)')
    fig.legend(handles=[Line2D([],[],c=c,label=f'{o} ps before') for o,c in zip(offsets,colors,strict=True)],ncol=4,loc='upper center',bbox_to_anchor=(.52,1.01))
    fig.tight_layout(rect=(0,0,1,.96),h_pad=2,w_pad=2)
    previous.save(fig,root,'10_same_event_forecasts',config['dpi'])


def spatial(root,config):
    """The original three-panel visual grammar, with the actual 25-slot inputs."""
    a=np.load(root/'technical/context-clouds.npz');m=json.loads((root/'technical/context-method.json').read_text())
    theta,phi=.38,.55
    rz=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
    rotation=rz@rx
    q=a['queries']@rotation;r=a['representatives']@rotation
    colors=['#202b3b']+['#3296b8']*12+['#d9953c']*12
    slots=range(25)
    if m['query_slots']!=list(slots):
        raise ValueError('Full-context figure requires all 25 reconstructed MACE neighborhoods')
    fig=plt.figure(figsize=(14,6.3))
    # One rigid orthographic camera for real atoms, queries and local crops.
    ax=fig.add_axes([.015,.13,.32,.75])
    full=a['full'];full=full[np.linalg.norm(full,axis=1)<32]@rotation
    order=np.argsort(full[:,2]);ax.scatter(*full[order,:2].T,s=2.2,color='#bfc6ce',alpha=.23,linewidths=0)
    for radius,ls in ((10,':'),(20,'--')):
        ax.add_patch(Circle((0,0),radius,fill=False,ec='#6d7886',lw=.9,ls=ls))
    # Draw every local support, rear-to-front, with the center on top.
    for slot in [*sorted(range(1,25),key=lambda i:r[i,2]),0]:
        cloud=a[f'mace_{slot}']@rotation+r[slot]
        order=np.argsort(cloud[:,2]);ax.scatter(*cloud[order,:2].T,s=3.6,color=colors[slot],alpha=.40,linewidths=0)
        ax.add_patch(Circle(r[slot,:2],m['radii_A']['mace'],fill=False,ec=colors[slot],lw=.65,alpha=.65))
    # Open nominal queries and filled assigned atom centers both remain visible.
    for slot in range(25):
        ax.plot([q[slot,0],r[slot,0]],[q[slot,1],r[slot,1]],c=colors[slot],lw=.65,alpha=.7,zorder=4)
        ax.scatter(*q[slot,:2],s=30,facecolors='none',edgecolors=colors[slot],linewidths=.75,zorder=4)
        ax.scatter(*r[slot,:2],s=18,color=colors[slot],edgecolors='white',linewidths=.45,zorder=5)
    for slot in slots:
        ax.scatter(*r[slot,:2],s=32,color=colors[slot],edgecolors='white',linewidths=.65,zorder=6)
        ax.annotate(str(slot),r[slot,:2],xytext=(4,4),textcoords='offset points',fontsize=6.8,
            color=colors[slot],weight='bold',zorder=7,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.65,pad=.2))
    ax.text(0,21.5,'20 Å',ha='center',fontsize=9)
    ax.text(-1,-11.8,'10 Å',ha='center',fontsize=8,color='#687481')
    ax.plot([-28,-18],[-30,-30],color=previous.INK,lw=1.5)
    ax.text(-23,-33,'10 Å',ha='center',fontsize=8)
    ax.set(xlim=(-33,33),ylim=(-34,33),aspect='equal');ax.axis('off')
    for x,letter,title in ((.025,'a','Spatial representatives'),(.375,'b','Local encoder inputs'),(.76,'c','Space–time context')):
        fig.text(x,.95,letter,weight='bold',fontsize=11);fig.text(x+.03,.95,title,fontsize=10)
    # All 25 atomic crops, at identical magnification and with the same camera.
    radius=m['radii_A']['mace']
    fig.text(.535,.883,f'MACE · $R_{{local}}$ = {radius:.2f} Å',ha='center',fontsize=9)
    for slot in slots:
        row,col=divmod(slot,5)
        patch=fig.add_axes([.375+col*.067,.711-row*.147,.061,.132]);cloud=a[f'mace_{slot}']@rotation
        order=np.argsort(cloud[:,2])
        patch.scatter(*cloud[order,:2].T,s=3.5,color=colors[slot],alpha=.65,lw=.15,edgecolors='white')
        patch.scatter(0,0,s=13,facecolors='white',edgecolors=previous.INK,linewidths=.65,zorder=5)
        patch.add_patch(Circle((0,0),radius,fill=False,ec='#bfc6ce',lw=.55))
        patch.set(aspect='equal',xlim=(-radius-.6,radius+.6),ylim=(-radius-.6,radius+.6));patch.axis('off')
        patch.text(.5,-.065,str(slot),transform=patch.transAxes,ha='center',fontsize=7,color=colors[slot])
    # Every row contains all 25 embeddings. Squares denote spatial attention
    # modules retaining the slots, not per-frame pooling before temporal work.
    diagram=fig.add_axes([.75,.12,.245,.74]);diagram.set(xlim=(-1,10),ylim=(-1.5,5));diagram.axis('off')
    for row,time in enumerate((-48,-12,-3,0)):
        y=4-row;diagram.text(-.6,y,str(time),ha='right',va='center',fontsize=8)
        for slot,color in enumerate(colors):diagram.scatter(slot*.18,y,s=10,color=color,linewidths=0)
        diagram.add_patch(FancyArrowPatch((4.8,y),(6.25,y),arrowstyle='->',mutation_scale=9,lw=.8,color='#7d8793'))
        diagram.scatter(6.65,y,s=48,marker='s',color='#718599')
    diagram.text(-.6,4.7,'ps',ha='right',fontsize=8)
    diagram.text(2.15,4.7,'25 embeddings',ha='center',fontsize=8)
    diagram.text(6.65,4.7,'Spatial',ha='center',fontsize=8)
    diagram.add_patch(FancyArrowPatch((7.9,4.2),(7.9,.4),arrowstyle='->',mutation_scale=10,lw=1,color='#5f6d7f'))
    diagram.text(8.6,2.3,'Causal temporal',rotation=90,ha='center',va='center',fontsize=8)
    diagram.text(6.65,.35,'×2',ha='center',fontsize=8,color='#5f6d7f')
    diagram.add_patch(FancyArrowPatch((6.65,.12),(6.65,-.35),arrowstyle='->',mutation_scale=9,lw=.8,color='#5f6d7f'))
    diagram.scatter(6.65,-.55,s=75,marker='s',color=previous.COLORS['direct'])
    diagram.text(6.65,-1.2,'Pool → forecast',ha='center',fontsize=9)
    fig.add_artist(FancyArrowPatch((.335,.52),(.37,.52),transform=fig.transFigure,arrowstyle='->',mutation_scale=11,lw=1,color='#8792a0'))
    fig.add_artist(FancyArrowPatch((.714,.46),(.752,.46),transform=fig.transFigure,arrowstyle='->',mutation_scale=11,lw=1,color='#8792a0'))
    previous.save(fig,root,'07_symmetric_context',config['dpi'])


def overview(root,config):
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.6));x=np.arange(4)
    base=resolve_path(config['input'])/'technical/runs'
    for encoder,color,dx in (('mace','#2378a3',-.08),('gatr','#d48728',.08)):
        metrics=[json.loads((base/f'{encoder}-{method}-symmetric-E36/metrics.json').read_text()) for method in config['methods']]
        axes[0].plot(x+dx,[m['dense_integrated_brier'] for m in metrics],marker='o',color=color,label=encoder.upper(),lw=1)
        axes[1].plot(x+dx,[100*m['short_horizon']['classification']['12.0']['average_precision'] for m in metrics],marker='o',color=color,lw=1)
    axes[0].set_ylabel('Integrated Brier · 0–96 ps');axes[1].set_ylabel('Average precision · 12 ps (%)')
    for i,ax in enumerate(axes):
        ax.set(xticks=x,xticklabels=[previous.NAMES[n] for n in config['methods']]);previous.panel(ax,chr(97+i))
        ax.tick_params(axis='x',labelsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.51,1.02))
    fig.tight_layout(rect=(0,0,1,.93),w_pad=3)
    previous.save(fig,root,'00_model_comparison',config['dpi'])


def descriptions(root,config,cases,metrics):
    receipt=json.loads((root/'technical/prepared.json').read_text());cohort=json.loads((root/'technical/event-cohort.json').read_text())
    context=json.loads((root/'technical/context-method.json').read_text());illustration=np.load(root/'technical/event-examples.npz')['indices']
    notes={
        'plots/00_model_comparison.png':('Completed forecast comparison','Eight completed one-seed fits, 44,385 at-risk origins from 30 held-out source trajectories. Equal-source integrated Brier spans 128 bins through 96 ps; 12 ps AP uses the natural window population. Predictors start from scratch with frozen encoders, identical 25-query context and observed descriptor auxiliaries. MACE/GATr retain different local supports, so this is not an architecture-only comparison. Lines connect categorical model choices; no interpolation or uncertainty inference is intended.'),
        'plots/07_symmetric_context.png':('Structured context from real atomic clouds',f'Source {context["source"]}, tracked atom {context["tracked_atom_id"]}, time {context["time_ps"]:g} ps. Panel a uses the original orthographic-cloud style: all 25 nominal queries (open markers), assigned atom centers (filled markers), and short assignment offsets. Dashed/dotted circles mark the 20/10 Å spherical query radii in projection; maximum assignment offset here is {context["max_assignment_offset_A"]:.2f} Å. All 25 MACE neighborhoods are highlighted, each with its full local support circle and real atoms: tracked center 0, inner-shell slots 1–12, and outer-shell slots 13–24. Overlapping neighborhoods share atoms. Gray background atoms extend to approximately 32 Å for illustration and do not define the model support. Panel b shows all 25 corresponding real atomic crops in slot order, with one common camera and identical magnification; every crop has a 7.94 Å support radius. Both panels show the entire MACE spatial context at this observed time. Projected overlaps do not mean atoms coincide in 3D. Panel c shows all 25 spatial slots at all four input times. Squares denote spatial attention retaining the slots; the ×2 indicates two alternating spatial/causal temporal blocks, followed by a single pooling stage and forecast. The stencil is box-fixed; queries are symmetric, real atom assignments need not be. Slot identity is a spatial query, not a tracked neighboring atom. No future frame enters prediction.'),
        'plots/08_event_offset_skill.png':('Same events, changing forecast lead',f'The identical {receipt["matched_events"]} distinct first local onset events and {receipt["matched_events"]} matched control records from {receipt["matched_sources"]} sources are reused at every offset and for every model. A control is another tracked center in the same source, liquid/at risk at each forecast origin, with no first onset by the reference event time; it may crystallize later. One seeded control is chosen using availability/labels only and reused at every offset. Some controls repeat across pairs. Each source has equal total weight, its events share this weight, and cases/controls have equal class weight. Left: pooled weighted AP using predicted probability of onset by the reference event time (forecast horizon equals the actual lead); the balanced prevalence baseline is 0.5. This is a case-control diagnostic, not the previous population AP. Right: absolute error of the 96 ps restricted-mean predicted onset time across ALL events, including missed alarms; survival mass stays at 96 ps and no true event time truncates the timing estimator. Each requested origin rounds down to the existing 3 ps grid, making actual leads nominal to nominal+2.25 ps. Shading: 95% paired source-bootstrap intervals, 1,000 draws; no seed uncertainty. Different offsets change the horizon used for AP, but the cohort and estimator stay fixed. Events without all offsets or a control are excluded and listed in the cohort manifest.'),
        'plots/09_event_offset_probability.png':('Probability and conditional timing checks','Same fixed event/control cohort and source weighting as Figure 8. Solid lines: average predicted onset probability by the reference event; dashed lines: matched controls. The horizon grows with lead, so a larger cumulative probability alone does not establish earlier warning. Right: predicted mean time conditional on onset somewhere in the full next 96 ps, evaluated on ALL true events regardless of alarm. This conditional error can conceal low event probability; the restricted-mean error in Figure 8 retains survival mass. Neither estimator is conditioned on onset by the known true event time.'),
        'plots/10_same_event_forecasts.png':('The same event viewed from different origins','Three score-independent examples: first choose the median-onset event per source, then the 20/50/80% source representatives ordered by onset time. Left/right show MACE autoregressive and GATr mixture on exactly the same events. Each curve is an archived open-loop CDF launched approximately 3, 12, 24 or 48 ps before onset; its start is marked at probability zero. The x-axis is aligned to actual onset (dashed zero). The future onset is used only to align this retrospective figure; predictors receive only prior observations. These selected predictor examples do not replace the all-eight-model aggregate curves.')}
    for encoder in config['encoders']:
        prefix=encoder+'/plots/'
        notes[prefix+'01_structural_trajectories.png']=(encoder.upper()+' structural forecasts','Identical four source-separated examples for both backbones: early onset, later onset, no onset within 96 ps, and an early event missed by the MACE direct model. Examples are median-error windows within sources, then median across eligible sources, using MACE direct only for illustration. Gray past is observed; zero is forecast origin; dashed line is first onset. Direct/AR predict q6 alongside latent and physical states, not by decoding a latent or reconstructing coordinates. All methods observe −48, −12, −3 and 0 ps and roll out open loop through 96 ps. Dense black past is shown only for orientation.')
        notes[prefix+'02_onset_probabilities.png']=(encoder.upper()+' onset probabilities','Saved onset CDFs for the same four windows. Onset is the tracked atom first crystalline for three consecutive 0.75 ps frames. The “missed early onset” panel is defined by MACE direct’s calibration-only 5%-FPR threshold and is shared across backbones; it is not necessarily missed by every model. No onset means none within 96 ps, not permanent survival.')
        notes[prefix+'03_predictive_spread.png']=(encoder.upper()+' probabilistic trajectories',f'Mixture/diffusion mean q6 and pointwise 5–95% sample interval from {config["samples"]} fresh fixed-seed inference draws; five predetermined sample paths are faintly shown. Two examples match Figures 1–2. All predictors use the same four observation times. Shading is model spread, not a confidence interval or a simultaneous coverage claim; no best-of-samples selection. Aggregates use the original archived evaluation predictions.')
        notes[prefix+'04_forecast_quality.png']=(encoder.upper()+' held-out forecast quality','Shared physical128 standardized mean-path MSE and dense onset Brier by forecast horizon, using original saved predictions on all 44,385 windows. Every source has equal total weight; bands are 95% paired whole-source bootstrap intervals from 1,000 draws. Persistence keeps current physical state constant. Latent error is deliberately excluded from cross-backbone comparisons. Selection uses development sources; the test population has been examined previously.')
        notes[prefix+'05_embedding_umap.png']=(encoder.upper()+' embedding space','128D frozen center embeddings. Standardization and UMAP fit only on 5,760 states from 90 training sources; 1,920 states from 30 test sources are transformed afterward. Both maps use exactly the same outcome-independent sampled identities and timeline range 0–594 ps, including post-onset states; color shows PTM status, q6 and temperature. Separate maps are fitted for MACE/GATr, so axes are not aligned and cross-map distances are meaningless. Euclidean UMAP: 30 neighbors, min_dist 0.15, fixed seed. Colors do not enter fitting.')
        notes[prefix+'06_forecast_umap_paths.png']=(encoder.upper()+' embedding trajectories','Observed and predicted paths for the same four windows in that backbone’s training-fitted UMAP. White circle: origin; squares: 96 ps endpoints. Gray dotted: observed history; black: observed future; blue/orange: direct/AR predicted future. The 128D predictive mean is transformed before plotting. UMAP can distort or compress off-manifold errors; apparent 2D agreement is not a forecast-quality metric.')
    intro=(f'# Symmetric-context prediction analysis\n\nAll eight completed MACE/GATr predictors; no new training. '
           f'The event-aligned study follows **{receipt["matched_events"]} local events from {receipt["matched_sources"]} sources** '
           'at every offset, with fixed same-source controls. **Matched AP has 50% case prevalence** and is not comparable numerically with natural-population AP. '
           '**Timing uses every event**, retaining survival probability through a 96 ps restricted mean. PNG only; captions below.\n\n')
    intro+='[Readable metric table](tables/event_offset_summary.csv) · [Definitions](tables/METRICS.md) · [HTML gallery](index.html)\n\n'
    intro+='| Model | Nominal lead (ps) | Matched AP | All-event timing MAE (ps) |\n|---|---:|---:|---:|\n'
    for key in ('mace-direct','mace-ar_mse','gatr-mixture'):
        for offset in (3,12,24,48):
            v=metrics[key][str(offset)]
            intro+=f'| {key} | {offset} | {v["matched_ap"]["value"]:.3f} | {v["restricted_timing_mae_ps"]["value"]:.2f} |\n'
    intro+='\nNear-onset discrimination is stronger; around 48 ps, matched AP is close to the 0.5 reference. Timing errors here include every event and 96 ps survival mass, so they are not the earlier detected-only 12 ps timing errors. These are exploratory, one-seed results.\n\n'
    text=intro
    for path,(title,caption) in notes.items():text+=f'## {title}\n\n![{title}]({path})\n\n{caption}\n\n'
    text+='## Example identities\n\n'+json.dumps(cases,indent=2)+'\n\nEvent-aligned illustration records:\n\n'+json.dumps([cohort['records'][int(i)] for i in illustration],indent=2)+'\n'
    (root/'README.md').write_text(text)
    cards=''.join(f'<section><h2>{html.escape(title)}</h2><img src="{path}" alt="{html.escape(title)}"><p>{html.escape(caption)}</p></section>' for path,(title,caption) in notes.items())
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Symmetric-context forecasts</title><style>body{max-width:1180px;margin:40px auto;font:16px/1.6 system-ui;color:#252932;padding:0 24px}img{width:100%;height:auto}section{margin:55px 0}h1,h2{font-weight:550}p{max-width:1050px}</style><h1>Symmetric-context forecast analysis</h1><p>'+html.escape(intro.split('\n\n')[1])+'</p>'+cards)


def render(config):
    root=resolve_path(config['output']);previous.style()
    cases=json.loads((root/'technical/examples.json').read_text())
    metrics=json.loads((root/'technical/event-metrics.json').read_text())
    write_metric_table(metrics,root,family='structured_figures',name='event_offsets')
    rows=[]
    for model,values in metrics.items():
        for offset,entry in values.items():
            row=dict(model=model,nominal_lead_ps=float(offset),actual_lead_min_ps=entry['actual_lead_min_ps'],
                actual_lead_max_ps=entry['actual_lead_max_ps'],events=entry['events'],sources=entry['sources'])
            for key in ('matched_ap','restricted_timing_mae_ps','conditional_timing_mae_ps','case_probability','control_probability','event_mass_96ps'):
                row[key]=entry[key]['value'];row[key+'_lower95']=entry[key]['ci95'][0];row[key+'_upper95']=entry[key]['ci95'][1]
            rows.append(row)
    with (root/'tables/event_offset_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for encoder in config['encoders']:
        out=root/encoder;observed=np.load(out/'technical/observed.npz')
        snapshot_metric_docs(out,'structured_figures')
        predictions={name:dict(np.load(out/'technical'/f'{name}-examples.npz')) for name in config['methods']}
        previous.trajectories(out,cases,observed,predictions,config['dpi'])
        previous.onset(out,cases,predictions,config['dpi'])
        previous.uncertainty(out,cases,observed,predictions,config['dpi'])
        previous.aggregate(out,config['dpi']);previous.embedding(out,config['dpi']);previous.embedding_paths(out,cases,config['dpi'])
    overview(root,config);spatial(root,config);event_curves(root,config,metrics);event_examples(root,config)
    descriptions(root,config,cases,metrics)
