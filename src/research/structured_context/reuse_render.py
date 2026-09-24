"""Minimal-text PNG figures; numerical definitions and captions stay separate."""
import html
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from sklearn.metrics import precision_recall_curve,average_precision_score
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.metrics import source_weights
from src.research.crystallization_paths.figure_render import style,panel,save,NAMES
from .geometry import stencil

COLORS={'observed':'#427ca6','relaxed':'#d77b39'}
LABELS={'observed':'Original MACE','relaxed':'Relaxed MACE'}
METHODS=('direct','ar_mse','mixture','diffusion')
DOMAINS=('observed','relaxed')
CAPTIONS={}


def finish(fig,root,name,config,caption):
    caption=re.sub(r'([A-Za-z])(\d)',r'\1 \2',caption)
    caption=re.sub(r'(\d)(ps|Å)\b',r'\1 \2',caption)
    save(fig,root,name,config['dpi']);CAPTIONS[name]=caption


def legend_domains(fig,y=1.02):
    fig.legend(handles=[Line2D([],[],c=COLORS[d],label=LABELS[d]) for d in DOMAINS],loc='upper center',bbox_to_anchor=(.52,y),ncol=2)


def summary_plots(config,root,corpus,predictions,metrics):
    labels=[NAMES[m] for m in METHODS];x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.7))
    for j,(key,ylabel) in enumerate([('ap','Average precision at 12 ps'),('brier','Integrated Brier score')]):
        ax=axes[j]
        for i,m in enumerate(METHODS):
            vals=[metrics[d+'-'+m]['short_horizon']['classification']['12.0']['average_precision'] if key=='ap' else metrics[d+'-'+m]['dense_integrated_brier'] for d in DOMAINS]
            ax.plot([i-.12,i+.12],vals,c='#b4b9be',lw=1)
            for d,xx,y in zip(DOMAINS,[i-.12,i+.12],vals):ax.scatter(xx,y,c=COLORS[d],s=35,zorder=3)
        ax.set(xticks=x,xticklabels=labels,ylabel=ylabel);ax.tick_params(axis='x',labelrotation=15);panel(ax,chr(97+j))
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.94),w_pad=3)
    finish(fig,root,'01_prediction_summary',config,'Matched test comparison across all four predictor families. Higher AP at12 ps and lower integrated Brier over0.75–96 ps are better. Points are one-seed estimates; connecting lines indicate matched methods, not confidence intervals. Both encoder and input domain change together.')
    p0=predictions['relaxed-ar_mse'];weights=source_weights(corpus.source_ids[p0['test_indices']]);truth=p0['test_event']<16
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.4),sharex=True,sharey=True)
    for ax,m,letter in zip(axes.flat,METHODS,'abcd'):
        for d in DOMAINS:
            p=predictions[d+'-'+m];precision,recall,_=precision_recall_curve(truth,p['evaluated_risk12'],sample_weight=weights)
            ax.plot(recall,precision,c=COLORS[d])
        ax.axhline(weights@truth,c='#a3a7ac',ls=':',lw=.8)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision');panel(ax,letter,NAMES[m])
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    finish(fig,root,'02_precision_recall',config,'12 ps precision–recall curves, equal total weight per test source. Dotted lines show source-weighted prevalence. Curves are computed from saved predictions and reproduce the exported AP values. There are7654 test windows from30 sources;226 raw windows are positive, and overlapping windows are not independent events.')
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.8))
    for d,offset in [('observed',-.18),('relaxed',.18)]:
        values=[metrics[d+'-'+m]['short_horizon']['timing']['12.0'] for m in METHODS]
        axes[0].bar(x+offset,[v['detected_timing_mae_ps'] for v in values],.32,color=COLORS[d])
        axes[1].bar(x+offset,[v['missed_windows']/v['event_windows'] for v in values],.32,color=COLORS[d])
    for ax,label,letter in zip(axes,['Detected-window timing MAE (ps)','Missed positive windows / all positives'],'ab'):
        ax.set(xticks=x,xticklabels=labels,ylabel=label,ylim=(0,None));ax.tick_params(axis='x',labelrotation=15);panel(ax,letter)
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.94),w_pad=3)
    finish(fig,root,'03_timing_and_misses',config,'Timing error must be read together with misses. Left: conditional event-time MAE only on detected positive12 ps windows. Right: raw missed-window fraction among226 positives. Each fit uses its own threshold calibrated to5% FPR on calibration sources; realized test FPR can differ. The detected populations differ between fits, so lower left-hand bars alone do not establish better overall timing.')
    horizons=np.array([.75,3,6,9,12,24,48,96]);fig,axes=plt.subplots(2,2,figsize=(8.8,6.4),sharex=True,sharey=True)
    for ax,m,letter in zip(axes.flat,METHODS,'abcd'):
        for d in DOMAINS:
            p=predictions[d+'-'+m];ys=[average_precision_score(p['test_event']<round(h/.75),p['evaluated_horizon_risk'][:,j],sample_weight=weights) for j,h in enumerate(horizons)]
            ax.plot(horizons,ys,c=COLORS[d],marker='o',ms=3)
        ax.set(xlabel='Forecast horizon (ps)',ylabel='Average precision',xticks=[0,12,48,96],ylim=(0,1));panel(ax,letter,NAMES[m])
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    finish(fig,root,'04_skill_by_horizon',config,'AP as the forecast horizon changes for the same full test population. This is not the fixed-event offset analysis: longer horizons change which windows are positive and increase event prevalence. Source weights and saved CDFs match the primary evaluation.')
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.6),sharex=True)
    names=['physical','bond_order','crystallinity','embedding']
    for j,(ax,block) in enumerate(zip(axes.flat,names)):
        bi={'embedding':0,'physical':1,'bond_order':2,'crystallinity':3}[block]
        for d in DOMAINS:
            for m,ls in [('ar_mse','-'),('mixture','--')]:
                y=np.einsum('n,nt->t',weights,predictions[d+'-'+m]['test_path_scores'][:,:,bi])
                ax.plot(np.arange(1,33)*3,y,c=COLORS[d],ls=ls)
        baseline=np.einsum('n,nt->t',weights,p0['test_persistence_scores'][:,:,bi]);ax.plot(np.arange(1,33)*3,baseline,c='#80858b',ls=':',lw=1)
        ax.set(xlabel='Forecast horizon (ps)',ylabel='Standardized MSE',xticks=[3,24,48,72,96]);panel(ax,chr(97+j),block.replace('_',' ').capitalize())
    handles=[Line2D([],[],c=COLORS[d],ls=ls,label=LABELS[d]+' · '+short) for d in DOMAINS for ls,short in [('-','AR'),('--','mixture')]]
    handles.append(Line2D([],[],c='#80858b',ls=':',label='MD persistence'))
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.025),fontsize=8);fig.tight_layout(rect=(0,0,1,.91),h_pad=2)
    finish(fig,root,'05_physical_trajectory_error',config,'Source-weighted mean-path squared error by future time for autoregressive and mixture predictors. Blocks use identical training-normalized original-MD targets in both arms, including the common original-MACE latent targets. Dotted persistence uses the original MD present state; it is a reference baseline and does not imply that state was supplied to the relaxed predictor. Mixture means are estimated using the saved evaluation samples.')
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.5),sharex=True,sharey=True)
    for ax,m,letter in zip(axes.flat,METHODS,'abcd'):
        ax.plot([0,1],[0,1],c='#aaa',ls=':',lw=1)
        for d in DOMAINS:
            risk=predictions[d+'-'+m]['evaluated_risk12'];xs=[];ys=[];size=[]
            for lo in np.arange(0,1,.1):
                keep=(risk>=lo)&(risk<(lo+.1 if lo<.9 else 1.000001))
                if not keep.any():continue
                w=weights[keep]/weights[keep].sum();xs.append(w@risk[keep]);ys.append(w@truth[keep]);size.append(15+90*weights[keep].sum())
            ax.plot(xs,ys,c=COLORS[d],lw=1);ax.scatter(xs,ys,s=size,c=COLORS[d],alpha=.8)
        ax.set(xlabel='Predicted 12 ps probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1));panel(ax,letter,NAMES[m])
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    finish(fig,root,'06_calibration',config,'Reliability diagrams in10 fixed probability bins at12 ps. Means and event fractions use equal-source weights. Marker area increases with bin probability mass; sparse high-risk bins should not be read as precise calibration estimates. Predictions are raw saved model risks, not fitted test-set recalibrations.')


def event_plots(config,root,predictions,matched,summary,examples):
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharex=True)
    x=np.array(config['offsets_ps'])
    for i,m in enumerate(('ar_mse','mixture')):
        for j,(metric,label) in enumerate([('matched_ap','Matched average precision'),('restricted_timing_mae_ps','96 ps restricted-time MAE (ps)')]):
            ax=axes[i,j]
            for d in DOMAINS:
                rows=[summary[d+'-'+m][str(o)][metric] for o in x];y=np.array([r['value'] for r in rows]);ci=np.array([r['ci95'] for r in rows])
                ax.plot(x,y,c=COLORS[d],marker='o',ms=3);ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[d],alpha=.12,lw=0)
            if j==0:ax.axhline(.5,c='#aaa',ls=':',lw=.8);ax.set_ylim(0,1)
            ax.set(xlabel='Lead-bin lower bound (ps)',ylabel=label,xticks=x);panel(ax,chr(97+i*2+j),NAMES[m])
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.95),h_pad=2,w_pad=3)
    finish(fig,root,'07_same_event_offset_skill',config,'Fixed cohort of66 local onset/control pairs from18 test sources, present at all four lead bins. At nominal leadL the archived origin has actual lead in[L,L+12) ps; no prediction is interpolated. Controls come from the same source and survive beyond the case onset. Matched AP has50% weighted prevalence and is not comparable numerically to natural-population AP. Timing uses the96 ps restricted mean for every case, including weak or missed predictions. Bands:1000 whole-source bootstrap draws, conditional on the trained seed and selected cohort.')
    fig,axes=plt.subplots(3,2,figsize=(9.5,8.3),sharex=True,sharey=True);colors=['#d25042','#d49e39','#368f97','#6465a9']
    for row,example in enumerate(examples):
        for col,d in enumerate(DOMAINS):
            ax=axes[row,col]
            for j,color in enumerate(colors):
                lead=matched['leads'][example,j];r=matched['rows'][example,j,0]
                ax.plot(np.arange(129)*.75-lead,np.r_[0,predictions[d+'-ar_mse']['test_cdf'][r]],c=color)
                ax.scatter(-lead,0,c=color,s=14)
            ax.axvline(0,c='#333',ls='--',lw=.8)
            ax.set(xlim=(-60,84),ylim=(-.02,1.02),xticks=[-48,-24,0,24,48,72],yticks=[0,.5,1]);panel(ax,chr(97+2*row+col),f'Event {row+1} · '+LABELS[d])
            if col==0:ax.set_ylabel('Cumulative onset probability')
            if row==2:ax.set_xlabel('Time relative to actual onset (ps)')
    fig.legend(handles=[Line2D([],[],c=c,label=f'{o}–{o+12} ps lead') for c,o in zip(colors,config['offsets_ps'])],ncol=4,loc='upper center',bbox_to_anchor=(.52,1.01),fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96),h_pad=2)
    finish(fig,root,'08_same_event_forecasts',config,'Autoregressive onset CDFs for three label/availability-selected events from distinct sources. Both columns show identical atoms, origins and outcomes. Each color is an independently initialized open-loop forecast from a real archived origin; zero marks actual onset. Dots mark forecast origins. Examples were selected with a fixed random seed before reading forecast quality. Exact source, atom-index, onset and origin coordinates are in technical/matched-events.json.')


def replay_plots(config,root,corpus):
    a=np.load(root/'technical/trajectory-examples.npz');meta=json.loads((root/'technical/trajectory-examples.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.4),sharey=True)
    for i,ax in enumerate(axes):
        case=meta['cases'][i];s,ai,ci,t=corpus.rows[meta['ids'][i]];anchor=corpus.plan['anchors'][ai]
        frames=np.arange(max(0,anchor-96),anchor+129,4);truth=corpus.arrays[s]['order'][ci,frames,1]
        ax.plot((frames-anchor)*.75,truth,c='#30343a',lw=1.2)
        for d in DOMAINS:ax.plot(np.arange(1,33)*3,a[d+'-ar_mse'][i,0,:,257],c=COLORS[d])
        ax.axvline(0,c='#999',ls=':',lw=.8);ax.axvline(case['lead_ps'],c='#333',ls='--',lw=.8);ax.axvspan(-72,0,color='#f1f2f4',zorder=-1)
        ax.set(xlim=(-72,96),xticks=[-72,0,48,96],xlabel='Time from forecast origin (ps)');panel(ax,chr(97+i),f'Event {i+1}')
    axes[0].set_ylabel(r'$q_6$');fig.legend(handles=[Line2D([],[],c='#30343a',label='MD truth')]+[Line2D([],[],c=COLORS[d],label=LABELS[d]) for d in DOMAINS],ncol=3,loc='upper center',bbox_to_anchor=(.52,1.05))
    fig.tight_layout(rect=(0,0,1,.92));finish(fig,root,'09_structural_forecasts',config,'Original-MD q6 trajectories and autoregressive forecasts for the same three example events at the shortest available lead bin. Grey shading marks the available past span; only the three selected snapshots entered the predictor, not the dense line shown as truth. Dashed vertical lines mark actual sustained onset. Model checkpoint replay reproduced saved event CDFs within2e-5; physical states use the saved training normalizers.')
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for i in range(2):
        for j,d in enumerate(DOMAINS):
            ax=axes[i,j];q=a[d+'-mixture'][i,:,:,257];low,high=np.quantile(q,[.05,.95],axis=0);time=np.arange(1,33)*3
            ax.fill_between(time,low,high,color=COLORS[d],alpha=.16,lw=0)
            for sample in q[:4]:ax.plot(time,sample,c=COLORS[d],lw=.5,alpha=.2)
            ax.plot(time,q.mean(0),c=COLORS[d]);ax.plot(time,a['truth'][i,:,257],c='#30343a',lw=1.2)
            ax.axvline(meta['cases'][i]['lead_ps'],c='#333',ls='--',lw=.8)
            ax.set(xlabel='Forecast horizon (ps)',ylabel=r'$q_6$',xticks=[0,24,48,72,96]);panel(ax,chr(97+i*2+j),f'Event {i+1} · '+LABELS[d])
    fig.tight_layout(h_pad=2,w_pad=2);finish(fig,root,'10_mixture_trajectories',config,'Mixture forecasts for the first two fixed examples:32 replay samples, mean, and pointwise5th–95th percentile envelope; black is original MD truth. These are model sample intervals, not demonstrated90% coverage. Individual paths are not independent posterior parameter draws. Example choice is identical across domains.')
    a=np.load(root/'technical/umap.npz');fig,axes=plt.subplots(2,2,figsize=(9,7))
    for j,d in enumerate(DOMAINS):
        xy=a[d+'_xy'];test=~a['train']
        axes[0,j].scatter(*xy[a['train']].T,s=3,c='#d8dadd',alpha=.25,rasterized=True)
        points=axes[0,j].scatter(*xy[test].T,c=a['temperature'][test],s=7,cmap='viridis',vmin=400,vmax=520,alpha=.75,linewidths=0)
        axes[1,j].scatter(*xy[test & ~a['event12']].T,c='#b8c7d3',s=7,alpha=.5,linewidths=0)
        axes[1,j].scatter(*xy[test & a['event12']].T,c='#cf663d',s=22,alpha=.9,linewidths=.3,edgecolors='white')
        for i in range(2):
            axes[i,j].set(xticks=[],yticks=[],xlabel='UMAP 1',ylabel='UMAP 2');panel(axes[i,j],chr(97+i*2+j),LABELS[d]);axes[i,j].set_box_aspect(.75)
    fig.colorbar(points,ax=list(axes[0]),label='Temperature (K)',fraction=.025,pad=.025)
    axes[1,1].legend(handles=[Line2D([],[],marker='o',ls='',c='#b8c7d3',label='No onset within 12 ps'),Line2D([],[],marker='o',ls='',c='#cf663d',label='Onset within 12 ps')],loc='lower center',bbox_to_anchor=(.4,-.35),ncol=2,fontsize=8)
    fig.subplots_adjust(left=.08,right=.84,bottom=.13,top=.95,wspace=.28,hspace=.4)
    finish(fig,root,'11_embedding_umap',config,'Separate UMAP maps of instantaneous central embeddings. Each domain uses the same sampled windows (up to24 per source); StandardScaler and UMAP fit only training sources, then transform held-out sources. Top: grey training reference and test temperature. Bottom: test windows colored by future12 ps onset; these labels were not used to fit UMAP. Axes between encoders are not aligned, and apparent clusters do not establish predictive sufficiency.')


def camera():
    theta,phi=.38,.55
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])@np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])


def real_cloud_plots(config,root):
    from scipy.spatial.distance import pdist,squareform
    a=np.load(root/'technical/paired-clouds.npz');meta=json.loads((root/'technical/paired-clouds.json').read_text());r=camera()
    kinds=['liquid','before_onset','crystalline'];names=['Liquid','Before onset','Crystalline']
    fig,axes=plt.subplots(3,3,figsize=(10,9.4))
    for row,(kind,name) in enumerate(zip(kinds,names)):
        for col,d in enumerate(DOMAINS):
            ax=axes[row,col];raw=a[kind+'_'+d];x=raw@r;dist=squareform(pdist(raw));ii,jj=np.where(np.triu((dist>0)&(dist<3.5),1))
            for i,j in zip(ii,jj):ax.plot(x[[i,j],0],x[[i,j],1],c=COLORS[d],alpha=.14,lw=.5,zorder=1)
            order=np.argsort(x[:,2]);ax.scatter(*x[order,:2].T,s=25,c=COLORS[d],alpha=.8,edgecolors='white',linewidths=.3,zorder=2);ax.scatter(*x[0,:2],s=60,c='#26303b',edgecolors='white',linewidths=.6,zorder=3)
        ax=axes[row,2];x=a[kind+'_observed']@r;u=a[kind+'_displacement']@r
        ax.scatter(*x[:,:2].T,s=10,c='#b5bdc4',zorder=1);ax.quiver(x[:,0],x[:,1],u[:,0],u[:,1],angles='xy',scale_units='xy',scale=1,color='#7b668e',width=.005,headwidth=3,zorder=2)
        for col,ax in enumerate(axes[row]):
            ax.set(xlim=(-8,8),ylim=(-8,8),aspect='equal');ax.set_axis_off()
            if row==0:ax.set_title(['Original coordinates','Relaxed coordinates','Atomic displacements'][col],fontsize=10)
            ax.text(-7.6,7.3,chr(97+row*3+col),weight='bold',fontsize=11)
            if col==0:ax.text(-9.8,0,name,rotation=90,va='center',ha='center',fontsize=10)
        axes[row,0].plot([-7,-5],[-7,-7],c='#333',lw=2);axes[row,0].text(-6,-7.8,'2 Å',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.99,bottom=.03,top=.95,wspace=.02,hspace=.05)
    finish(fig,root,'12_original_vs_relaxed_atoms',config,'Real matched atomic neighborhoods from the reused test archives: one originally liquid, one0–12 ps before original-MD sustained onset, and one originally crystalline. Every pair has exactly the same80 observed-nearest atom IDs, centered on the same tracked atom (dark marker), with one common orthographic camera and scale. Thin lines connect pairs closer than3.5 Å in each displayed structure. No alignment or synthetic lattice is used. Right: center-relative observed-to-relaxed displacement arrows at their true scale; none of the panels is magnified or deformed. State names refer to original MD, not reassigned relaxed PTM. Exact identities and RMS displacements are in technical/paired-clouds.json.')
    fig,axes=plt.subplots(1,2,figsize=(11,5.6));q=stencil()@r;colors=['#25303b']+['#3296b8']*12+['#d9953c']*12
    for ax,d,letter in zip(axes,DOMAINS,'ab'):
        full=a['context_'+d]@r;centers=a['queries_'+d]@r
        ax.scatter(*full[:,:2].T,s=2,c='#b7c0c9',alpha=.15,linewidths=0)
        for i in np.argsort(centers[:,2]):
            points=a[f'context_observed_cloud_{i}'] if d=='observed' else a['clouds_'+d][i]
            points=points[np.linalg.norm(points,axis=1)<meta['context']['radius_A']]
            cloud=points@r+centers[i]
            ax.scatter(*cloud[:,:2].T,s=4,c=colors[i],alpha=.42,linewidths=0)
            ax.add_patch(Circle(centers[i,:2],meta['context']['radius_A'],fill=False,ec=colors[i],lw=.6,alpha=.7))
        ax.scatter(*q[:,:2].T,s=24,facecolors='none',edgecolors=colors,linewidths=.7)
        ax.scatter(*centers[:,:2].T,s=11,c=colors,edgecolors='white',linewidths=.2)
        ax.set(xlim=(-31,31),ylim=(-31,31),aspect='equal',xlabel='Projected position (Å)',ylabel='Projected position (Å)',xticks=[-20,0,20],yticks=[-20,0,20]);panel(ax,letter,'Original context' if d=='observed' else 'Relaxed context')
    fig.tight_layout(w_pad=3)
    finish(fig,root,'13_full_spatial_context',config,'Whole25-slot context for the same pre-onset sample in the paired-atom illustration. Center plus12 slots at10 Å and12 at20 Å; hollow markers are nominal symmetric queries, filled markers the actual assigned atoms. Query IDs are selected once in observed geometry and retained after relaxation. All25 neighborhoods are shown. Circles mark the maximum7.94 Å support. Original MACE uses all neighbors inside this radius; the relaxed checkpoint retains its80 observed-nearest candidates and crops them after relaxation. Thus the actual local supports differ, as in training. Colors distinguish query shells, not phase labels. Projection can overlap supports; the queries are symmetric in3D.')


def learning(config,root):
    base=resolve_path(config['input'])/'technical/runs';fig,axes=plt.subplots(2,2,figsize=(8.8,6.3),sharex=True)
    for ax,m,letter in zip(axes.flat,METHODS,'abcd'):
        for d in DOMAINS:
            folder=base/f'{d}-mace-{m}-reuse-E36';metrics=json.loads((folder/'metrics.json').read_text());per=metrics['training']['updates_per_epoch']
            records=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
            x=[r['step']/per for r in records];y=[r['selection_brier'] for r in records];ax.plot(x,y,c=COLORS[d])
            ax.scatter(metrics['training']['selected_step']/per,metrics['training']['best_selection_brier'],c=COLORS[d],s=60,marker='*',zorder=3)
        ax.set(xlabel='Training epoch',ylabel='Selection Brier score',xlim=(0,36),xticks=[0,12,24,36]);panel(ax,letter,NAMES[m])
    legend_domains(fig);fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    finish(fig,root,'14_learning_curves',config,'Open-loop selection-source integrated Brier by epoch. Stars mark the selected checkpoint. Curves end at the actual early-stopping point or36-epoch ceiling. Test outcomes were not used for checkpoint selection; this is a one-seed optimization trace.')


def render(config,root,plan,corpus,predictions,metrics,matched,summary,examples):
    style();summary_plots(config,root,corpus,predictions,metrics);event_plots(config,root,predictions,matched,summary,examples)
    replay_plots(config,root,corpus);real_cloud_plots(config,root);learning(config,root)
    text=['# Relaxed versus original MACE: matched forecast figures','',
        'PNG only,300 dpi. Completed eight-fit comparison; existing relaxed archives, no new simulation. All plots use the matched reused-data cohort; these are not direct replications of the earlier dense-history comparison.','']
    cards=[]
    for name,caption in CAPTIONS.items():
        title=name[3:].replace('_',' ').capitalize();text += [f'## {title}','',f'![{title}](plots/{name}.png)','',caption,'']
        cards.append(f'<section><h2>{html.escape(title)}</h2><a href="plots/{name}.png"><img loading="lazy" src="plots/{name}.png" alt="{html.escape(title)}"></a><p>{html.escape(caption)}</p></section>')
    (root/'README.md').write_text('\n'.join(text))
    (root/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Relaxed MACE forecast comparison</title><style>body{font:16px/1.55 system-ui;color:#27313d;max-width:1100px;margin:40px auto;padding:0 20px}section{margin:50px 0}img{width:100%;height:auto}h1,h2{font-weight:550}p{max-width:980px}</style></head><body><h1>Relaxed versus original MACE</h1><p>Matched archived-data forecasts. One seed; PNG figures with separate captions.</p>'+''.join(cards)+'</body></html>')
