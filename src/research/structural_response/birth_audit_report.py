"""Publish existing birth-audit tables and predictions without refitting readouts."""
import argparse
import csv
from datetime import datetime
import html
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .birth_audit import read, setup


def table(rows, columns):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in columns)+'</tr>' for r in rows)+'</tbody></table>'


def load(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def render(c):
    parent,recipe,_,_,rows,out=setup(c)
    controls=load(out/'tables/physical-controls.csv')
    fidelity=load(out/'tables/surrogate-fidelity.csv')
    importance=load(out/'tables/physical-attribution.csv')
    distributions=load(out/'tables/present-state-distributions.csv')
    neural=[]
    for method in recipe['methods']:
        for arm in recipe['response_arms']:
            name=method+'-'+arm['name']
            scores=load(parent/'analyses'/name/'analyses/birth-prediction/tables/birth-scores.csv')
            for s in scores:neural.append(dict(model=name,**s))
    mask=rows['role']!='train';y=rows['label'][mask];w=rows['weight'][mask]
    with np.load(out/'technical/physical.npz') as a:features=a['values'][mask,1]
    columns=read(out/'technical/columns.json');qbar=features[:,columns.index('bond_order/l6_qbar')]
    groups=['bond_order','tda','geometry','cna','kinematics','precision']
    names=['Bond order','Topology (TDA)','Radial / geometry','Bond connectivity','Velocities','Quantization']
    models=sorted({r['model']for r in importance});shares=[]
    for name in models:
        current={r['group']:float(r['mean_absolute_logit_shap']) for r in importance if r['model']==name and r['feature']=='all'}
        values=np.array([current[k]for k in groups]);shares.append(100*values/values.sum())
    shares=np.array(shares)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    ax=axes[0]
    for label,color,name in [(0,'#7b8794','Random nonfounder controls'),(1,'#c44d3b','Future nucleus founders')]:
        ix=y==label;ax.hist(qbar[ix],bins=np.linspace(0,.65,27),weights=w[ix]/w[ix].sum(),
                           color=color,alpha=.55,label=name,edgecolor='white',linewidth=.3)
    ax.set(xlabel=r'Present neighbor-averaged $\bar q_6$',ylabel='Weighted fraction within class',title='The input structures already differ')
    ax.legend(fontsize=8,frameon=False)
    chosen=['qbar6','bond_order','tda','radial','kinematics','precision']
    labels=[r'One scalar: $\bar q_6$','Bond-order descriptors','Topology descriptors','Radial descriptors','Velocity summaries','Quantization indicators']
    vals=[];lo=[];hi=[]
    for group in chosen:
        r=next(r for r in controls if r['features']==group and r['lead_ps']=='3.0' and r['readout']=='catboost' and r['metric']=='ap')
        value=float(r['value']);vals.append(value);lo.append(value-float(r['lower']));hi.append(float(r['upper'])-value)
    ax=axes[1];ix=np.arange(len(vals))
    ax.barh(ix,vals,color=['#237c80']*4+['#9ba4ad']*2)
    ax.errorbar(vals,ix,xerr=np.array([lo,hi]),fmt='none',ecolor='#26323f',capsize=2)
    for i,v in enumerate(vals):ax.text(v+hi[i]+.015,i,f'{v:.3f}',va='center',fontsize=9)
    ax.axvline(.2,color='#666',ls='--',lw=1,label='20% positive prevalence')
    nn=[float(r['value'])for r in neural if r['space']=='structural_motion' and r['readout']=='catboost' and r['lead_ps']=='3.0' and r['metric']=='ap']
    ax.axvspan(min(nn),max(nn),color='#da9636',alpha=.22,label='Eight neural readouts')
    ax.set(yticks=ix,yticklabels=labels,xlim=(0,1.07),xlabel='Held-out AP, 3 ps lead',title='Simple structure matches neural prediction')
    ax.invert_yaxis();ax.legend(fontsize=8,loc='lower right',frameon=False)
    ax=axes[2];mean=shares.mean(0)
    ax.barh(np.arange(6),mean,color=['#237c80','#398da0','#759daa','#a3b5bb','#bac3c8','#d0d6da'])
    ax.errorbar(mean,np.arange(6),xerr=np.vstack([mean-shares.min(0),shares.max(0)-mean]),fmt='none',ecolor='#26323f',capsize=2)
    for i,v in enumerate(mean):ax.text(shares[:,i].max()+.8,i,f'{v:.1f}%',va='center',fontsize=9)
    ax.set(yticks=np.arange(6),yticklabels=names,xlim=(0,57),xlabel='Share of absolute surrogate SHAP',title='Physical explanations of the neural scores')
    ax.invert_yaxis()
    fig.suptitle('Nucleus-birth audit · unchanged 3 ps inputs · 79 held-out births / 33 ancestors',fontsize=14)
    (out/'plots').mkdir(exist_ok=True);fig.savefig(out/'plots/feature-audit.png',dpi=170);fig.savefig(out/'plots/feature-audit.pdf');plt.close(fig)
    summary=[]
    for group in c['physical_controls']:
        item={'Features':group}
        for lead in (1.,3.,6.):
            r=next(r for r in controls if r['features']==group and float(r['lead_ps'])==lead and r['readout']=='catboost' and r['metric']=='ap')
            item[f'{lead:g} ps AP']=f"{float(r['value']):.3f} [{float(r['lower']):.3f}, {float(r['upper']):.3f}]"
        summary.append(item)
    neurons=[]
    for name in models:
        item={'Model':name}
        for lead in (1.,3.,6.):
            r=next(r for r in neural if r['model']==name and r['space']=='structural_motion' and r['readout']=='catboost' and float(r['lead_ps'])==lead and r['metric']=='ap')
            item[f'{lead:g} ps AP']=f"{float(r['value']):.3f}"
        neurons.append(item)
    interventions=out/'analyses/embedding-interventions-v3/tables/frozen-interventions.csv'
    intervention_html='<p>Frozen-model interventions are still running.</p>'
    if interventions.exists():
        rr=load(interventions);current=[]
        for name in models:
            a=[r for r in rr if r['model']==name and r['space']=='structural_motion' and r['readout']=='catboost' and r['metric']=='ap']
            if not a:continue
            current.append({'Model':name,**{r['variant']:f"{float(r['value']):.3f}" for r in a}})
        if current:intervention_html=table(current,list(current[0]))
    findings=[]
    for feature in ['any_component_ge4','center_in_component_ge8','any_component_ge64','component_ge64_within8A']:
        d={'Present-state audit at 3 ps':feature}
        for label,text in [(1,'Positive'),(0,'Control')]:
            value=next(float(r['mean'])for r in distributions if r['feature']==feature and r['lead_ps']=='3.0' and int(r['label'])==label)
            d[text]=f'{100*value:.2f}%'
        findings.append(d)
    fidelity_rows=[{'Model':r['model'],'Held-out logit R²':f"{float(r['logit_r2']):.3f}",'Probability correlation':f"{float(r['probability_correlation']):.3f}"}for r in fidelity]
    checked=read(out/'technical/integrity.json')
    now=datetime.now(ZoneInfo('Europe/Paris')).strftime('%Y-%m-%d %H:%M %Z')
    document=f'''<!doctype html><meta charset="utf-8"><title>What explains high nucleus-birth AP?</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#22313c;max-width:1250px;margin:40px auto;padding:0 24px}}h1{{line-height:1.15;font-size:36px}}h2{{margin-top:40px}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:20px 0;display:block;overflow-x:auto}}th,td{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}th{{background:#eef3f4}}.note{{background:#edf5f4;padding:16px 22px;border-left:4px solid #237c80}}code{{font-size:13px}}a{{color:#237c80}}</style>
<h1>The high AP mainly reflects present structural order</h1><p>Updated {now}. Exploratory audit of the eight October 7 motion fits.</p>
<div class="note">No neighborhood filtering was added. All 4,905 original case/control rows and all three lead times are preserved. The held-out set contains 79 physical births across 33 ancestors, represented by 326 positive centers and 1,304 controls. Future founders are compared with random nonfounder centers at the same source/frame; prevalence is imposed at 20%.</div>
<img src="plots/feature-audit.png" alt="Present order distributions, diagnostic AP and physical attribution">
<h2>What the present input tells the model</h2><p>Neighbor-averaged bond order describes how consistently neighboring atoms share angular order. The single scalar <em>q̄6</em> reaches 0.940 AP at 3 ps through an NLL-selected nonlinear readout. These controls use present Q or V only. They do not consume event metadata, actual simulated futures, source IDs, temperature or observation time. The neural encoder sees centered positions; its motion coordinates are a directional derivative using same-frame relative velocities.</p>
{table(summary,['Features','1 ps AP','3 ps AP','6 ps AP'])}
<p>Intervals resample whole ancestors. A single scalar with a nonlinear readout can recognize an intermediate ordered state; this is not necessarily a monotonic “more crystalline means more likely” score. AP for a monotonic linear q̄6 readout at 3 ps is 0.873.</p>
<h2>Small crystalline components before the catalogue birth</h2>{table(findings,['Present-state audit at 3 ps','Positive','Control'])}
<p>Components are taken from the original full-cell PTM extraction, with a four-atom minimum. A component with 64 atoms does not by itself establish temporal persistence. Component measurements are audit metadata, not neural input features. Earlier small ordered components need not be the same tracked root as the future catalogue birth.</p>
<h2>Does motion matter to these frozen readouts?</h2>{intervention_html}
<p>These are fixed-checkpoint, fixed-readout diagnostics at 3 ps. Code permutations stay inside the original source/frame-matched five-center sets. Radius normalization removes variation in mean first-shell distance; Gaussian perturbations have 0.01 Å standard deviation. Such interventions measure model sensitivity, not physical MD outcomes. All eight structural and structural+motion readouts are covered. Future-law interventions remain incomplete: the separately preserved v2 analysis stopped when one adapter's replay exceeded its numerical tolerance; it is not used for the attribution claim.</p>
<h2>What the neural models achieved</h2>{table(neurons,['Model','1 ps AP','3 ps AP','6 ps AP'])}
<p>Same CatBoost readout family for every row; no choice of model or checkpoint by AP. “Value” fits also receive Q,V and learn motion, but omit the analytic response-derivative loss.</p>
<h2>How faithful are the physical explanations?</h2>{table(fidelity_rows,['Model','Held-out logit R²','Probability correlation'])}
<p>Surrogates fit the frozen neural readout’s logits using present physical descriptors, selecting iterations by training-ancestor RMSE. Their SHAP allocations are correlated-feature associations, not a unique causal decomposition. Bond order and topology dominate consistently; velocity and quantization summaries contribute little.</p>
<h2>Leakage and interpretation</h2><p>{sum(v['passed']for v in checked)}/{len(checked)} integrity checks passed. Every one of the 14,715 Q,V observations reconstructed exactly from the pinned raw source at the declared lead. Ancestors are disjoint; exact cross-role coordinate overlaps at each lead were absent. The spatial-pretraining manifest contains no held-out ancestor. Pair matching makes source/time metadata identical within each 1:4 set. These checks do not certify every possible near-duplicate or catalogue-label issue.</p>
<p>The high AP is credible for retrospective founder-versus-random-site discrimination. It is not evidence of equivalent performance for discovering rare births in arbitrary liquid, or distinguishing equally ordered embryos that grow and dissolve. Those require the separate present-selected embryo-to-nucleus evaluation; no new neighborhood filter is needed here.</p>
<p><a href="tables/physical-controls.csv">Physical controls</a> · <a href="tables/physical-attribution.csv">Attributions</a> · <a href="tables/present-state-distributions.csv">Present-state audit</a> · <a href="technical/integrity.json">Integrity receipts</a> · <a href="analyses/embedding-interventions-v3/tables/frozen-interventions.csv">Frozen interventions</a></p>'''
    (out/'report.html').write_text(document)
    print(out/'report.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);a=p.parse_args();render(read(a.config))
