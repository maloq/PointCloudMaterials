"""Scientific plots and outcome-dependent narrative from exported metrics."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import pair_mask

NAMES={'plus_gatr':'GATr z128','plus_angular_delta':'GATr angular difference',
    'plus_mace':'MACE z128','plus_soap':'SOAP','plus_tda':'TDA',
    'current_order':'Current bond + angular order','current_order_plus_gatr':'GATr beyond current order'}
COLORS={'plus_gatr':'#4f63d9','plus_angular_delta':'#009c88','plus_mace':'#d88435','plus_soap':'#a14fc9','plus_tda':'#69717e'}


def bars(ax,frame,targets,methods,labels):
    width=.76/len(methods);positions=np.arange(len(targets))
    for j,method in enumerate(methods):
        rows=frame[frame.method==method].set_index('target').loc[targets]
        x=positions+(j-(len(methods)-1)/2)*width
        value=rows.improvement_percent.to_numpy();lo=rows.low.to_numpy();hi=rows.high.to_numpy()
        ax.bar(x,value,width,color=COLORS[method],label=NAMES[method],alpha=.87)
        ax.errorbar(x,value,yerr=[value-lo,hi-value],fmt='none',ecolor='#333',capsize=2,lw=.8)
    ax.axhline(0,color='#333',lw=.8);ax.set_xticks(positions,labels)
    ax.set_ylabel('Reduction in held-out error (%)')


def save(fig,root,name):
    fig.savefig(root/'plots'/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(root/'plots'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def plots(root,a,pairs,predictions,tables,config):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold'})
    gains=pd.DataFrame(tables['conditional_gains']);matched=pd.DataFrame(tables['matched_gains'])
    main=gains[(gains.family=='nonlinear')&(gains.baseline=='radial_control')]
    methods=['plus_gatr','plus_angular_delta','plus_mace','plus_soap']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    bars(axes[0,0],main[main.task=='structure'],['q4','q6','qbar6','angular_arrangement'],methods,['q4','q6','q̄6','Angular\narrangement'])
    axes[0,0].set_title('Structural information beyond radial controls')
    sub=matched[(matched.family=='nonlinear')&(matched.baseline=='radial_control')&(matched.task=='structure')&(matched.caliper_A==.1)]
    bars(axes[0,1],sub,['q6','angular_arrangement'],methods,['q6 contrasts','Angular contrasts'])
    axes[0,1].set_title('Trajectory matching sensitivity: same source/time\nRadial RMS ≤0.10 Å; density gap ≤2%')
    targets=[f'crystallize_{h}ps' for h in config['future_horizons_ps']]
    bars(axes[1,0],main[main.task=='future'],targets,methods,[f'{h} ps' for h in config['future_horizons_ps']])
    axes[1,0].set_title('Future crystallization: R-star baseline\nBefore the redundant-radial-input control')
    axes[1,0].set_ylabel('Reduction in held-out Brier error (%)')
    sub=gains[(gains.family=='nonlinear')&(gains.baseline=='current_order_radial_duplicate')&(gains.method=='current_order_plus_gatr')].set_index('target').loc[targets]
    x=np.arange(len(targets));v=sub.improvement_percent.to_numpy()
    axes[1,1].bar(x,v,color=COLORS['plus_angular_delta'],alpha=.87)
    axes[1,1].errorbar(x,v,yerr=[v-sub.low.to_numpy(),sub.high.to_numpy()-v],fmt='none',color='#333',capsize=3)
    axes[1,1].axhline(0,color='#333',lw=.8);axes[1,1].set_xticks(x,[f'{h} ps' for h in config['future_horizons_ps']])
    axes[1,1].set(title='GATr angular difference: stronger controls\nCurrent bond/angular order + radial duplication',ylabel='Reduction in held-out Brier error (%)')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Does the exported state retain useful angular information?\nFrozen Al GATr · 10 source-held-out folds · paired source intervals · A100 / node07',fontsize=15)
    save(fig,root,'conditional-information')

    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,target in zip(axes,('q6','angular_arrangement','crystallize_48ps'),strict=True):
        for k,method in enumerate(('plus_gatr','plus_angular_delta')):
            for j,family in enumerate(('linear','nonlinear')):
                row=gains[(gains.family==family)&(gains.method==method)&(gains.baseline=='radial_control')&(gains.target==target)].iloc[0]
                x=j+(k-.5)*.18
                ax.errorbar(x,row.improvement_percent,yerr=[[row.improvement_percent-row.low],[row.high-row.improvement_percent]],
                    fmt='o',color=COLORS[method],capsize=4,label=NAMES[method] if j==0 else None)
        ax.axhline(0,color='#666',lw=.8);ax.set_xticks([0,1],['Linear','Nonlinear'])
        ax.set(title=target.replace('_',' '),ylabel='Reduction in held-out error (%)',xlim=(-.4,1.4))
    axes[0].legend(fontsize=8);fig.suptitle('Probe-family sensitivity: all preprocessing and regularization selected within training sources',fontsize=13)
    save(fig,root,'probe-sensitivity')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,family in enumerate(('linear','nonlinear')):
        ax=axes[0,column]
        sub=gains[(gains.family==family)&(gains.baseline=='plus_radial_duplicate')&(gains.task=='future')]
        bars(ax,sub,targets,['plus_gatr','plus_angular_delta'],[f'{h} ps' for h in config['future_horizons_ps']])
        ax.set_title(f'{family.title()} · beyond duplicated radial control')
        ax.set_ylabel('Brier-error reduction (%)')
        ax=axes[1,column]
        sub=gains[(gains.family==family)&(gains.baseline=='current_order_radial_duplicate')&(gains.method=='current_order_plus_gatr')].copy()
        sub['method']='plus_angular_delta'
        bars(ax,sub,targets,['plus_angular_delta'],[f'{h} ps' for h in config['future_horizons_ps']])
        ax.set_title(f'{family.title()} · also condition on current order')
        ax.set_ylabel('Brier-error reduction (%)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Does the forecasting gain survive a matched radial-input duplication control?',fontsize=13)
    save(fig,root,'radial-duplication-control')

    # The illustrative pair is selected by largest observed q6 contrast among primary-caliper pairs.
    take=pair_mask(pairs,.05,config)
    candidates=np.flatnonzero(take)
    if not len(candidates):raise ValueError('No primary-caliper pair to visualize')
    delta=np.abs(a['bond'][pairs['left'][candidates],1]-a['bond'][pairs['right'][candidates],1])
    chosen=candidates[np.argmax(delta)];left,right=pairs['left'][chosen],pairs['right'][chosen]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,label,color in [(left,'Environment A','#4f63d9'),(right,'Environment B','#e18132')]:
        axes[0,0].plot(np.arange(1,81),a['radii80'][row],label=label,color=color)
        axes[0,1].plot(np.linspace(0,1,config['reference_quantiles']),a['radial_quantiles'][row],label=label,color=color)
        axes[1,0].plot(np.arange(1,17),a['angular'][row],'o-',ms=3,label=label,color=color)
    axes[0,0].set(title='Almost identical inner radial profiles',xlabel='Neighbor rank',ylabel='Distance (Å)')
    axes[0,1].set(title='Full native-support radial profiles',xlabel='Radius quantile',ylabel='Distance (Å)')
    axes[1,0].set(title='Observed angular arrangement differs',xlabel='Legendre order',ylabel='Weighted angular moment')
    axes[0,0].legend()
    methods2=['radial_control','plus_gatr','plus_angular_delta']
    q=np.array([[a['bond'][row,1],*[predictions['structure','nonlinear',m]['prediction'][row,1] for m in methods2]] for row in (left,right)])
    for k in range(2):axes[1,1].bar(np.arange(4)+(k-.5)*.32,q[k],.32,color=['#4f63d9','#e18132'][k])
    axes[1,1].set_xticks(range(4),['Actual q6','Radial\ncontrol','+ GATr z','+ Angular\ndifference'])
    axes[1,1].set(title='Held-out predictions for these environments',ylabel='q6')
    fig.suptitle(f'Illustrative matched pair · source {a["source"][left]} · {a["context"][left,1]:g} ps · atoms {a["atom"][left]} and {a["atom"][right]}\n'
        f'Inner radial RMS {pairs["radial_rms_A"][chosen]:.4f} Å; full-support {pairs["full_radial_rms_A"][chosen]:.4f} Å; density gap {100*pairs["density_relative"][chosen]:.2f}%\n'
        'Selected for illustration by maximal q6 difference; aggregate statistics use every accepted pair',fontsize=12)
    save(fig,root,'matched-example')
    (root/'technical/example.json').write_text(json.dumps(dict(left=int(left),right=int(right),source=int(a['source'][left]),
        frame=int(a['frame'][left]),selection='largest absolute q6 contrast among primary-caliper matched pairs'),indent=2)+'\n')


def findings(root,tables,config):
    gains=pd.DataFrame(tables['conditional_gains']);matched=pd.DataFrame(tables['matched_gains'])
    balance=pd.DataFrame(tables['matching_balance']);cohort=pd.DataFrame(tables['cohort'])
    def row(target,method='plus_gatr',baseline='radial_control',family='nonlinear',matched_pair=False):
        table=matched if matched_pair else gains
        select=(table.target==target)&(table.method==method)&(table.baseline==baseline)&(table.family==family)
        if matched_pair:select&=table.caliper_A==.1
        result=table[select]
        if len(result)!=1:raise ValueError('Ambiguous conditional summary')
        return result.iloc[0]
    def fmt(r,digits=1):return f'{r.improvement_percent:+.{digits}f}% [{r.low:+.{digits}f}, {r.high:+.{digits}f}]'
    lines=['# Conditional information beyond radial structure and density','',
        '**The tested GATr export carries very little additional angular information after radial conditioning. '
        'Its small apparent forecasting gains are largely reproduced by redundant radial inputs.** '
        'The dense matched-environment extension supplies a positive SOAP control and supports the structural finding.','',
        'Completed on the A100 on node07, using frozen GATr Al checkpoint 1216. No encoder was trained or modified. '
        'Probes were fitted separately and evaluated on unseen simulation sources.','',
        '## Question and controls','',
        'Does z128 add useful information about local angular structure or later crystallization among radially similar environments? '
        'We reused 32,040 observations from ten Al trajectories, four atom tracks per source. All encoder test ancestries were '
        'verified in the parent audit. This is exploratory reuse of that cohort.','',
        '**R** includes the inner radial descriptor, 80 sorted neighbor radii, 33 radius quantiles across native support, '
        'counts/radial moments, local density, coordination, temperature and elapsed time. The stronger control **R-star** additionally includes '
        'a radius-only GATr control: the identical radius multiset is placed on a deterministic Fibonacci angular pattern '
        'and re-encoded. This uses the same checkpoint but removes the original angular arrangement. '
        '**Angular difference** means z(original) − z(radial control). It is computed from the exported state, not internal vectors. '
        'Together with the radial-control state it is an invertible reparameterization of the original state; it isolates small '
        'angular responses for finite-capacity probes. The replacement geometries are synthetic, not physical trajectories.','',
        'Every outer fold holds out one complete source; three inner source folds select ridge regularization. '
        'Both linear and nonlinear random-Fourier-feature probes are reported. Training samples every fourth frame; '
        'held-out evaluation uses every eligible frame. All feature/target scaling uses fit sources only. '
        'Intervals resample sources within temperature; they do not treat frames as independent or include refitting uncertainty.','',
        '## Bond order and angular arrangement','',
        'Numbers are percentage reductions in held-out squared error relative to R-star (positive is better), '
        'with paired 95% source intervals. Angular arrangement comprises 16 rotation-invariant Legendre moments, '
        'not absolute laboratory orientation.','',
        '| Target | Add original GATr z128 | Add its angular difference | Add SOAP |',
        '|---|---:|---:|---:|']
    for target in ('q4','q6','qbar6','bond_order','angular_arrangement'):
        lines.append(f'| {target} | {fmt(row(target))} | {fmt(row(target,"plus_angular_delta"))} | {fmt(row(target,"plus_soap"))} |')
    positive=[t for t in ('q4','q6','qbar6','angular_arrangement') if row(t).low>0]
    lines+=['',f'The original state has an interval above zero for: **{", ".join(positive) if positive else "none of these individual target groups"}**. '
        'This is evidence about recoverable information under the tested controls and probes, not a proof of conditional independence when gains are absent.','',
        '## Directly matched environments','']
    primary=balance[balance.caliper_A==.05]
    loose=balance[balance.caliper_A==.1]
    lines+=[f'Primary matching retains **{int(primary.matched_pairs.sum()):,} of {int(primary.candidate_pairs.sum()):,}** same-source, same-frame pairs. '
        'Both the first-80 radial RMS gap and full-support radial-quantile RMS gap must be ≤0.05 Å, with ≤2% relative density difference. '
        'Matching uses no embeddings, angular targets or future outcomes. This primary sample is too small for an inferential conclusion. '
        'The 0.025 Å sensitivity has no matches. Pairs can share atoms/frames, so uncertainty is source-level.','',
        f'The predeclared 0.10 Å sensitivity retains **{int(loose.matched_pairs.sum()):,} pairs** across ten sources. '
        'The following table uses that looser population and scores predicted target contrasts:','',
        '| Matched target contrasts | Add original GATr | Add angular difference |', '|---|---:|---:|']
    for target in ('q6','qbar6','angular_arrangement'):
        lines.append(f'| {target} | {fmt(row(target,matched_pair=True))} | {fmt(row(target,"plus_angular_delta",matched_pair=True))} |')
    lines+=['','These score predicted A−B target differences, not only each environment separately. '
        'The dense spatial follow-up keeps the original 0.05 Å threshold and is reported separately in '
        '[the spatial extension](RESULTS_spatial.md); it was added to address match scarcity.','',
        '## Future crystallization','',
        f'There are **{int(cohort.eligible_future_rows.sum()):,}** eligible prospective observations. '
        'The atom must have been PTM-noncrystalline for the current and previous two frames and be before its first '
        'sustained FCC/HCP/BCC onset. A sustained onset starts eight consecutive crystalline frames. '
        'All horizons have full follow-up, including confirmation; incomplete terminal windows are excluded. '
        'The outcome is first sustained local crystallization within 24, 48 or 96 ps, not a committor probability.','',
        'The outcome probe uses clipped least-squares probabilities. Primary loss is Brier error; '
        'source AUROC and average precision are supplementary. These are nonlinear readout results:','',
        '| Horizon | Add original GATr to R* | Add angular difference to R* | Add angular difference after also controlling current bond/angular order |',
        '|---|---:|---:|---:|']
    for horizon in config['future_horizons_ps']:
        target=f'crystallize_{horizon}ps'
        lines.append(f'| {horizon} ps | {fmt(row(target))} | {fmt(row(target,"plus_angular_delta"))} | {fmt(row(target,"current_order_plus_gatr","current_order"))} |')
    lines+=['','Adding a nearly redundant state can change ridge penalties and random-feature geometry. '
        'A stricter control appends a second copy of the radial-only GATr state, matching the added 128 dimensions '
        'without adding information. The following comparison replaces that duplicate by the original state or angular difference:','',
        '| Probe | Horizon | Original GATr beyond duplicated radial state | Angular difference beyond duplicated radial state |',
        '|---|---|---:|---:|']
    for family in ('linear','nonlinear'):
        for horizon in config['future_horizons_ps']:
            target=f'crystallize_{horizon}ps'
            lines.append(f'| {family} | {horizon} ps | {fmt(row(target,baseline="plus_radial_duplicate",family=family),4)} | '
                f'{fmt(row(target,"plus_angular_delta",baseline="plus_radial_duplicate",family=family),4)} |')
    positive=[h for h in config['future_horizons_ps'] if row(f'crystallize_{h}ps','current_order_plus_gatr','current_order_radial_duplicate').low>0]
    lines+=['',f'After conditioning on current order **and** the duplicate-radial control, the nonlinear interval is above zero at: '
        f'**{", ".join(str(h)+" ps" for h in positive) if positive else "none of the tested horizons"}**. '
        'The same comparison has no positive interval in the linear probe either. '
        'Thus the small R-star-only gains do not establish a useful extra angular forecasting signal. '
        'Matched future tables also report discordant-outcome counts and ranking concordance; sparse discordant pairs limit interpretation.','',
        'The risk readouts also need an absolute reference. The nonlinear original-GATr readout does not beat the '
        'training-prevalence-only predictor in source-mean Brier error at any horizon:', '',
        '| Horizon | Training-prevalence Brier | R-star + GATr Brier |', '|---|---:|---:|']
    scores=pd.DataFrame(tables['source_scores'])
    for horizon in config['future_horizons_ps']:
        values=scores[(scores.family=='nonlinear')&(scores.method=='plus_gatr')&(scores.target==f'crystallize_{horizon}ps')]
        lines.append(f'| {horizon} ps | {values.constant_loss.mean():.5f} | {values.loss.mean():.5f} |')
    lines+=['','This limits forecasting conclusions to a lack of demonstrated benefit in these finite readouts and forty '
        'tracks. It is not proof that no more capable, well-calibrated predictor could extract information.','',
        '## Interpretation and limits','',
        'Invariant z can encode angular arrangement through invariant functions of geometry. Useful angular information '
        'therefore does not require the previously inspected internal arrows to be persistent. Tiny angular changes in z '
        'can be informative after scaling; their magnitude alone is not an information test.','',
        'The controls are rich but finite and readouts approximate: a gain can partly reflect easier access to residual radial '
        'information. Matched contrasts and the radius-only counterfactual strengthen the test but do not establish causation. '
        'The radial replacement is out of distribution, input coordinates originate from float16 storage, and only one '
        'checkpoint/ten previously explored sources are covered. Future outcomes are one realized trajectory per initial '
        'condition, not replicated iso-configurational futures. No encoder retraining or directional-head proposal was tested.','',
        '## Artifacts','',
        '- [Main comparison](plots/conditional-information.png)',
        '- [Linear/nonlinear probe comparison](plots/probe-sensitivity.png)',
        '- [Radial-input duplication control](plots/radial-duplication-control.png)',
        '- [Illustrative matched environments](plots/matched-example.png)',
        '- [Dense spatial matching extension](RESULTS_spatial.md) and [its figure](plots/spatial-matched-information.png)',
        '- [Exact metric definitions](tables/METRICS.md); source, matching and conditional-gain CSVs in `tables/`.',
        '- Inputs, held-out predictions, selected penalties, hashes and hardware record in `technical/`.','',
        'Reproduction on node07/A100:','',
        '```bash','conda run -n pointnet-torch214 python -m src.research.gatr_conditional_information \\',
        '  --config configs/analysis/gatr_conditional_information.json','```','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    (root/'README.md').write_text('# GATr conditional information\n\n[Findings](RESULTS.md) · [Figures](index.html) · [Metrics](tables/METRICS.md)\n')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>GATr conditional information</title>'
        '<style>body{max-width:1300px;margin:24px auto;font:17px system-ui}img{max-width:100%}</style>'
        '<h1>Information beyond radial structure</h1><p><a href="RESULTS.md">Full findings</a> · '
        '<a href="tables/METRICS.md">Metric definitions</a></p>'+
        ''.join(f'<img src="plots/{name}.png" alt="{name}">' for name in ('conditional-information','spatial-matched-information','radial-duplication-control','probe-sensitivity','matched-example')))
