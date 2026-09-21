"""Checkpoint-wise conditioning gains and independently fitted structural readouts."""
import json
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.training_methods.bcr.runtime import load_data
from src.training_methods.bcr.model import BCR
from src.training_methods.bcr.data import pack,balanced_subset
from src.training_methods.bcr.evaluate import reconstruction,matching_bins,paired_root_gain
from src.training_methods.bcr.probes import descriptors,frozen_probes
from src.experiment_runner.metric_docs import write_metric_table


def checkpoint(root,arm,step):return root/'technical/runs'/arm/'technical'/('initial.pt' if step==0 else f'step-{step:06d}.pt')


def load_model(path,manifest,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['data_identity']!=manifest['identity']:raise ValueError('Comparator data identities differ')
    model=BCR(saved['config']).to(device).eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
    return model


def compare(config,step,device='cuda'):
    torch.set_num_threads(1);root=resolve_path(config['output']);out=root/'technical/evaluations'/f'{step:06d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    patches,manifest=load_data(resolve_path(config['data']));records=manifest['records']
    descriptor_path=root/'technical/descriptors.npz'
    if descriptor_path.exists():
        with np.load(descriptor_path) as a:targets={k:a[k] for k in ('radial','angular','rich')};cov=a['covariates']
    else:
        targets,cov=descriptors(patches,config['radius_A']);np.savez(descriptor_path,**targets,covariates=cov)
    tr=[i for i,r in enumerate(records) if r['split']=='train'];dev=[i for i,r in enumerate(records) if r['split']=='development']
    chosen=balanced_subset(records,dev,config['evaluation_anchors']);roots=np.array([records[i]['root'] for i in chosen]);liquid=cov[chosen,2]<.35
    bins=matching_bins(cov[tr]);models={arm:load_model(checkpoint(root,arm,step),manifest,device) for arm in ('bcr','unconditional','frozen_random')}
    reports={}
    for arm in ('bcr','frozen_random'):
        path=out/f'{arm}-reconstruction.json'
        if path.exists():reports[arm]=json.loads(path.read_text());continue
        reports[arm]=reconstruction(models[arm],[patches[i] for i in chosen],[records[i] for i in chosen],manifest['noise_levels'],
            draws=config['evaluation_draws'],seed=731,unconditional=models['unconditional'] if arm=='bcr' else None,bins=bins,covariates=cov[chosen],
            shuffles=config['evaluation_shuffles'],chunk=16,include_swaps=arm=='bcr')
        path.write_text(json.dumps(reports[arm],indent=2)+'\n')
    summary=dict(step=step,roots=len(set(roots)),anchors=len(chosen),liquid_anchors=int(liquid.sum()),levels={},matching_bins=bins,
        paired_anchor_keys=[[records[i]['root'],records[i]['frame'],records[i]['center_atom_id']] for i in chosen],uncertainty='paired root bootstrap only, one training seed')
    for level in manifest['noise_levels']:
        c=reports['bcr']['levels'][str(level)];r=reports['frozen_random']['levels'][str(level)]
        true=np.array(c['per_anchor_nmse']);unconditional=np.array(c['per_anchor_unconditional_nmse']);random=np.array(r['per_anchor_nmse']);swap=np.array(c['per_anchor_swap_relax0_nmse'])
        value={}
        for population,mask in [('all',np.ones(len(true),bool)),('liquid_q6_lt035',liquid)]:
            value[population]=dict(unconditional=paired_root_gain(true[mask],unconditional[mask],roots[mask]),frozen_random=paired_root_gain(true[mask],random[mask],roots[mask]),
                matched_swap=paired_root_gain(true[mask],swap[mask],roots[mask]),unrestricted_swap=paired_root_gain(true[mask],np.array(c['per_anchor_swap_relax2_nmse'])[mask],roots[mask]))
        summary['levels'][str(level)]=value
    # Engineering gate frozen before results: 5% mean useful conditioning,
    # positive paired intervals over both trained controls and strict swaps,
    # >=50% strict-match coverage on >=4 development roots, at >=2 levels >=.04.
    passing=[]
    for level,row in summary['levels'].items():
        r=row['liquid_q6_lt035'];u=r['unconditional'];s=r['matched_swap'];q=r['frozen_random']
        if float(level)>=.04 and u['gain'] is not None and u['gain']>=.05 and s['coverage']>=.5 and s['roots']>=4:
            if all(v['ci95'] is not None and v['ci95'][0]>0 for v in (u,s,q)):passing.append(float(level))
    summary['G1']=dict(passing_levels=passing,pass_gate=len(passing)>=2,interpretation='pilot evidence only; no automatic G2/G3 or transfer claim')
    # Probe tuning is confined to TWO training roots. None of the six encoder-
    # development roots chooses ridge penalties or feature/target transforms.
    train_roots=sorted({records[i]['root'] for i in tr});tuning=set(train_roots[-2:]);fit=[i for i in tr if records[i]['root'] not in tuning];tune=[i for i in tr if records[i]['root'] in tuning]
    probe_results={};feature_stats={}
    for arm in ('bcr','frozen_random'):
        target_file=out/f'{arm}-probes.json'
        if target_file.exists():probe_results[arm]=json.loads(target_file.read_text());continue
        with torch.no_grad():z=torch.cat([models[arm].encode(pack(patches[i:i+16],device)).cpu() for i in range(0,len(patches),16)]).numpy()
        np.save(out/f'{arm}-features.npy',z)
        results={};root_records=[records[i] for i in chosen]
        for family,y in targets.items():
            results[family]=frozen_probes(z[fit],z[tune],z[chosen],y[fit],y[tune],y[chosen],root_records)
            q=np.array(chosen)[liquid]
            results[family]['within_liquid']=frozen_probes(z[fit],z[tune],z[q],y[fit],y[tune],y[q],[records[i] for i in q])
        target_file.write_text(json.dumps(results,indent=2)+'\n');probe_results[arm]=results
    summary['probes']=probe_results;summary['probe_tuning_roots']=sorted(tuning)
    (out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    report(config)


def report(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=resolve_path(config['output']);(root/'plots').mkdir(exist_ok=True)
    rows=[json.loads(p.read_text()) for p in sorted((root/'technical/evaluations').glob('*/complete.json'))]
    metrics={f'update_{r["step"]}':r for r in rows};write_metric_table(metrics,root,family='bcr_pilot',name='conditioning')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for row in rows:
        x=[float(k) for k in row['levels']]
        for ax,key in zip(axes,('unconditional','matched_swap')):
            values=[row['levels'][str(v)]['liquid_q6_lt035'][key] for v in x]
            y=[v['gain'] for v in values];ax.plot(x,y,'o-',label=f'{row["step"]:,} updates')
            if all(v['ci95'] is not None for v in values):ax.fill_between(x,[v['ci95'][0] for v in values],[v['ci95'][1] for v in values],alpha=.12)
            ax.set(xlabel='Per-component noise / d0',ylabel='Relative noise-MSE gain',title=key.replace('_',' '));ax.axhline(0,color='k',lw=.5)
    axes[0].legend();fig.tight_layout();fig.savefig(root/'plots/conditioning-gain.png',dpi=160);plt.close(fig)
    lines=['# BCR independent-root G1 pilot','', '12 training roots / 6 development roots; 1325 K independently melted Al. One seed. Intervals resample roots and exclude seed uncertainty.', '',
        '| Updates | Noise/d0 | Gain over unconditional (liquid) | Gain over frozen random | Strict swap coverage |', '|---:|---:|---:|---:|---:|']
    for row in rows:
        for level,result in row['levels'].items():
            d=result['liquid_q6_lt035'];lines.append(f'| {row["step"]} | {level} | {d["unconditional"]["gain"]:.3%} | {d["frozen_random"]["gain"]:.3%} | {d["matched_swap"]["coverage"]:.1%} |')
    lines+=['','Absolute errors, paired intervals, all-population results and structural probes are in tables/conditioning.csv and technical/evaluations. Matching coverage and G1 status are explicit. No crystallization-supervised training or final test access.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
