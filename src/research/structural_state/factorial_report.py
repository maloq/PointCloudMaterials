"""Collect two matched encoder seeds with exact-row onset comparisons."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.structural_state_onset_review import weighted_ap
from .common import sha,write_json,digest
from .report import table


def initial_audit(reference_path, candidate_path, reference_features, candidate_features):
    """Verify actual initialization, separating CUDA reductions from weights."""
    reference=torch.load(reference_path,map_location='cpu',weights_only=False)
    candidate=torch.load(candidate_path,map_location='cpu',weights_only=False)
    if reference['encoder_config']!=candidate['encoder_config'] or reference['auxiliary_targets']!=candidate['auxiliary_targets']:
        raise ValueError('Initial encoder configuration or target construction differs')
    if reference['model'].keys()!=candidate['model'].keys():raise ValueError('Initial state fields differ')
    normalization=('encoder.pooled_mean','encoder.pooled_scale')
    for key,value in reference['model'].items():
        if key in normalization:
            torch.testing.assert_close(value,candidate['model'][key],rtol=1e-6,atol=1e-8)
        elif key.startswith('encoder.'):
            torch.testing.assert_close(value,candidate['model'][key],rtol=0,atol=0)
    if reference_features.shape!=candidate_features.shape:raise ValueError('Initial feature shapes differ')
    delta=candidate_features.astype(float)-reference_features
    scale=float(np.linalg.norm(reference_features))
    if scale==0:raise ValueError('Collapsed initial export')
    relative_rms=float(np.linalg.norm(delta)/scale);maximum=float(np.abs(delta).max())
    if not np.isfinite(delta).all() or relative_rms>1e-5 or maximum>1e-4:
        raise ValueError(f'Initial exports exceed audited CUDA noise: relative RMS={relative_rms}, max={maximum}')
    heads={}
    for name,calibration in reference['calibration'].items():
        if calibration['ridge']!=candidate['calibration'][name]['ridge']:
            raise ValueError(f'Initial head calibration penalty differs: {name}')
        predictions=[]
        for saved,features in ((reference,reference_features),(candidate,candidate_features)):
            weight=saved['model'][f'heads.{name}.weight'].numpy().astype(float)
            bias=saved['model'][f'heads.{name}.bias'].numpy().astype(float)
            predictions.append(features.astype(float)@weight.T+bias)
        difference=predictions[1]-predictions[0]
        rms=float(np.sqrt(np.mean(difference**2)));peak=float(np.abs(difference).max())
        if not np.isfinite(difference).all() or rms>1e-4 or peak>1e-3:
            raise ValueError(f'Initial calibrated head predictions differ: {name}, RMS={rms}, max={peak}')
        heads[name]=dict(rms=rms,max_abs=peak)
    return dict(encoder_state_exact_except_normalization=True,export_relative_rms=relative_rms,
                export_max_abs=maximum,head_prediction_differences=heads)


def onset_scores(path, indices, source, event, weights):
    with np.load(path) as p:
        for key,expected in [('indices',indices),('source',source),('event',event)]:
            np.testing.assert_array_equal(p[key],expected)
        risk=p['risks'][:,-1].astype(float)
        probability=risk.clip(1e-7,1-1e-7)
        logits=p['logits'].astype(float)
    actual=event<5
    bins=np.arange(5)[None]
    nll=(np.logaddexp(0,logits)*(bins<event[:,None])+np.logaddexp(0,-logits)*(bins==event[:,None])).sum(1)
    values=dict(average_precision=weighted_ap(actual,risk,weights),
        brier=weights@((risk-actual)**2),
        log_loss=weights@-(actual*np.log(probability)+(~actual)*np.log1p(-probability)),nll=weights@nll)
    np.testing.assert_allclose(values['average_precision'][0],average_precision_score(actual,risk,sample_weight=weights[0]),rtol=1e-12)
    published=json.loads(path.with_name('metrics.json').read_text())
    for name,value in values.items():
        expected=published['nll'] if name=='nll' else published['horizons']['12.0'][name]
        np.testing.assert_allclose(value[0],expected,rtol=1e-6,atol=1e-8)
    return values


def mechanism_effects(seed, contrasts, physical, neighbors, heads, onset):
    """Predeclared noncrystalline retention/transfer rules, without model selection."""
    def get(rows, **keys):
        selected=[r for r in rows if all(r[k]==v for k,v in keys.items())]
        if len(selected)!=1:raise ValueError(f'Expected one metric row for {keys}, got {len(selected)}')
        return float(selected[0]['mse'])
    rows=[]
    for label,reference,candidate in contrasts:
        retention=[]
        for family in ('relaxed_radial','relaxed_angular','relaxed_l6','current_order'):
            kw=dict(representation='exported',target=family,readout='ridge',population='PTM_other')
            retention.append(100*(get(physical,arm=candidate,**kw)/get(physical,arm=reference,**kw)-1))
        for block in ('radial','l2','l4'):
            kw=dict(checkpoint='last',block=block,population='PTM_other')
            retention.append(100*(get(heads,arm=candidate,**kw)/get(heads,arm=reference,**kw)-1))
        discrepancy=[]
        for family in ('relaxed_angular','relaxed_l6'):
            kw=dict(representation='exported',target=family,population='noncrystalline')
            discrepancy.append(100*(get(neighbors,arm=candidate,**kw)/get(neighbors,arm=reference,**kw)-1))
        kw=dict(representation='exported',target='future_order_12',readout='ridge',population='PTM_other')
        future=100*(get(physical,arm=candidate,**kw)/get(physical,arm=reference,**kw)-1)
        ap=onset[candidate,'exported','mlp']['average_precision'][0]-onset[reference,'exported','mlp']['average_precision'][0]
        brier=onset[candidate,'exported','mlp']['brier'][0]-onset[reference,'exported','mlp']['brier'][0]
        worst=max(retention);distance=float(np.mean(discrepancy))
        rows.append(dict(seed=seed,contrast=label,reference=reference,candidate=candidate,
            worst_current_error_change_percent=worst,withheld_neighbor_error_change_percent=distance,
            future12_error_change_percent=future,onset_ap_delta=float(ap),onset_brier_delta=float(brier),
            retention_pass=worst<=2.,neighbor_rule_pass=worst<=2. and distance<=-1.,
            future_rule_pass=worst<=2. and future<=-1.))
    return rows


def run(config_path):
    config_path=resolve_path(config_path)
    config=json.loads(config_path.read_text());root=resolve_path(config['output'])
    (root/'technical').mkdir(parents=True,exist_ok=True)
    with (root/'technical/collect.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return collect(config,root)


def collect(config,root):
    complete=[];pending=[];summary=[];differences=[];physical=[];ranks=[];inputs={}
    neighbors=[];heads=[];comparisons=[];mechanisms=[]
    seeds_arrays={};reference_population=None;initial_differences={};initial_checks={}
    for item in config['seeds']:
        directory=resolve_path(item['output']);tech=directory/'technical'
        status=tech/'collection.json'
        if not status.exists() or json.loads(status.read_text())['state']!='complete':
            pending.append(item['seed']);continue
        identity=json.loads((tech/'identity.json').read_text());study=identity['config']
        if study['seed']!=item['seed'] or json.loads(status.read_text())['identity']!=digest(identity):
            raise ValueError('Factorial seed/collection identity mismatch')
        cache=resolve_path(study['cache']);manifest=json.loads((cache/'manifest.json').read_text())
        if sha(cache/'manifest.json')!=identity['data_sha256'] or sha(cache/'records.json')!=manifest['files']['records.json']:
            raise ValueError('Factorial population provenance changed')
        records=json.loads((cache/'records.json').read_text())
        with np.load(tech/'evaluation/B-relaxed/exported/hazard_mlp/predictions.npz') as p:
            indices,source,event=[p[k] for k in ('indices','source','event')]
        np.testing.assert_array_equal(source,[records[i]['source'] for i in indices])
        if any(records[i]['split']!='development' for i in indices):raise ValueError('Onset population is not held out')
        population=np.c_[indices,source,event]
        if reference_population is not None:np.testing.assert_array_equal(reference_population,population)
        reference_population=population
        ids,inverse,counts=np.unique(source,return_inverse=True,return_counts=True)
        temperatures=np.array([records[indices[np.flatnonzero(source==s)[0]]]['temperature_K'] for s in ids])
        rng=np.random.default_rng(config['bootstrap_seed'])
        draws=np.concatenate([rng.choice(np.flatnonzero(temperatures==t),size=(config['bootstrap'],int((temperatures==t).sum())))
                              for t in np.unique(temperatures)],axis=1)
        multiplicity=np.array([np.bincount(row,minlength=len(ids)) for row in draws])
        weights=np.vstack([np.ones(len(ids)),multiplicity])[:,inverse]/counts[inverse][None]/len(ids)
        np.testing.assert_allclose(weights.sum(1),1.)
        scores={};initial=None
        for arm in study['arms']:
            name=arm['name'];fit=tech/'fits'/name
            receipt=json.loads((fit/'complete.json').read_text())
            if receipt['identity']!=digest(identity) or receipt['step']!=4096:
                raise ValueError('Factorial requires matched final4096 checkpoints')
            for filename,key in [('last.pt','checkpoint_sha256'),('features.npz','feature_sha256')]:
                path=fit/filename
                if sha(path)!=receipt[key]:raise ValueError(f'Changed completed fit: {path}')
                inputs[str(path)]=receipt[key]
            with np.load(fit/'features.npz') as features:
                if initial is None:initial=features['initial_exported'].copy()
                initial_checks[f'{item["seed"]}/{name}']=initial_audit(
                    tech/'fits/B-relaxed/initial.pt',fit/'initial.pt',initial,features['initial_exported'])
                initial_differences[f'{item["seed"]}/{name}']=float(np.max(np.abs(initial-features['initial_exported'])))
                inputs[str(fit/'initial.pt')]=sha(fit/'initial.pt')
            representations=['exported','initial_exported']
            if name=='B-relaxed':representations+=['descriptor','descriptor_pca64','conditions']
            for representation in representations:
                for head in ('linear','mlp'):
                    path=tech/'evaluation'/name/representation/('hazard_'+head)/'predictions.npz'
                    scores[name,representation,head]=onset_scores(path,indices,source,event,weights)
                    inputs[str(path)]=sha(path)
                    for metric,values in scores[name,representation,head].items():
                        valid=values[1:][np.isfinite(values[1:])]
                        lo,hi=np.quantile(valid,[.025,.975])
                        summary.append(dict(seed=item['seed'],arm=name,representation=representation,head=head,metric=metric,
                                            value=float(values[0]),ci_low=float(lo),ci_high=float(hi),valid_draws=len(valid)))
        for label,reference,candidate in study['contrasts']:
            for head in ('linear','mlp'):
                for metric in ('average_precision','brier','log_loss','nll'):
                    delta=scores[candidate,'exported',head][metric]-scores[reference,'exported',head][metric]
                    seeds_arrays.setdefault((label,head,metric),[]).append(delta)
                    valid=delta[1:][np.isfinite(delta[1:])];lo,hi=np.quantile(valid,[.025,.975])
                    differences.append(dict(seed=item['seed'],contrast=label,head=head,metric=metric,
                        reference=reference,candidate=candidate,delta=float(delta[0]),ci_low=float(lo),ci_high=float(hi),valid_draws=len(valid)))
        seed_tables={}
        for filename,destination in [('physical.csv',physical),('embedding_geometry.csv',ranks),
                                     ('neighbors.csv',neighbors),('training_heads.csv',heads),('comparisons.csv',comparisons)]:
            path=directory/'tables'/filename
            inputs[str(path)]=sha(path)
            with path.open() as stream:seed_tables[filename]=list(csv.DictReader(stream))
            destination.extend(dict(seed=item['seed'],**row) for row in seed_tables[filename])
        mechanisms.extend(mechanism_effects(item['seed'],study['contrasts'],seed_tables['physical.csv'],
                                           seed_tables['neighbors.csv'],seed_tables['training_heads.csv'],scores))
        complete.append(item['seed'])
    snapshot_metric_docs(root,'structural_state_future')
    for name,rows in [('onset',summary),('paired_onset',differences),('physical',physical),('embedding_geometry',ranks),
                     ('neighbors',neighbors),('training_heads',heads),('comparisons',comparisons),('mechanism_effects',mechanisms)]:
        if rows:table(root/'tables'/f'{name}.csv',rows)
    averages=[]
    for (contrast,head,metric),values in seeds_arrays.items():
        average=np.mean(values,axis=0);valid=average[1:][np.isfinite(average[1:])];lo,hi=np.quantile(valid,[.025,.975])
        averages.append(dict(contrast=contrast,head=head,metric=metric,seeds=len(values),delta=float(average[0]),
                             ci_low=float(lo),ci_high=float(hi),note='Shared paired source bootstrap; fixed trained seeds, not seed-population uncertainty'))
    if averages:table(root/'tables/mean_seed_effects.csv',averages)
    write_json(root/'technical/status.json',dict(state='complete' if not pending else 'partial',completed_seeds=complete,pending_seeds=pending,
        inputs=inputs,initial_export_max_abs_delta=initial_differences,initialization_audit=initial_checks))
    lines=['# Geometry, distance and future encoder training','',f'Completed seeds: {complete}; pending: {pending}.',
        '', 'Four arms per seed, matched 4096 updates, relaxed 8 Å inputs and common current-structure labels. Final weights are primary; no onset labels train the encoder.',
        '', 'The 9 ps residual baseline and target scalers use fitting sources only. Angular/l6 observations and 3 ps/12 ps future horizons remain outside encoder targets; onset evaluation uses 643 reused development windows with 18 positive 12 ps windows.',
        '', 'Two initialization/sampling seeds, fixed probe seed. Report per-seed effects; source intervals do not quantify uncertainty over all possible training seeds.',
        '', '[Onset scores](tables/onset.csv) · [Paired onset differences](tables/paired_onset.csv) · [Mean seed effects](tables/mean_seed_effects.csv) · [Physical probes](tables/physical.csv) · [Embedding geometry](tables/embedding_geometry.csv)',
        '', '[Mechanism rules and matched forecasting changes](tables/mechanism_effects.csv) · [Liquid neighbors](tables/neighbors.csv) · [Physical source intervals](tables/comparisons.csv)',
        '', 'Physical retention tolerance is 2% relative MSE versus the matched reference. A future or neighborhood mechanism rule also requires at least 1% lower error in its declared withheld metric in both seeds; inspect per-seed rule fields. These are predeclared practical screening rules, not significance tests or model selection. These are development mechanism experiments, not an independent final-test claim.']
    if summary:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        names=['B-relaxed','D-physical-distance','E-future-residual','F-distance-future']
        for ax,metric in zip(axes,('average_precision','brier')):
            for seed in complete:
                values=[next(r['value'] for r in summary if r['seed']==seed and r['arm']==name and r['representation']=='exported' and r['head']=='mlp' and r['metric']==metric) for name in names]
                ax.plot(range(4),values,'o-',label=str(seed))
            ax.set_xticks(range(4),['Geometry','+Distance','+Future','+Both']);ax.set_ylabel('12ps '+metric);ax.legend(title='Encoder seed');ax.grid(alpha=.2)
        fig.suptitle('Matched encoder-training interventions; reused development cohort')
        fig.savefig(root/'plots/onset-factorial.png',dpi=180);plt.close(fig)
        lines+=['','![Encoder-training comparison](plots/onset-factorial.png)']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return not pending


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
