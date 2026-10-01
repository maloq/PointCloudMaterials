"""Common moment selectors, full-target frozen readouts and paired source errors."""
import csv
import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.project_runtime.paths import resolve_path
from src.research.predictive_baseline.data import moments
from src.research.predictive_baseline.evaluate import ridge, source_average, groups
from .common import FAMILY, load, read, features, jobs, name, folder, output, cache


def moment_selected_ridge(c, data, x):
    chosen, choices = ridge(x, data['target'][:, :18], data, c['ridge_alphas'])
    # Ridge outputs separate by target coordinate; refit all coordinates with
    # the single alpha selected on the common 18 moments only.
    full, _ = ridge(x, data['target'], data, [chosen['alpha']])
    return full, choices


def readouts(c, arm, data, z, root):
    root.mkdir(parents=True, exist_ok=True)
    (root/'technical').mkdir(exist_ok=True)
    full, choices = moment_selected_ridge(c, data, z)
    pred = full.pop('prediction')
    np.savez_compressed(root/'technical/readout.npz', prediction=pred, **full,
                        parent=data['parent'], atom_ids=data['atom_ids'])
    write_metric_rows(choices, root, family=FAMILY, name='readout-selection')
    tr = data['roles'] == 'train'
    mean, scale = moments(data['extra_future'][tr], data['weights'][tr])
    extra = (data['extra_future']-mean)/scale
    probe, choices = ridge(z, extra, data, c['ridge_alphas'])
    prediction = probe.pop('prediction')
    np.savez(root/'technical/extra-probe.npz', **probe, target_mean=mean, target_scale=scale)
    rows = []
    for population, mask in groups(data):
        sources, error = source_average(((prediction-extra)**2).mean(1), data, mask)
        for source, value in zip(sources, error, strict=True):
            rows.append(dict(source=source, population=population, extra_observable_mse=float(value)))
    write_metric_rows(rows, root, family=FAMILY, name='extra-observables')
    write_json(root/'technical/readout-complete.json', dict(state='complete', arm=arm,
        readout_sha256=sha(root/'technical/readout.npz'), extra_selected_alpha=float(probe['alpha'])))


def controls(c):
    data, manifest = load(c)
    for source in ['descriptors', *[s for s in c['sources'] if s != 'joint']]:
        x = data['descriptors'] if source == 'descriptors' else features(c, data, manifest, source)
        arm = dict(source=source, target='full', variance='free', seed=-1)
        root = output(c)/'analyses'/f'{source}-ridge'
        readouts(c, arm, data, x, root)
    write_json(output(c)/'technical/controls-complete.json', dict(state='complete', target_identity=manifest['identity']))


def prediction_stream(c, data):
    tr = data['roles'] == 'train'
    prior = np.average(data['target'][tr], axis=0, weights=data['weights'][tr])
    yield dict(source='prior', target='full', variance='free', seed=-1), 'prior', np.broadcast_to(prior, data['target'].shape)
    for kind in ('joint', 'probe'):
        for arm in jobs(c, kind):
            tech = folder(c, arm)/'technical'
            done = read(tech/'complete.json')
            if done['arm'] != arm or done['target_identity'] != c['baseline_target_identity']:
                raise ValueError(f'Fit identity mismatch: {arm}')
            if sha(tech/'predictions.npz') != done['predictions_sha256']:
                raise ValueError(f'Fit predictions changed: {arm}')
            with np.load(tech/'predictions.npz') as a:
                if not np.array_equal(a['parent'], data['parent']) or not np.array_equal(a['atom_ids'], data['atom_ids']):
                    raise ValueError(f'Fit row identities changed: {arm}')
                yield arm, 'nonlinear', a['prediction']
            if kind == 'joint':
                receipt = read(tech/'readout-complete.json')
                if sha(tech/'readout.npz') != receipt['readout_sha256']:
                    raise ValueError('Frozen joint readout changed')
                with np.load(tech/'readout.npz') as a:
                    yield arm, 'linear_z', a['prediction']
    for source in ['descriptors', *[s for s in c['sources'] if s != 'joint']]:
        root = output(c)/'analyses'/f'{source}-ridge/technical'
        receipt = read(root/'readout-complete.json')
        if sha(root/'readout.npz') != receipt['readout_sha256']:
            raise ValueError(f'Changed frozen reference: {source}')
        with np.load(root/'readout.npz') as a:
            yield dict(source=source, target='full', variance='free', seed=-1), 'linear', a['prediction']
    # Retain the original selector and identify those predictions explicitly.
    for seed in c['fit_seeds']:
        root = resolve_path(c['baseline_output'])/'analyses'/f'joint-seed-{seed}'/'technical'
        receipt = read(root/'complete.json')
        if sha(root/'predictions.npz') != receipt['predictions_sha256']:
            raise ValueError('Original baseline predictions changed')
        with np.load(root/'predictions.npz') as a:
            yield dict(source='joint_v1', target='full', variance='free', seed=seed), 'historical_full_selector', a['prediction']


def summarize(c, rows):
    result = []
    keys = sorted({(r['model'], r['readout'], r['population'], r['scope']) for r in rows})
    indexed = {}
    for row in rows:
        indexed.setdefault((row['model'], row['readout'], row['population'], row['scope']), []).append(row)
    for model, readout, population, scope in keys:
        selected = indexed[model, readout, population, scope]
        prior = indexed['prior-full-free', 'prior', population, scope]
        sources = sorted({r['source_id'] for r in selected})
        values = np.array([np.mean([r['corrected_error'] for r in selected if r['source_id'] == s]) for s in sources])
        prior_values = np.array([next(r['corrected_error'] for r in prior if r['source_id'] == s) for s in sources])
        delta = values-prior_values
        rng = np.random.default_rng(c['target']['seed'])
        draw = rng.integers(len(sources), size=(c['bootstrap_draws'], len(sources)))
        low, high = np.quantile(delta[draw].mean(1), [.025, .975])
        seed_values = [np.mean([r['corrected_error'] for r in selected if r['seed'] == s]) for s in sorted({r['seed'] for r in selected})]
        result.append(dict(model=model, readout=readout, population=population, scope=scope,
            sources=len(sources), seeds=len(seed_values), error=float(np.mean([r['error'] for r in selected])),
            corrected_error=float(values.mean()), delta_vs_prior=float(delta.mean()),
            delta_ci_low=float(low), delta_ci_high=float(high), seed_sd=float(np.std(seed_values)),
            negative_variance_fraction=float(np.mean([r['negative_variance_fraction'] for r in selected]))))
    return result


def contrasts(c, rows):
    # Predeclared paired comparisons; no post-hoc best arm selection.
    pairs = []
    for source in c['sources']:
        for variance in c['variance_arms']:
            pairs.append((f'{source}-full-{variance}', 'nonlinear', f'{source}-moments-{variance}', 'nonlinear', 'moments'))
        for target in c['target_arms']:
            pairs.append((f'{source}-{target}-nonnegative', 'nonlinear', f'{source}-{target}-free', 'nonlinear', 'moments'))
        if source != 'joint':
            pairs.append((f'{source}-full-free', 'nonlinear', f'{source}-full-free', 'linear', 'full'))
            pairs.append(('joint-full-free', 'nonlinear', f'{source}-full-free', 'nonlinear', 'full'))
    result = []
    for left, lh, right, rh, scope in pairs:
        for pop in ('all', 'clear_liquid'):
            a = [r for r in rows if (r['model'],r['readout'],r['scope'],r['population']) == (left,lh,scope,pop)]
            b = [r for r in rows if (r['model'],r['readout'],r['scope'],r['population']) == (right,rh,scope,pop)]
            sources = sorted({r['source_id'] for r in a})
            delta = np.array([np.mean([r['corrected_error'] for r in a if r['source_id']==s])-
                              np.mean([r['corrected_error'] for r in b if r['source_id']==s]) for s in sources])
            draw = np.random.default_rng(c['target']['seed']).integers(len(sources), size=(c['bootstrap_draws'],len(sources)))
            lo, hi = np.quantile(delta[draw].mean(1), [.025,.975])
            result.append(dict(left=left,left_head=lh,right=right,right_head=rh,scope=scope,population=pop,
                sources=len(sources),delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi)))
    return result


def collect(c):
    data, manifest = load(c)
    root = output(c)/'analyses/comparison-v1'
    root.mkdir(parents=True, exist_ok=True)
    fmap = np.load(cache(c)/'feature_map.npz')
    feat = np.asarray(data['features'], dtype=np.float64)
    target = feat.mean(1)
    noise = feat.var(1, ddof=1)/12
    y = np.asarray(data['y'], dtype=np.float64)
    ym, yv = y.mean(1), y.var(1, ddof=1)
    rows, physical_rows = [], []
    for arm, head, pred in prediction_stream(c, data):
        d = pred.shape[1]
        if d not in (18,274) or not np.isfinite(pred).all():
            raise ValueError(f'Invalid follow-up prediction shape/values: {arm}, {pred.shape}')
        error = (pred-target[:, :d])**2
        corrected = error-noise[:, :d]
        raw = pred[:, :18]/np.sqrt(fmap['metric'][:18])+fmap['center'][:18]
        mean = raw[:, :9]*fmap['y_scale']+fmap['y_mean']
        variance = (raw[:, 9:18]-raw[:, :9]**2)*fmap['y_scale']**2
        scopes = [('moments',0,18),('mean',0,9),('second',9,18)]
        if d == 274:
            scopes += [('rff',18,274),('full',0,274)]
        for population, mask in groups(data):
            _, negative = source_average((variance<0).mean(1), data, mask)
            for scope, lo, hi in scopes:
                sources, scores = source_average(np.column_stack((error[:,lo:hi].sum(1),corrected[:,lo:hi].sum(1))),data,mask)
                for source, score, neg in zip(sources,scores,negative,strict=True):
                    rows.append(dict(model=name(arm),readout=head,seed=arm['seed'],population=population,
                        scope=scope,source_id=source,error=float(score[0]),corrected_error=float(score[1]),
                        negative_variance_fraction=float(neg)))
            _, physical = source_average(np.concatenate(((mean-ym)**2,(variance-yv)**2),1),data,mask)
            for index, observable in enumerate(manifest['target_columns']):
                physical_rows.append(dict(model=name(arm),readout=head,seed=arm['seed'],population=population,
                    observable=observable,mean_mse=float(physical[:,index].mean()),variance_mse=float(physical[:,9+index].mean())))
    summary = summarize(c, rows)
    paired = contrasts(c, rows)
    write_metric_rows(rows,root,family=FAMILY,name='source-scores')
    write_metric_rows(summary,root,family=FAMILY,name='comparison')
    write_metric_rows(paired,root,family=FAMILY,name='paired-contrasts')
    write_metric_rows(physical_rows,root,family=FAMILY,name='physical-moments')
    extra_rows = []
    for path in sorted((output(c)/'analyses').glob('*/tables/extra-observables.csv')):
        with path.open() as stream:
            extra_rows.extend(dict(representation=path.parent.parent.name,**row) for row in csv.DictReader(stream))
    write_metric_rows(extra_rows,root,family=FAMILY,name='extra-observables')
    plot(c, summary, paired)
    for arm in jobs(c,'joint'):
        selected = [r for r in rows if r['model']==name(arm) and r['readout']=='nonlinear' and
                    r['seed']==arm['seed'] and r['population']=='all' and r['scope']=='moments']
        update_recorded_summary(folder(c,arm)/'technical/wandb.json',
            {'followup/test_corrected_moment_error':float(np.mean([r['corrected_error'] for r in selected])),
             'followup/test_negative_variance_fraction':float(np.mean([r['negative_variance_fraction'] for r in selected]))},
            evaluation='predictive-followup-v1',expected=dict(mode='online'))
    write_json(root/'technical/complete.json',dict(state='complete',target_identity=manifest['identity'],
        joint_fits=len(jobs(c,'joint')),local_frozen_heads=len(jobs(c,'probe')),test_sources=6))


def plot(c, summary, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = output(c)/'analyses/comparison-v1/plots'
    root.mkdir(parents=True, exist_ok=True)
    labels = dict(joint='New MACE128',mace_vicreg='VICReg128',mace_epi='Epi128',mm_tda_block_direct_full='MM-TDA256')
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,pop in zip(axes,('all','clear_liquid'),strict=True):
        chosen = [r for r in summary if r['scope']=='full' and r['population']==pop and
                  r['model'] in [f'{s}-full-free' for s in c['sources']] and r['readout'] in ('linear','nonlinear')]
        for i,row in enumerate(chosen):
            source = row['model'].removesuffix('-full-free')
            ax.hlines(i,row['delta_ci_low'],row['delta_ci_high'],color='#547a9b')
            ax.scatter(row['delta_vs_prior'],i,color='#143c5b')
        ax.set_yticks(range(len(chosen)),[labels[r['model'].removesuffix('-full-free')]+' / '+r['readout'] for r in chosen])
        ax.axvline(0,color='gray',lw=1);ax.set_title(pop.replace('_',' '));ax.set_xlabel('Full-feature error minus prior; lower is better')
    fig.suptitle('Matched head comparison; common moment selector\nSix historical test sources; source-paired 95% intervals')
    fig.savefig(root/'matched-heads.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,pop in zip(axes,('all','clear_liquid'),strict=True):
        chosen = [r for r in pairs if r['population']==pop and r['left'].startswith('joint-') and
                  r['right'].startswith('joint-') and r['scope']=='moments']
        for i,row in enumerate(chosen):
            ax.hlines(i,row['ci_low'],row['ci_high'],color='#527c66');ax.scatter(row['delta'],i,color='#164a2e')
        ax.set_yticks(range(len(chosen)),[r['left'].removeprefix('joint-')+' minus '+r['right'].removeprefix('joint-') for r in chosen])
        ax.axvline(0,color='gray',lw=1);ax.set_title(pop.replace('_',' '));ax.set_xlabel('Common moment error difference; negative favors left')
    fig.suptitle('Fourier supervision and nonnegative variance: paired ablations')
    fig.savefig(root/'target-variance-ablations.png',dpi=160);plt.close(fig)
