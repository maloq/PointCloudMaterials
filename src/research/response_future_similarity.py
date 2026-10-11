"""Held-out future-distribution geometry and retrieval from frozen encoders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import rankdata

FAMILY = 'response_future_similarity'
METRICS = ['spearman', 'recall1', 'recall5', 'gain1', 'gain5']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def paths(c):
    from src.project_runtime.paths import resolve_path
    return resolve_path(c['campaign']), resolve_path(c['output'])


def bank(c):
    import torch
    campaign, _ = paths(c)
    receipt = read(campaign/'technical/data-complete.json')
    if receipt['identity'] != c['campaign_identity'] or sha(receipt['cache']) != receipt['sha256']:
        raise ValueError('The sealed response bank changed')
    data = torch.load(receipt['cache'], map_location='cpu', weights_only=False)
    rows = [r for r in data['records'] if r['role'] == 'test']
    source = np.array([r['source'] for r in rows])
    if len(rows) != 120 or len(np.unique(source)) != 30:
        raise ValueError('Expected 30 held-out ancestors with four velocity fields each')
    return data, rows, source


def prepare(c):
    from src.experiment_runner.metric_docs import check_metric_docs
    from src.project_runtime.paths import REPO
    check_metric_docs(family=FAMILY)
    campaign, out = paths(c)
    binding = read(campaign/'technical/binding.json')
    if binding['identity'] != c['campaign_identity']:
        raise ValueError('Changed scientific campaign')
    if binding['config']['fixed_identity'] != c['fixed_identity']:
        raise ValueError('Changed fixed Al64 release')
    if read(campaign/'technical/complete.json')['fits'] != 12:
        raise ValueError('All twelve frozen fits must finish before this comparison')
    data, rows, source = bank(c)
    models = []
    for seed in binding['config']['fit_seeds']:
        for arm in binding['config']['arms']:
            name = arm['name']+'-'+str(seed)
            folder = campaign/'analyses'/name/'technical'
            receipt = read(folder/'complete.json')
            checkpoint = folder/'best.pt'
            if sha(checkpoint) != receipt['checkpoint_sha256']:
                raise ValueError('Changed selected checkpoint: '+name)
            models.append(dict(name=name, arm=arm['name'], seed=seed,
                velocities=arm['velocities'], checkpoint=str(checkpoint),
                sha256=receipt['checkpoint_sha256']))
    record = dict(config=c, models=models, campaign_binding=sha(campaign/'technical/binding.json'),
        data_receipt=read(campaign/'technical/data-complete.json'),
        producer_sha256=sha(__file__), metric_description_sha256=sha(REPO/'docs/metrics'/f'{FAMILY}.md'),
        inputs=read(campaign/'technical/prediction-context.json'),
        evaluation=dict(track='all64 fixed release, original 30 test response ancestors',
            static_clustering_rows=False, new_simulations=0, new_training=0,
            states=120, independent_ancestors=30, candidate_same_source_excluded=True))
    path = out/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Analysis changed; use a new output revision')
    write(path, record)
    shutil.copy2(__file__, out/'technical/producer.py')
    write(out/'technical/config.json', c)
    print('Prepared twelve frozen models and thirty held-out ancestors', flush=True)


def extract(c, lane):
    # Run this stage from the campaign's frozen source tree, via PYTHONPATH.
    import gc
    import torch
    from src.research.response_velocity.model import Predictor
    from src.research.response_geometry.features import observed
    from src.project_runtime.paths import resolve_path
    campaign, out = paths(c)
    import src.research.response_velocity.model as producer
    if Path(producer.__file__).resolve() != (campaign/'technical/code/src/research/response_velocity/model.py').resolve():
        raise ValueError('Extraction must import the frozen training implementation')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = read(out/'technical/binding.json')
    if sha(__file__) != contract['producer_sha256']:
        raise ValueError('Analysis producer changed')
    cfg = read(campaign/'technical/code/config.json')
    data, rows, source = bank(c)
    x = torch.stack([torch.cat((r['position'], r['velocity']), -1) for r in rows]).float().cuda()
    unique, first, inverse = np.unique(source, return_index=True, return_inverse=True)
    (out/'data').mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if lane == 0:
            norm = torch.load(resolve_path(cfg['geometry_reference'])/'technical/feature-normalization.pt', weights_only=False)
            descriptor = observed(x[first, :, :3].reshape(len(first), -1))
            descriptor = (descriptor-norm['mean'].float().cuda())/norm['scale'].float().cuda()
            np.savez_compressed(out/'data/initial-descriptor.npz', descriptor=descriptor.cpu().numpy()[inverse], source=source)
        for index, item in enumerate(contract['models']):
            if index % c['lanes'] != lane:
                continue
            name = item['name']; destination = out/'data'/(name+'.npz')
            if sha(item['checkpoint']) != item['sha256']:
                raise ValueError('Checkpoint changed: '+name)
            model = Predictor(cfg, item['velocities']).cuda()
            model.load_state_dict(torch.load(item['checkpoint'], map_location='cuda', weights_only=False)['model'], strict=True)
            model.eval().requires_grad_(False)
            # Q is exactly the same function for every velocity field at fixed Q.
            take = np.arange(len(rows)) if item['velocities'] else first
            z = torch.cat([model.encode(b) for b in x[take].split(c['batch_size'])])
            prediction = model.head(z)
            z = z.cpu().numpy(); prediction = prediction.cpu().numpy()
            if not item['velocities']:
                z = z[inverse]; prediction = prediction[inverse]
            if not np.isfinite(z).all() or not np.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite frozen representations: '+name)
            previous = np.load(campaign/'analyses'/name/'technical/predictions.npz')
            error = float(np.max(np.abs(previous['prediction']-prediction)))
            if error > 2e-4 or not np.array_equal(previous['source'], source):
                raise ValueError(f'Frozen prediction reproduction differs: {name}, {error}')
            np.savez_compressed(destination, embedding=z, prediction=prediction, source=source)
            write(out/'technical'/('extracted-'+name+'.json'), dict(checkpoint_sha256=item['sha256'],
                artifact_sha256=sha(destination), prediction_reproduction_max_abs=error))
            print('extracted', name, 'prediction difference', error, flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
    write(out/'technical'/f'extraction-lane{lane}.json', dict(state='complete', models=6))


def squared_distance(x):
    return np.mean((x[:, None]-x[None, :])**2, axis=-1)


def outcome_distance(shots, mixture):
    mean = shots.mean(1)
    variance = shots.var(1, ddof=1)/shots.shape[1]
    if mixture is not None:
        mean = np.stack([mean[g].mean(0) for g in mixture])
        # Four fixed velocity fields, each with independent thermostat shots.
        variance = np.stack([variance[g].sum(0)/len(g)**2 for g in mixture])
    noise = variance.mean(-1)
    return squared_distance(mean)-noise[:, None]-noise[None, :]


def neighbors(distance, eligible, k):
    d = np.where(eligible, distance, np.inf)
    threshold = np.partition(d, k-1, axis=1)[:, k-1, None]
    below = eligible & (d < threshold)
    tied = eligible & (d == threshold)
    fraction = (k-below.sum(1))/tied.sum(1)
    weights = below.astype(float)+tied*fraction[:, None]
    return weights


def correlation(a, b, eligible):
    x = np.stack([rankdata(row[m]) for row, m in zip(a, eligible)])
    y = np.stack([rankdata(row[m]) for row, m in zip(b, eligible)])
    x -= x.mean(1, keepdims=True); y -= y.mean(1, keepdims=True)
    norm = np.sqrt((x*x).sum(1)*(y*y).sum(1))
    if np.any(norm == 0):
        raise ValueError('Constant candidate-distance ranking; inspect collapsed representation')
    return float(np.mean((x*y).sum(1)/norm))


def score(distance, truth, halves, source, reference=False):
    eligible = source[:, None] != source[None, :]
    a, b = halves
    baseline = float(np.mean(truth[eligible]))
    if baseline <= 0:
        raise ValueError('Mean future separation is below the estimable noise floor')
    result = dict(spearman=correlation(a, b, eligible) if reference else correlation(distance, truth, eligible))
    for k in [1, 5]:
        wa, wb = neighbors(a, eligible, k), neighbors(b, eligible, k)
        if reference:
            loss = float(np.mean(((wa*b).sum(1)+(wb*a).sum(1))/(2*k)))
            random = float(np.mean(((a+b)/2)[eligible]))
            recall = float(np.mean((wa*wb).sum(1)/k))
        else:
            w = neighbors(distance, eligible, k)
            loss = float(np.mean((w*truth).sum(1)/k)); random = baseline
            recall = float(np.mean((w*(wa+wb)/2).sum(1)/k))
        if random <= 0:
            raise ValueError('Nonpositive random-neighbor loss')
        result.update({f'recall{k}': recall, f'gain{k}': 1-loss/random,
            f'future_distance{k}': loss, f'random_distance{k}': random})
    return result


def jackknife(distance, truth, halves, source, reference):
    values = []
    for excluded in np.unique(source):
        keep = np.flatnonzero(source != excluded); take = np.ix_(keep, keep)
        row = score(distance[take], truth[take], [a[take] for a in halves], source[keep], reference)
        values.append([row[m] for m in METRICS])
    return np.array(values)


def interval(point, replicates):
    n = len(replicates)
    se = float(np.sqrt((n-1)/n*np.sum((replicates-replicates.mean())**2)))
    return point-1.96*se, point+1.96*se


def analyze(c):
    from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
    check_metric_docs(family=FAMILY)
    campaign, out = paths(c)
    contract = read(out/'technical/binding.json')
    if sha(__file__) != contract['producer_sha256']:
        raise ValueError('Changed analysis implementation')
    data, records, source = bank(c)
    unique = np.unique(source); groups = [np.flatnonzero(source == s) for s in unique]
    if any(sorted(records[i]['field'] for i in g) != [0, 1, 2, 3] for g in groups):
        raise ValueError('Incomplete fixed velocity mixture')
    values = np.stack([r['values'].double().numpy() for r in records])
    scale = data['scale'].double().numpy()
    descriptor = np.load(out/'data/initial-descriptor.npz')['descriptor'].astype(float)
    artifacts = {}
    for item in contract['models']:
        p = out/'data'/(item['name']+'.npz')
        receipt = read(out/'technical'/('extracted-'+item['name']+'.json'))
        if sha(p) != receipt['artifact_sha256'] or receipt['checkpoint_sha256'] != item['sha256']:
            raise ValueError('Changed extracted representation')
        artifacts[item['name']] = np.load(p)
    rows = []; replicates = {}; distance_arrays = {}
    for view in ['phase_state', 'geometry_mixture']:
        mixture = groups if view == 'geometry_mixture' else None
        sources = unique if mixture is not None else source
        def average(x):
            return np.stack([x[g].mean(0) for g in groups]) if mixture is not None else x
        for horizon, block in [('100', slice(0, 128)), ('500', slice(128, 256)), ('100+500', slice(0, 256))]:
            y = values[:, :, block]
            truth = outcome_distance(y, mixture)
            halves = [outcome_distance(y[:, offset::2], mixture) for offset in [0, 1]]
            distance_arrays[f'{view}_{horizon}_future'] = truth
            candidates = [('initial_descriptor', 'control', -1, 'descriptor', squared_distance(average(descriptor)), False),
                ('split_shot_reference', 'reference', -1, 'reference', halves[0], True)]
            for item in contract['models']:
                artifact = artifacts[item['name']]
                z = average(artifact['embedding'].astype(float))
                # Invert the training-only per-coordinate normalization before distances.
                p = average(artifact['prediction'].astype(float)*scale)[:, block]
                candidates.extend((item['name'], item['arm'], item['seed'], space, squared_distance(x), False)
                    for space, x in [('native', z), ('prediction_head', p)])
            for name, arm, seed, space, distance, reference in candidates:
                key = f'{view}_{horizon}_{name}_{space}'
                metric = score(distance, truth, halves, sources, reference)
                jk = jackknife(distance, truth, halves, sources, reference)
                replicates[key] = jk; distance_arrays[key] = distance
                rows.append(dict(model=name, arm=arm, seed=seed, space=space, view=view, horizon_fs=horizon,
                    queries=len(sources), independent_sources=len(unique), candidates_per_query=int(np.sum(sources != sources[0])),
                    negative_target_distance_fraction=float(np.mean(truth[sources[:, None] != sources[None, :]] < 0)), **metric))
            print('scored', view, horizon, flush=True)
    np.savez_compressed(out/'data/distance-matrices.npz', **distance_arrays)
    np.savez_compressed(out/'data/leave-one-source-out.npz', **replicates)
    write_metric_rows(rows, out, family=FAMILY, name='models')
    summary = []
    grouping = sorted(set((r['arm'], r['space'], r['view'], r['horizon_fs']) for r in rows))
    for arm, space, view, horizon in grouping:
        selected = [r for r in rows if (r['arm'], r['space'], r['view'], r['horizon_fs']) == (arm, space, view, horizon)]
        for metric in METRICS:
            scores = [r[metric] for r in selected]
            jk = np.mean([replicates[f'{view}_{horizon}_{r["model"]}_{space}'][:, METRICS.index(metric)] for r in selected], axis=0)
            value = float(np.mean(scores)); low, high = interval(value, jk)
            summary.append(dict(arm=arm, space=space, view=view, horizon_fs=horizon, metric=metric,
                fits=len(scores), mean=value, seed_sd=float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                jackknife_lower=low, jackknife_upper=high))
    write_metric_rows(summary, out, family=FAMILY, name='summary')
    comparisons = []
    for view in ['phase_state', 'geometry_mixture']:
        for horizon in ['100', '500', '100+500']:
            for space in ['native', 'prediction_head']:
                for before, after in [('q_values', 'q_response'), ('qv_values', 'qv_response'), ('q_response', 'qv_response')]:
                    for metric in METRICS:
                        paired = []; jk = []
                        for seed in sorted(set(i['seed'] for i in contract['models'])):
                            def take(arm):
                                return next(r for r in rows if (r['arm'],r['seed'],r['space'],r['view'],r['horizon_fs']) == (arm,seed,space,view,horizon))
                            a, b = take(before), take(after)
                            paired.append(b[metric]-a[metric]); j = METRICS.index(metric)
                            jk.append(replicates[f'{view}_{horizon}_{b["model"]}_{space}'][:,j]-replicates[f'{view}_{horizon}_{a["model"]}_{space}'][:,j])
                        value = float(np.mean(paired)); low, high = interval(value, np.mean(jk,axis=0))
                        comparisons.append(dict(before=before,after=after,space=space,view=view,horizon_fs=horizon,
                            metric=metric,difference=value,jackknife_lower=low,jackknife_upper=high,
                            min_seed_difference=float(min(paired)),max_seed_difference=float(max(paired))))
    write_metric_rows(comparisons, out, family=FAMILY, name='paired-comparisons')
    write(out/'technical/results.json', dict(rows=rows, summary=summary, comparisons=comparisons))
    plots(out, summary)
    report(out, summary, comparisons)
    write(out/'technical/complete.json', dict(state='complete', models=12, sources=30, states=120,
        scientific_training_runs_created=0, new_simulations=0, finished_unix=time.time()))


def plots(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'plots').mkdir(exist_ok=True)
    arms = ['control','q_values','q_response','qv_values','qv_response']
    labels = ['Initial\ndescriptors','Q\nvalues','Q\nresponse','Q+V\nvalues','Q+V\nresponse']
    for view in ['phase_state','geometry_mixture']:
        fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for col, horizon in enumerate(['100','500']):
            for row, metric in enumerate(['gain5','spearman']):
                ax=axes[row,col]
                selected=[next(r for r in summary if r['arm']==arm and r['space']==('descriptor'if arm=='control'else'native') and r['view']==view and r['horizon_fs']==horizon and r['metric']==metric)for arm in arms]
                factor=100 if metric=='gain5'else 1
                y=np.array([r['mean']for r in selected])*factor
                error=np.array([[r['mean']-r['jackknife_lower'],r['jackknife_upper']-r['mean']]for r in selected]).T*factor
                ax.bar(np.arange(5),y,color=['#8c8c8c','#4985b5','#125880','#b48ad2','#744394'])
                ax.errorbar(np.arange(5),y,yerr=error,fmt='none',ecolor='#333333',capsize=3)
                ax.axhline(0,color='#444444',lw=.7)
                ax.set_xticks(np.arange(5),labels);ax.set_title(f'{int(horizon)/1000:g} ps')
                ax.set_ylabel('Future-distance reduction vs random (%)'if metric=='gain5'else'Mean per-query Spearman correlation')
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(('120 phase states; same-source candidates excluded'if view=='phase_state'else'30 geometries; four fixed velocity fields averaged')+'\nNative embeddings; error bars: leave-one-source-out jackknife')
        fig.savefig(out/'plots'/f'{view}.png',dpi=170);fig.savefig(out/'plots'/f'{view}.svg');plt.close(fig)


def report(out, summary, comparisons):
    lines=['# Similarity of future outcomes','',
        'Twelve frozen models; 30 held-out Al ancestors, four sampled velocity fields per ancestor. No new training or MD. Selection/calibration sources are not used.', '',
        'Primary view: 120 phase states, 116 candidates from other ancestors per query. The secondary view averages embeddings and distributions across four fixed velocity fields per geometry (30 queries, 29 candidates); it is not an exact velocity-marginal law.', '',
        'Future distance is noise-corrected squared distance between raw Fourier-feature means. Each target includes absolute future structure and change from the initial structure. Results describe these finite features, not full atomic trajectory distributions.', '',
        '| Representation | Correlation, 0.1 ps | Correlation, 0.5 ps | Top-5 future-distance reduction, 0.1 ps | Top-5 reduction, 0.5 ps |',
        '|---|---:|---:|---:|---:|']
    for arm in ['control','q_values','q_response','qv_values','qv_response']:
        space='descriptor'if arm=='control'else'native'
        def get(metric,h):return next(r['mean']for r in summary if(r['arm'],r['space'],r['view'],r['horizon_fs'],r['metric'])==(arm,space,'phase_state',h,metric))
        lines.append(f'| {arm} | {get("spearman","100"):.3f} | {get("spearman","500"):.3f} | {100*get("gain5","100"):.1f}% | {100*get("gain5","500"):.1f}% |')
    lines += ['', 'Higher correlation and larger distance reduction are better. Random retrieval has zero expected gain. Descriptor control uses the original 24 radial/angular observables standardized by the recorded training-only normalizer. No label-based fitting or retrieval tuning.', '',
        'All native/head spaces, horizons (including the equally weighted concatenation), top-1/top-5 recalls, split-shot reliability references, seed spread and paired source-jackknife intervals are in [tables/summary.csv](tables/summary.csv) and [tables/paired-comparisons.csv](tables/paired-comparisons.csv).', '',
        'The split-shot reference selects neighbors from one eight-shot half and scores them on the independent half. It measures finite-shot reliability; it is not a formal performance ceiling. Negative corrected distances are retained. Tied distances receive fractional neighbor weights instead of arbitrary velocity-field selection.', '',
        'Jackknife intervals delete each source from both the query and candidate populations and recompute rankings. They are approximate for these nonsmooth retrieval metrics, conditional on the fitted seeds. Only 30 source ancestors are independent; pair counts and velocity replicas do not increase the sample size.', '',
        '![Phase-state retrieval](plots/phase_state.png)', '', '![Four-field mixture retrieval](plots/geometry_mixture.png)', '']
    (out/'README.md').write_text('\n'.join(lines))


def tracking(c):
    from src.experiment_runner.wandb_tracking import update_recorded_summary
    campaign,out=paths(c);rows=read(out/'technical/results.json')['rows']
    for item in read(out/'technical/binding.json')['models']:
        selected=[r for r in rows if r['model']==item['name']]
        fields={f'future_similarity/{r["view"]}/{r["space"]}/{r["horizon_fs"]}/{m}':r[m]for r in selected for m in METRICS}
        update_recorded_summary(campaign/'analyses'/item['name']/'technical/wandb.json', fields,
            evaluation='future-similarity-v1',expected=dict(entity='teshbek',project='PointCloudMaterials'))
        print('updated training summary',item['name'],flush=True)
    write(out/'technical/tracking-complete.json',dict(state='complete',updated_training_runs=12,new_runs=0))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','extract','analyze','tracking'])
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();c=read(a.config)
    if a.stage=='extract':extract(c,a.lane)
    else:globals()[a.stage](c)


if __name__=='__main__':main()
