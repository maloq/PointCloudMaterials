"""Matched native-0.10-ps outer predictions, proper scores and ancestry uncertainty."""
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from . import dense_data as data
from .dense import FAMILY, tasks, fit_root
from .data import read
from .temporal_analysis import metric_bundle, interval


def collect(c,design,progress):
    rows,_,manifest = data.dataset(c)
    if manifest['identity'] != design['dataset']: raise ValueError('Changed held-out dataset')
    root = resolve_path(c['output'])/'analyses/nested-cv-v1'
    (root/'technical').mkdir(parents=True,exist_ok=True)
    summary,folds,paired,calibration,evidence = [],[],[],[],[]
    cached = {}; ids = np.arange(len(rows['id']))
    bootstrap_rows = dict(rows,source=rows['ancestor'])
    with threadpool_limits(limits=c['analysis_threads']):
        for spec in tasks(c):
            arm,obs = spec['arm'],spec['observation']['name']
            probability,prior = np.full(len(ids),np.nan),np.full(len(ids),np.nan)
            covered = np.zeros(len(ids),int); train_scores = []
            for fold in range(c['outer_folds']):
                tech = fit_root(c,spec,fold)/'technical'
                done,split = read(tech/'complete.json'),read(tech/'outer-split.json')
                if split['design'] != design['identity'] or sha(tech/'predictions.npz') != done['predictions_sha256']:
                    raise ValueError('Changed dense outer fit')
                with np.load(tech/'predictions.npz') as saved:
                    if not np.array_equal(saved['id'],rows['id']): raise ValueError('Changed outer row order')
                    held = saved['role'] == 'merged_test'
                    expected = np.asarray([design['outer'][g] == fold for g in rows['ancestor']])
                    if not np.array_equal(held,expected): raise ValueError('Outer predictions include training ancestors')
                    probability[held] = saved['probability'][held]; prior[held] = split['train_prior']; covered[held] += 1
                for score in done['scores']:
                    role = dict(train='outer_training',training_selection_cv='inner_selection',merged_test='outer_test')[score['role']]
                    folds.append(dict(fold=fold,**dict(score,role=role)))
                    if role == 'outer_training': train_scores.append(score)
                evidence.append(dict(arm=arm,observation=obs,fold=fold,identity=done['identity'],
                                     predictions_sha256=done['predictions_sha256']))
            if not np.all(covered == 1) or not np.isfinite(probability).all():
                raise ValueError('Each row requires exactly one finite outer-held-out prediction')
            progress.update(analysis=arm+'/'+obs)
            np.savez_compressed(root/'technical'/f'{arm}-{obs}.npz',probability=probability,prior=prior,
                outer_fold=np.asarray([design['outer'][g] for g in rows['ancestor']]),**rows)
            for stratum in ('combined','established','failed_all'):
                mask = (np.ones(len(ids),bool) if stratum == 'combined' else
                        rows['stratum'] == 'established' if stratum == 'established' else rows['stratum'] != 'established')
                selected = ids[mask]
                if not len(selected): continue
                value,boot = metric_bundle(bootstrap_rows,selected,probability[mask],c)
                base,base_boot = metric_bundle(bootstrap_rows,selected,prior[mask],c)
                summary.append(dict(arm=arm,observation=obs,stratum=stratum,rows=len(selected),
                    events=len(set(zip(rows['source'][mask],rows['event'][mask]))),ancestors=len(set(rows['ancestor'][mask])),
                    mean_train_nll=float(np.mean([s['nll'] for s in train_scores])),
                    mean_train_brier=float(np.mean([s['brier'] for s in train_scores])),prior_nll=base['nll'],
                    prior_brier=base['brier'],prior_ap=base['ap'],**value))
                cached[(arm,obs,stratum)] = value,boot
                for metric in ('nll','brier'):
                    lo,hi = interval(boot[metric]-base_boot[metric])
                    paired.append(dict(arm=arm,stratum=stratum,contrast='model_minus_prior',left=obs,right='prior',
                        metric=metric,difference=value[metric]-base[metric],lo=lo,hi=hi))
                bins = np.minimum((probability[mask]*10).astype(int),9)
                for k in range(10):
                    chosen = selected[bins == k]
                    if len(chosen):
                        calibration.append(dict(arm=arm,observation=obs,stratum=stratum,bin=k,rows=len(chosen),
                            predicted_mean=float(np.average(probability[chosen],weights=rows['weight'][chosen])),
                            positive_fraction=float(np.average(rows['label'][chosen],weights=rows['weight'][chosen]))))
        for (arm,obs,stratum),(left,ll) in cached.items():
            if obs != 'history6': continue
            for reference in ('snapshot','repeated6','early_snapshot'):
                right,rr = cached[(arm,reference,stratum)]
                for metric in ('nll','brier','ap','auroc','matched_auc'):
                    lo,hi = interval(ll[metric]-rr[metric])
                    paired.append(dict(arm=arm,stratum=stratum,contrast='history_minus_control',left=obs,right=reference,
                        metric=metric,difference=left[metric]-right[metric],lo=lo,hi=hi))
    for name,values in [('out-of-fold',summary),('fold-train-and-test',folds),('paired-comparisons',paired),('calibration',calibration)]:
        write_metric_rows(values,root,family=FAMILY,name=name)
    lines = ['# Native 0.10 ps appearance prediction','',
        'Six original-MD observations at −0.6 through −0.1 ps before first local PTM crystal appearance.',
        'Five outer and five inner ancestry folds; NLL selection; one seed; identical cohorts for all treatments.',
        'Retrospective event/control sampling, 20% prevalence. This is not a population-wide onset alarm.',
        'Ancestry-bootstrap intervals condition on the fitted predictions and do not include refitting/seed uncertainty.',
        '', '| Model | Input | Train NLL | Held-out NLL | AP | AUROC |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['stratum'] == 'combined':
            lines.append(f'| {r["arm"]} | {r["observation"]} | {r["mean_train_nll"]:.4f} | {r["nll"]:.4f} | {r["ap"]:.4f} | {r["auroc"]:.4f} |')
    lines += ['', '[Scores and intervals](tables/out-of-fold.csv) · [Paired differences](tables/paired-comparisons.csv) · '
              '[Train/test errors](tables/fold-train-and-test.csv) · [Definitions](tables/METRICS.md)','']
    (root/'README.md').write_text('\n'.join(lines))
    write_json(root/'technical/complete.json',dict(design=design['identity'],fits=len(evidence),evidence=evidence,
        predictions={p.name:sha(p) for p in (root/'technical').glob('*.npz')}))
