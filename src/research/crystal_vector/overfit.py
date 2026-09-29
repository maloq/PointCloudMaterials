"""Local saved-prediction audit of fit gaps, feature spectra and source separation."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.spatial_approach.evaluate import csv_rows
from .evaluate import spectrum
from .unseen import alarm_tables


def population_weights(meta,role):
    ids=np.flatnonzero((meta['role']==role)&(meta['kind']<2));w=np.zeros(len(meta['atom']))
    kinds=np.unique(meta['kind'][ids])
    for kind in kinds:
        selected=ids[meta['kind'][ids]==kind]
        sources,inv,count=np.unique(meta['source'][selected],return_inverse=True,return_counts=True)
        w[selected]=1/(len(kinds)*len(sources)*count[inv])
    return w


def run(output):
    output=resolve_path(output);snapshot_metric_docs(output,'crystal_overfit')
    scores=[];features=[];neighbors=[];curves=[];alarms=[];inputs=[]
    for study in ('crystal_vector','crystal_interface'):
        root=resolve_path('${storage:analysis}')/study/'al64-random-20260928'
        c=json.loads((root/'technical/code/config.json').read_text())
        plan=json.loads((resolve_path(c['dataset']['root'])/'plan.json').read_text())
        sources=plan['sources']
        lineage_roles={}
        for s in sources:
            lineage_roles.setdefault(s['lineage'],set()).add(s['role'])
        if any(len(v)!=1 for v in lineage_roles.values()):raise ValueError('Ancestor leakage across roles')
        for variant in ('distance_only','distance_direction','distance_direction_vcreg'):
            runroot=root/variant;tech=runroot/'technical';a=runroot/'analyses/localization-v1'
            done=json.loads((a/'technical/complete.json').read_text())
            pred=a/'technical/predictions.npz';receipt=json.loads((a/'technical/predictions.json').read_text())
            if sha(pred)!=receipt['sha256'] or done['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('Changed saved inference')
            with np.load(a/'technical/rows.npz') as f:meta={k:f[k] for k in f.files}
            with np.load(pred) as f:values={k:f[k] for k in f.files}
            # Identical source/center/frame tuples cannot cross a role; sources and
            # independent-melt ancestry remain disjoint even when windows overlap.
            if any(len(np.unique(meta['role'][meta['source']==s['id']]))!=1 for s in sources):raise ValueError('Source crosses roles')
            inputs.append(dict(study=study,variant=variant,sources=len(sources),ancestors=len(lineage_roles),
                cross_role_ancestors=0,predictions_sha256=receipt['sha256'],
                tensor_inputs=['positions','inverse','actual'],labels_excluded=['distance','direction','valid','visible_context','inside_crystal','source','frame','role'],
                encoder_input='constant atom channel; geometry only; radius<8 atoms and cutoff<5 edges',
                predictor_input='learned scalar/vector exports and relative patch geometry; no conditions'))
            for role in ('train','selection','calibration','test'):
                base=population_weights(meta,role)
                for visibility,take in [('all',np.ones(len(base),bool)),('invisible',~meta['visible_context']),('visible',meta['visible_context'])]:
                    for population,which in [('mixture',meta['kind']<2),('fixed_at_risk',meta['kind']==0),('uniform',meta['kind']==1)]:
                        ids=np.flatnonzero((base>0)&take&which)
                        if not len(ids):continue
                        w=base[ids];w=w/w.sum();d=meta['distance'][ids];target=np.minimum(d,64)
                        valid=meta['direction_valid'][ids];valid_mass=float(w@valid)
                        scores.append(dict(study=study,variant=variant,role=role,visibility=visibility,population=population,
                            rows=len(ids),sources=len(np.unique(meta['source'][ids])),distance_nll=float(w@-values['log_likelihood'][ids]),
                            capped_mean_rmse_A=float(np.sqrt(w@(values['mean_A'][ids]-target)**2)),
                            capped_median_mae_A=float(w@np.abs(values['median_A'][ids]-target)),
                            brier20=float(w@(values['cdf'][ids,3]-(d<=20))**2),
                            prevalence20=float(w@(d<=20)),censored_mass=float(w@(d>=64)),
                            direction_nll_valid=float(w@values['direction_nll'][ids]/valid_mass) if valid_mass and variant!='distance_only' else None))
                ids=np.flatnonzero(base>0)
                for field in ('local_z','context_z','local_v'):
                    x=values[field][ids]
                    if not np.isfinite(x).all():raise ValueError(f'Nonfinite exported {study}/{variant}/{field}')
                    features.append(dict(study=study,variant=variant,role=role,field=field,rows=len(ids),
                        constant_channels=int((x.reshape(len(x),-1).std(0)<1e-6).sum()),
                        rms_norm=float(np.sqrt(np.mean(np.sum(x.reshape(len(x),-1)**2,axis=1)))),**spectrum(x)))
            train=np.flatnonzero((meta['role']=='train')&(meta['kind']<2));rng=np.random.default_rng(20260928)
            perm=rng.permutation(train);reference=perm[:4096];train_query=perm[4096:4608]
            for field in ('local_z','context_z'):
                scale=values[field][train].std(0).clip(1e-6);bank=values[field][reference]/scale
                for role in ('train','selection','test'):
                    pool=np.flatnonzero((meta['role']==role)&(meta['kind']<2))
                    ids=train_query if role=='train' else rng.choice(pool,min(512,len(pool)),replace=False)
                    distance=cdist(values[field][ids]/scale,bank).min(1)
                    neighbors.append(dict(study=study,variant=variant,field=field,role=role,reference_rows=len(reference),query_rows=len(ids),
                        median_nearest_train_distance=float(np.median(distance)),p95_nearest_train_distance=float(np.quantile(distance,.95))))
            val=[json.loads(x) for x in (tech/'validation.jsonl').read_text().splitlines()]
            logs=[json.loads(x) for x in (tech/'training.jsonl').read_text().splitlines()]
            best=min(val[11:],key=lambda r:r['validation/predictive_objective'])['train/epoch']
            for v in val:
                epoch=v['train/epoch'];local=[x['train/predictive_objective'] for x in logs if epoch-1<x['train/epoch']<=epoch]
                curves.append(dict(study=study,variant=variant,epoch=epoch,selected=epoch==best,
                    train_logged_objective_mean=float(np.mean(local)),validation_objective=v['validation/predictive_objective'],
                    validation_distance_nll=v['validation/distance_nll']))
            if study=='crystal_interface':
                alarms.extend(dict(variant=variant,**r) for r in alarm_tables(values,meta,plan))
            print(json.dumps(dict(stage='audited',study=study,variant=variant)),flush=True)
    for name,data in [('fit-gaps',scores),('feature-spectra',features),('feature-neighbors',neighbors),('learning-curves',curves),('unseen-alarms',alarms)]:
        csv_rows(output/'tables'/f'{name}.csv',data)
    write_json(output/'technical/input-audit.json',inputs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for i,study in enumerate(('crystal_vector','crystal_interface')):
        for j,variant in enumerate(('distance_only','distance_direction','distance_direction_vcreg')):
            records=[r for r in curves if r['study']==study and r['variant']==variant];ax=axes[i,j]
            ax.plot([r['epoch'] for r in records],[r['train_logged_objective_mean'] for r in records],label='Training minibatch mean')
            ax.plot([r['epoch'] for r in records],[r['validation_objective'] for r in records],label='Full validation')
            ax.set_title(study+'\n'+variant,fontsize=9);ax.set_xlabel('Nominal epoch');ax.set_ylabel('Predictive objective');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout();(output/'plots').mkdir(exist_ok=True);fig.savefig(output/'plots/learning-curves.png',dpi=160);plt.close(fig)
    write_json(output/'technical/complete.json',dict(models=len(inputs),tables={p.name:sha(p) for p in (output/'tables').glob('*.csv')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    run(p.parse_args().output)
