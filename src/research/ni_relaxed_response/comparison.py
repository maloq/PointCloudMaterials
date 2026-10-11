"""Paired held-out birth/appearance comparisons on unchanged physical rows."""
import numpy as np
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws,bootstrap_scores
from src.experiment_runner.metric_docs import write_metric_rows
from .data import root,resolve_path,read,write_json,FAMILY
from .evaluate import cohorts


def run(c,p):
    parent=resolve_path(c['parent']);report=[]
    for task,h,_,_,records in cohorts(c):
        for model in ['spatial']+[a['name'] for a in c['arms']]:
            for space in c['evaluation']['spaces']:
                for kind in ['linear','catboost']:
                    relative=f'analyses/{task}/analyses/fits-v1/{model}/{space}/lead-{h:g}ps/{kind}/technical/predictions.npz'
                    with np.load(parent/relative) as a, np.load(root(c)/relative) as b:
                        for k in ['id','ancestor','role','label','weight']:np.testing.assert_array_equal(a[k],b[k])
                        mask=np.isin(b['role'],['selection','calibration','test']);y=b['label'][mask];w=b['weight'][mask];old=a['probability'][mask];new=b['probability'][mask]
                        ancestors,slot,counts=source_draws(b['ancestor'][mask],c['evaluation']['bootstrap_draws'],c['seed'])
                    ov=scores(y,old,w);nv=scores(y,new,w);ob,_=bootstrap_scores(y,old,w,slot,counts,None);nb,_=bootstrap_scores(y,new,w,slot,counts,None)
                    for metric in ['nll','brier','ap','auroc']:
                        delta=nb[metric]-ob[metric];lo,hi=np.quantile(delta,[.025,.975])
                        report.append(dict(model=model,task=task,offset_ps=h,space=space,readout=kind,metric=metric,unrelaxed=ov[metric],relaxed=nv[metric],difference=nv[metric]-ov[metric],lo=lo,hi=hi,test_rows=len(y),test_ancestors=len(ancestors)))
    out=root(c)/'analyses/paired-comparison-v1';write_metric_rows(report,out,family=FAMILY,name='relaxed-minus-unrelaxed',columns=list(report[0]))
    write_json(out/'technical/complete.json',dict(state='complete',rows=len(report),parent=str(parent),future_descriptor_scores_compared=False,reason='Future geometry targets differ; compare identical original-MD birth/appearance labels only'))
