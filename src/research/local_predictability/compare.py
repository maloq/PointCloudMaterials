"""Paired source comparisons and sparse native-row alignment of descriptor fits."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.project_runtime.paths import load_json,resolve_path
from src.data.predictive_memory.prepare import write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .baselines import score_hazard
from .metrics import stratified_bootstrap


def compare(config):
    root=resolve_path(config['output']);technical=root/'technical'
    plan=json.loads(resolve_path(config['plan']).read_text())
    release=json.loads((technical/'release.json').read_text())
    results=json.loads((technical/'descriptor_results.json').read_text())
    paired=[];matched={}
    names=[name for name in results if name.startswith(('linear-','mlp-'))]
    for name in names:
        path=technical/'descriptors'/name/'predictions.npz';saved=np.load(path)
        mask=np.isin(saved['anchor'],release['native_anchors'])
        arrays={key:saved[key][mask] for key in ['source','split','temperature','center','anchor']}
        arrays['y']=saved['event_bin'][mask]
        masks={split:np.flatnonzero(arrays['split']==split) for split in ['train','selection','calibration','test']}
        matched[name]=score_hazard(arrays,saved['probability'][mask],saved['logits'][mask],masks,plan)
    comparisons=[]
    for model in ['linear','mlp']:
        comparisons.extend([(f'{model}-packet_H0',f'{model}-condition'),
            *[(f'{model}-packet_H{h}',f'{model}-packet_H0') for h in (3,12,48)],
            (f'{model}-packet_H12',f'{model}-packet_repeat12'),
            (f'{model}-packet_plus_center_H0',f'{model}-packet_H0'),
            (f'{model}-packet_plus_shell25_H0',f'{model}-packet_H0')])
    for population,collection in [('descriptor_grid',results),('native_grid',matched)]:
        for candidate,reference in comparisons:
            if candidate not in collection or reference not in collection:continue
            for horizon in plan['sampling']['horizons_ps']:
                a={r['source_id']:r for r in collection[candidate]['per_source'] if r['horizon_ps']==horizon}
                b={r['source_id']:r for r in collection[reference]['per_source'] if r['horizon_ps']==horizon}
                if set(a)!=set(b):raise ValueError(f'Unpaired sources: {candidate}, {reference}')
                sources=sorted(a)
                values=np.array([[a[s][m]-b[s][m] for m in ['log_loss','brier']] for s in sources])
                bounds=stratified_bootstrap(values,[a[s]['temperature_K'] for s in sources])
                for i,metric in enumerate(['log_loss','brier']):
                    paired.append(dict(population=population,candidate=candidate,reference=reference,horizon_ps=horizon,
                        metric=metric,difference=float(values[:,i].mean()),ci95_lower=float(bounds[0,i]),ci95_upper=float(bounds[1,i]),sources=len(sources)))
    snapshot_metric_docs(root,'local_predictability_comparison')
    with (root/'tables/paired_descriptor_differences.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    write_json(technical/'native_grid_descriptor_scores.json',matched)
    write_json(technical/'paired_descriptor_differences.json',paired)
    write_json(technical/'comparison_status.json',dict(state='complete',models=len(names),seed=20260919,
        intervals='Paired temperature-stratified whole-source bootstrap; conditional on one seed; unadjusted exploratory intervals',
        dense_alarm_status='Not computed by this stage; 3-ps scores are never interpolated into 0.75-ps alarms'))
    print('Matched descriptor comparison complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();compare(load_json(args.config))


if __name__=='__main__':main()
