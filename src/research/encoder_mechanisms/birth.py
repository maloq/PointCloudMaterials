"""Frozen held-out birth availability, never a future-centered predictor dataset."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha,write_json
from src.research.crystallization_origin.harvest import run,load_plan,CAUSES
from src.experiment_runner.metric_docs import write_metric_table
from .workflow import load


def run_role(c,index):
    role=('selection','calibration','test')[index]
    seal=resolve_path(c['birth_training_definition'])
    config=dict(json.loads(seal.read_text())['config'])
    root=resolve_path(c['output'])/'analyses/birth-availability'/role
    config.update(protocol='nucleus-harvest-heldout-audit-v2',role=role,output=str(root),
        training_definition=str(seal),training_definition_sha256=sha(seal),support_radii_A=[8.,32.],review_sources=[])
    plan=load_plan(config);write_json(root/'technical/frozen-config.json',config)
    run(config,c['birth_workers'],None)
    result={}
    for threshold in ('primary','size32','size128','persistent5'):
        result[threshold]={}
        for radius in (8,32):
            result[threshold][str(radius)]={}
            for criterion in ('established','ptm_all'):
                block={}
                for history in config['history_ps']:
                    counts=np.zeros((2,len(CAUSES)),np.int64);eligible_rows=0;exposure=0.;events=[set(),set()];sources=[set(),set()]
                    for source in plan['sources']:
                        path=root/f'technical/sources/{source["id"]}/{threshold}-references.npz'
                        with np.load(path) as a:
                            valid=(a['control']&a['eligible']&a[f'{criterion}_clear_{radius}A']&(a['observed_history_ps']>=history))
                            eligible_rows+=int(valid.sum())
                            exposure+=float(valid.sum()/a['uniform_control_inclusion_probability'])
                            for h in (0,1):
                                for cause in range(len(CAUSES)):counts[h,cause]+=int((valid&(a['label_code'][:,h]==cause)).sum())
                                positives=valid&(a['label_code'][:,h]==1)
                                events[h].update((source['id'],int(e)) for e in a['first_birth_event'][positives])
                                if positives.any():sources[h].add(source['id'])
                    block[str(history)]=dict(uniform_eligible_rows=eligible_rows,estimated_eligible_atom_origins=exposure,
                        outcomes={str(h):dict(counts=dict(zip(CAUSES,map(int,counts[i]))),
                            distinct_source_births=len(events[i]),sources_with_birth=len(sources[i])) for i,h in enumerate(config['horizons_ps'])})
                result[threshold][str(radius)][criterion]=block
    result['limitations']=dict(
        status='availability audit only; not a released predictive population',
        sampling='Outcome-independent uniform atom/origin draw only in these tables; event-centered candidate examples excluded',
        support='8 A local sphere; conservative 32 A spherical envelope for a hypothetical context predictor; not a halo claim',
        competing_outcomes='First confirmed-lineage contact competes with local establishment; ambiguous events remain separate',
        censoring='Only complete follow-up plus confirmation padding; causal current eligibility',
        weights='Inverse known uniform inclusion probability estimates atom-origin exposure; windows are correlated',
        holdout='Fixed original roles; this coverage inspection consumes the release for benchmark design',
        future_work='Probability fitting remains gated on adequate distinct held-out events and a sealed prediction release')
    write_json(root/'technical/support-availability.json',result)
    write_metric_table(result,root/'analyses/support-availability',family='encoder_mechanisms')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--index',type=int);args=parser.parse_args()
    run_role(load(args.config),args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__=='__main__':main()
