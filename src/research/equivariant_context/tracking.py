"""Expose existing scientific results in their original online runs; never create runs."""
import json
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import write_json, sha
from src.research.supervised_onset.tracking import final_fields, validation_fields, require_online


def sync_completed(study):
    import wandb
    require_online(study.config['wandb'])
    api=wandb.Api(timeout=60);updated=[]
    labels=dict(symmetric_invariant='Symmetric invariant',vector_messages='Vector messages',
                tensor_attention='Tensor attention',harmonic_hierarchy='Harmonic hierarchy')
    cohort='Al64' if 'fixed_dataset' in study.config else 'Al16'
    def update(receipt,fields,name,evidence):
        run=api.run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
        run.summary.update(fields)
        run.name=name;run.update()
        updated.append(dict(id=receipt['id'],url=receipt['url'],fields=list(fields),evidence=evidence))
        write_json(study.technical/'wandb-summary-sync.json',dict(updated=updated,created_runs=0))
    for domain in study.config['domains']:
        label='Observed' if domain=='hot' else 'Relaxed'
        for variant in study.config['variants']:
            root=study.root/f'{domain}-{variant}'/'technical'
            complete=root/'complete.json'
            if not complete.exists():continue
            record=json.loads(complete.read_text())
            receipt=json.loads((root/'wandb'/f'{domain}-{variant}'/'run.json').read_text())
            if record['identity']!=receipt['identity'] or record['state']!='complete':
                raise ValueError(f'Run receipt disagrees with completed results: {root}')
            metrics=json.loads((root/'metrics.json').read_text())
            update(receipt,final_fields(metrics,record),f'{cohort} | {label} | {labels[variant]}',
                   dict(metrics_sha256=sha(root/'metrics.json'),completion_sha256=sha(complete)))
        config=json.loads(resolve_path(study.config['base_configs'][domain]).read_text())
        technical=resolve_path(config['output'])/'technical';arm=config['arms'][0]['name']
        state_path=technical/'runs'/arm/'training-state.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text())
        if state['state']!='update_budget_complete':continue
        receipt=json.loads((technical/'wandb'/arm/'run.json').read_text())
        if state['identity']!=receipt['identity']:raise ValueError('Encoder tracking identity differs')
        best=state['best']
        fields=validation_fields(best)|{'checkpoint/selected_update':best['update'],
            'checkpoint/validation_event_nll':best['nll']}
        update(receipt,fields,f'{cohort} | {label} | Shared MACE encoder',dict(state_sha256=sha(state_path)))
    return dict(updated_runs=len(updated),created_runs=0)
