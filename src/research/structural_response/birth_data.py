"""Bind motion birth evaluation to the consolidated, exact-cadence catalogue.

This prepares the source/event contract. It does not turn catalogue counts into
prediction rows: lead availability and matched controls
controls still require the observation extraction stage.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from src.postprocessing.datasets.births import NucleiBirthDataset
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path


def prepare(c):
    spec=c['birth_evaluation']
    if spec['dataset']!='al-nuclei-birth-010':raise ValueError('Motion birth evaluation requires the exact010 catalogue')
    if spec['neighborhood_filter'] is not False:raise ValueError('User disabled additional neighborhood filtering')
    split=json.loads((resolve_path(c['fixed_dataset']['root'])/'splits.json').read_text())
    if split['identity']!=c['fixed_dataset']['identity']:raise ValueError('Changed fixed Al ancestry contract')
    roles={s['lineage']:s['role'] for s in split['sources']}
    sources=[];excluded=[];selected=[]
    with NucleiBirthDataset('010',expected_fingerprint=spec['fingerprint']) as data:
        events=list(data.events(selection=spec['selection'],detector=spec['detector']))
        by_source=Counter(e['source_id'] for e in events)
        rows=list(data.db.execute('''SELECT s.source_id,a.role_conflict,t.task_id,t.folder,t.files
            FROM sources s JOIN views v USING(source_id) JOIN tasks t USING(view_id)
            JOIN ancestry a USING(ancestor) WHERE v.cadence=? AND t.detector=?
            ORDER BY s.source_id''',('010',spec['detector'])))
        for row in rows:
            s=data.source(row['source_id']);reasons=[];manifests=[]
            if row['role_conflict']:reasons.append('historical_ancestor_role_conflict')
            if s['ancestry_group'] not in roles:reasons.append('outside_fixed_ancestry_contract')
            elif roles[s['ancestry_group']]!=s['role']:raise ValueError('Fixed ancestor role mismatch: '+s['id'])
            for part in s['parts']:
                path=Path(part['path'])/'manifest.json'
                if sha(path)!=part['manifest_sha256']:raise ValueError('Changed source manifest: '+str(path))
                manifest=json.loads(path.read_text());manifests.append(manifest)
                if 'velocities' not in manifest['arrays']:reasons.append('no_saved_velocities')
                elif manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Unexpected velocity units: '+s['id'])
            times=np.asarray(s['times_ps'])[s['views']['010']]
            if not np.allclose(np.diff(times),.1,atol=1e-8,rtol=0):raise ValueError('Nonexact010 timeline: '+s['id'])
            if reasons:
                excluded.append(dict(source_id=row['source_id'],source=s['id'],dataset=s['dataset'],
                    ancestor=s['ancestry_group'],role=s['role'],reasons=sorted(set(reasons)),
                    established_event_records=by_source[row['source_id']]))
                continue
            for part,manifest in zip(s['parts'],manifests,strict=True):
                q=manifest['arrays']['positions'];v=manifest['arrays']['velocities']
                if q['shape']!=v['shape']:raise ValueError('Position/velocity shape mismatch: '+s['id'])
                for name in ('positions','velocities'):
                    a=manifest['arrays'][name];path=Path(part['path'])/a['file']
                    array=np.load(path,mmap_mode='r',allow_pickle=False)
                    if list(array.shape)!=a['shape'] or str(array.dtype)!=a['dtype']:raise ValueError('Changed observation array: '+str(path))
            # Preserve the original source manifest, raw frame mapping and label
            # shard identity. A future catalogue refresh cannot rewrite this pin.
            sources.append(dict(source_id=row['source_id'],source=s,task_id=row['task_id'],
                label_folder=row['folder'],label_files=json.loads(row['files']),
                evaluation_role='train' if s['role']=='train' else 'test'))
        admitted={s['source_id'] for s in sources}
        selected=[dict(e,evaluation_role='train' if e['role']=='train' else 'test')
                  for e in events if e['source_id'] in admitted]
        result=dict(dataset=spec['dataset'],fingerprint=data.fingerprint,
            dataset_manifest_sha256=data.manifest_sha256,database_sha256=data.manifest['files']['catalogue.sqlite'],
            fixed_dataset=c['fixed_dataset'],protocol=spec,sources=sources,events=selected,excluded=excluded,
            counts=dict(catalogue_sources=len(rows),catalogue_established_event_records=len(events),
                velocity_eligible_sources=len(sources),velocity_eligible_ancestors=len({s['source']['ancestry_group'] for s in sources}),
                velocity_eligible_established_event_records=len(selected),
                event_records_by_original_role=dict(Counter(e['role'] for e in selected)),
                event_records_by_evaluation_role=dict(Counter(e['evaluation_role'] for e in selected))),
            stage='source_and_event_contract_prepared',
            pending=['exact lead observations','matched controls','physical event/view deduplication'],
            neighborhood_filter=False,
            interpretation='Catalogue event records are candidates, not eligible prediction cases or independent melt counts.',
            encoder_inputs='unrelaxed nearest80 positions and actual same-frame velocities; no conditions',
            predictor_inputs='frozen representation; no temperature, absolute time, simulation age or birth/fate metadata',
            selection='predictive NLL within original training ancestors; original selection/calibration/test pooled only for final evaluation')
        result['identity']=digest(result)
        folder=resolve_path(c['output'])/'technical';path=folder/'birth-cohort.json'
        if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Changed birth cohort; use another experiment revision')
        write_json(path,result);data.pin(folder/'nuclei-birth-dataset.json')
    return result
