"""Pinned Ni nucleus readout rows, preserving the original case-control assay."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from src.data.splits import train_test_role
from src.experiment_runner.metric_docs import check_metric_docs
from src.postprocessing.io import arrays
from src.research.ni_response import data as original
from .data import read, root, cache, sha, digest, write_json, atomic_save

FAMILY = 'ni_phase_births'


def prepare(c):
    phase_path = Path(c['phase_run'])/'technical/plan.json'
    phase = read(phase_path)
    if sha(phase_path) != c['phase_plan_sha256']:
        raise ValueError('Changed phase-model source plan')
    old = read(c['original_config'])
    if sha(cache(old)/'births.pt') != c['original_births_sha256']:
        raise ValueError('Changed original birth cohort')
    checkpoints = []
    for arm in phase['config']['arms']:
        folder = Path(c['phase_run'])/'analyses'/arm['name']/'technical'
        done = read(folder/'complete.json')
        if done['updates'] != 8000 or sha(folder/'final.pt') != done['checkpoint_sha256']:
            raise ValueError('Incomplete or changed final checkpoint: '+arm['name'])
        checkpoints.append(dict(arm=arm, path=str(folder/'final.pt'), sha256=done['checkpoint_sha256']))
    old_plan = read(root(old)/'technical/plan.json')
    additional = Path(c['additional_catalogue'])
    if sha(additional/'technical/plans.json') != c['additional_plan_sha256']:
        raise ValueError('Changed additional Ni birth catalogue')
    if read(additional/'technical/complete.json')['state'] != 'complete':
        raise ValueError('Additional birth catalogue is incomplete')
    catalogue = read(additional/'technical/plans.json')
    sources = [dict(s) for s in old_plan['sources']]
    sources += [dict(key=s['id'], source=s, search=str(additional), original_role=s['role'])
                for s in catalogue['parent']['sources']]
    phase_sources = {s['key']: s for s in phase['sources']}
    if {s['key'] for s in sources} != set(phase_sources):
        raise ValueError('Birth / phase source inventories differ')
    event_receipts = []
    for item in sources:
        prior = phase_sources[item['key']]
        if (item['source']['input_identity'] != prior['source']['input_identity']
                or item['source']['role'] != prior['source']['role']):
            raise ValueError('Birth / phase source identity or role differs')
        item['catalogue_role'] = item['source']['role']
        item['original_role'] = prior['original_role']
        path = Path(item['search'])/'technical/sources'/item['key']/'010/ptm/events.json'
        actual = sha(path)
        event_receipts.append(dict(source=item['key'], path=str(path), sha256=actual))
        item['events_sha256'] = actual
    ancestry = {s['source']['ancestry_group']: train_test_role(s['original_role']) for s in sources}
    for s in sources:
        if ancestry[s['source']['ancestry_group']] != train_test_role(s['original_role']):
            raise ValueError('Ancestor crosses original train/test')
    p = dict(config=c, phase_config=phase['config'], sources=sources, checkpoints=checkpoints,
             columns=phase['columns'], original_config=old, events=event_receipts,
             partition={r: sorted(a for a, role in ancestry.items() if role == r) for r in ['train', 'test']},
             implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    p['identity'] = digest(p)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != p:
        raise ValueError('Changed frozen birth evaluation; use a new revision')
    write_json(destination, p)
    return p


def collect(c, p):
    """Reuse exact old rows and add the completed thirty-source catalogue."""
    path = cache(c)/'cohort.pt'
    if path.exists() and (cache(c)/'cohort.json').exists():
        done = read(cache(c)/'cohort.json')
        if done['identity'] != p['identity'] or sha(path) != done['sha256']:
            raise ValueError('Changed birth evaluation cache')
        return torch.load(path, map_location='cpu', weights_only=False)
    old = original.load(p['original_config'], 'births')
    old_sources = {r['source'] for r in old['rows']}
    # Include sources with no births in the original source inventory too.
    old_plan = read(root(p['original_config'])/'technical/plan.json')
    old_sources.update(s['key'] for s in old_plan['sources'])
    packets = [old]
    recipe = dict(p['original_config'], cache=str(cache(c)/'additional'))
    for item in p['sources']:
        if item['key'] in old_sources:
            continue
        source = item['source']
        for part in source['parts']:
            raw = arrays(part['path'], part['manifest_sha256'])
            if not np.allclose(np.diff(raw['timesteps'])*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Non-native 0.1 ps source: '+item['key'])
        original.birth_source(recipe, p, item)
        folder = cache(recipe)/'births'/item['key']
        if sha(folder/'data.pt') != read(folder/'complete.json')['sha256']:
            raise ValueError('Unsealed additional birth observations')
        packets.append(torch.load(folder/'data.pt', map_location='cpu', weights_only=False))
        print('birth source', item['key'], len(packets[-1]['rows']), flush=True)
    rows = [dict(r) for packet in packets for r in packet['rows']]
    state = torch.cat([packet['state'] for packet in packets])/p['original_config']['length_factor']
    state -= state[:, :1].clone()  # Native Angstrom, Angstrom/ps; center-relative Q AND V.
    seen = {}; keep = []
    source_roles = {s['key']: s['original_role'] for s in p['sources']}
    catalogue_roles = {s['key']: s['catalogue_role'] for s in p['sources']}
    for i, r in enumerate(rows):
        r['original_role'] = source_roles[r['source']]
        r['catalogue_role'] = catalogue_roles[r['source']]
        r['role'] = train_test_role(r['original_role'])
        if r['ancestor'] not in p['partition'][r['role']]:
            raise ValueError('Birth / pretrained-model ancestry conflict')
        key = (r['task'], r['lead'], r['id'])
        if key in seen:
            if rows[seen[key]]['label'] != r['label']:
                raise ValueError('Conflicting duplicate birth label')
            continue
        seen[key] = i; keep.append(i)
    rows = [rows[i] for i in keep]; state = state[keep]
    counts = Counter((r['task'], r['lead'], r['event']) for r in rows)
    for r in rows:
        r['weight'] = 1/counts[r['task'], r['lead'], r['event']]
    # Repeated snapshots across tasks/leads are computed only once, not counted
    # as independent physical samples. Classification rows retain their task.
    unique = {}; first = []; indices = []
    for i, r in enumerate(rows):
        if r['id'] not in unique:
            unique[r['id']] = len(first); first.append(i)
        indices.append(unique[r['id']])
    data = dict(state=state[first], indices=np.asarray(indices), rows=rows,
                excluded=[e for packet in packets for e in packet['excluded']])
    if not torch.isfinite(data['state']).all():
        raise FloatingPointError('Nonfinite birth input')
    atomic_save(path, data)
    write_json(cache(c)/'cohort.json', dict(identity=p['identity'], sha256=sha(path)))
    coverage = []
    for task in c['tasks']:
        for lead in c['leads_ps']:
            for role in ['train', 'test']:
                selected = [r for r in rows if r['task'] == task and r['lead'] == lead and r['role'] == role]
                coverage.append(dict(task=task, lead_ps=lead, role=role, rows=len(selected),
                    positive_rows=sum(r['label'] for r in selected),
                    events=len({r['event'] for r in selected}), ancestors=len({r['ancestor'] for r in selected})))
    write_json(root(c)/'technical/cohort.json', dict(coverage=coverage, rows=len(rows),
        unique_states=len(first), excluded=data['excluded'], sha256=sha(path),
        sampled_sources=len(p['sources']), sources_with_events=len({r['source'] for r in rows})))
    return data
