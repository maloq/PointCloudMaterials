"""Replay the recorded neighborhood veto and quantify the components responsible."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import recorded_stage
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import Access, cache, read
from .pooled import FAMILY, load, root


def source_audit(c, p, item):
    s = item['source']; pc = p['config']; folder = Path(item['label_folder'])
    receipt = read(folder/'complete.json')
    for name in ['events.json', 'sizes.jsonl.gz']:
        if sha(folder/name) != receipt['files'][name]:
            raise ValueError('Changed lineage evidence: '+str(folder/name))
    all_events = read(folder/'events.json'); families = {e['track']:e for e in all_events}
    events = [e for e in all_events if e['record_kind']=='root' and e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None]
    if not events:
        return dict(events=[], centers=[], contacts=[], evidence=dict(source=s['id'],
                    lineage_receipt_sha256=sha(folder/'complete.json'), input_identity=s['input_identity']))
    access = Access(read(item['parent_path']), s)
    atom_ids = access.arrays['atom_ids']; view = np.asarray(s['views']['010'])
    steps = np.rint(np.asarray(pc['leads_ps'])/.1).astype(int)
    observed = {int(view[np.searchsorted(view, e['first_frame'])-step]) for e in events for step in steps
                if np.searchsorted(view, e['first_frame']) >= steps.max()}
    lineage = {}
    with gzip.open(folder/'sizes.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['frame'] in observed:
                lineage[(r['frame'], r['largest_component'])] = r['family']
    coverage = {r['event']:r for r in read(cache(pc)/'sources'/item['key']/'coverage.json')}
    rng = np.random.default_rng(int(digest([pc['seed'], item['key']])[:16], 16))
    centers, contacts, results = [], [], []
    for event in events:
        first = int(event['first_frame']); slot = int(np.searchsorted(view, first))
        event_view = item['key']+':'+str(event['track']); saved = coverage[event_view]
        members = np.searchsorted(atom_ids, event['first_members'])
        np.testing.assert_array_equal(atom_ids[members], event['first_members'])
        q, box, _, _ = access.frame(first); part, raw_frame = s['frames'][first]
        v = np.asarray(access.velocities[part][raw_frame], np.float32)/np.float32(1000.)
        signature = dict(ancestor=s['ancestry_group'], time_ps=round(float(s['times_ps'][first]), 8),
                         positions=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
                         velocities=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
                         box=box.tolist(), first_members=event['first_members'])
        physical = 'birth-'+digest(signature)[:32]
        if saved['physical_event'] is not None and saved['physical_event'] != physical:
            raise ValueError('Physical event identity changed')
        base = dict(event=physical, event_view=event_view, source=s['id'], dataset=s['dataset'],
                    original_role=s['role'], pooled_role='train' if s['role']=='train' else 'test',
                    ancestor=s['ancestry_group'], onset_ps=event['first_time_ps'])
        if slot-steps.max()<0 or slot+int(round(pc['control_followup_ps']/.1))>=len(view):
            raise ValueError('Unexpected timeline rejection in the pinned completed cohort')
        sampled = rng.permutation(members)[:pc['centers_per_event']]
        # The original sampler draws its control candidates even for vetoed events.
        rng.choice(s['atom_count'], min(pc['negative_candidates'], s['atom_count']), replace=False)
        event_centers, event_contacts = [], []
        clear = np.ones(len(sampled), bool)
        all_clear = np.ones(len(members), bool)
        for lead, frame in zip(pc['leads_ps'], view[slot-steps]):
            frame = int(frame); q, box, lab, _ = access.frame(frame)
            sizes = np.bincount(lab); sizes[0] = 0
            tree = cKDTree(q, boxsize=box)
            balls = tree.query_ball_point(q[members], r=pc['radius_A'], workers=1)
            by_member = {}
            for mi, (atom, ball) in enumerate(zip(members, balls)):
                indices = np.asarray(ball, int)
                active = np.unique(lab[indices]); active = active[(active>0)&(sizes[active]>=8)]
                by_member[int(atom)] = active
                all_clear[mi] &= len(active)==0
            for j, atom in enumerate(sampled):
                active = by_member[int(atom)]; blocked = len(active)>0
                clear[j] &= not blocked
                r = dict(base, lead_ps=lead, frame=frame, atom=int(atom_ids[atom]), blocked=blocked,
                         nearby_components=len(active), maximum_component_atoms=int(sizes[active].max()) if blocked else 0)
                selected_contacts = []
                for cid in active:
                    inds = np.flatnonzero(lab==cid)
                    delta = q[inds]-q[atom]; delta -= box*np.rint(delta/box)
                    distances = np.linalg.norm(delta, axis=1)
                    overlap = int(np.count_nonzero(lab[members]==cid))
                    family = lineage.get((frame, int(cid)))
                    fe = families[family] if family is not None else None
                    t = s['times_ps'][frame]
                    established = None if fe is None else bool(
                        (fe['establishment_confirmation_ps'] is not None and fe['establishment_confirmation_ps']<=t+1e-8)
                        or (fe['origin']=='parent_present' and len(fe['first_members'])>=64)
                        or fe['inherited_established_crystal'])
                    contact = dict(base, lead_ps=lead, frame=frame, atom=int(atom_ids[atom]), component=int(cid),
                                   component_atoms=int(sizes[cid]), nearest_atom_distance_A=float(distances.min()),
                                   component_atoms_in_8A=int(np.count_nonzero(distances<=pc['radius_A'])),
                                   center_is_component_member=bool(lab[atom]==cid), shared_onset_atoms=overlap,
                                   onset_atom_count=len(members), disjoint_from_onset_atoms=overlap==0,
                                   family=family, target_family=event['track'], established_at_observation=established)
                    selected_contacts.append(contact); event_contacts.append(contact)
                r.update(disjoint_component_nearby=any(x['disjoint_from_onset_atoms'] for x in selected_contacts),
                         shared_onset_component_nearby=any(x['shared_onset_atoms']>0 for x in selected_contacts),
                         component64_nearby=any(x['component_atoms']>=64 for x in selected_contacts),
                         established_component_nearby=any(x['established_at_observation'] is True for x in selected_contacts),
                         nearest_blocking_atom_A=min((x['nearest_atom_distance_A'] for x in selected_contacts), default=None))
                event_centers.append(r)
        if int(clear.sum()) != saved['eligible_centers']:
            raise ValueError(f'Cannot reproduce recorded gate: {event_view}, {clear.sum()} versus {saved["eligible_centers"]}')
        results.append(dict(base, accepted_pairs=saved['accepted_pairs'], rejected=saved['accepted_pairs']==0,
                            sampled_centers=len(sampled), eligible_sampled_centers=int(clear.sum()),
                            eligible_all_onset_centers=int(all_clear.sum()), onset_atoms=len(members),
                            any_disjoint_component=any(x['disjoint_from_onset_atoms'] for x in event_contacts),
                            any_shared_onset_component=any(x['shared_onset_atoms']>0 for x in event_contacts),
                            any_component64=any(x['component_atoms']>=64 for x in event_contacts),
                            any_established_component=any(x['established_at_observation'] is True for x in event_contacts),
                            maximum_component_atoms=max((x['component_atoms'] for x in event_contacts), default=0),
                            minimum_distance_A=min((x['nearest_atom_distance_A'] for x in event_contacts), default=None),
                            onset_nearest_established_crystal_A=event['nearest_established_crystal_A']))
        centers.extend(event_centers); contacts.extend(event_contacts)
    return dict(events=results, centers=centers, contacts=contacts,
                evidence=dict(source=s['id'], lineage_receipt_sha256=sha(folder/'complete.json'), input_identity=s['input_identity']))


def audit(c):
    p, _ = load(c); out = root(c)/'analyses/neighborhood-audit-v1'
    tech = out/'technical'; tech.mkdir(parents=True, exist_ok=True)
    records = dict(events=[], centers=[], contacts=[]); evidence=[]
    with recorded_stage(root(c)/'technical/neighborhood-audit.json', total_sources=len(p['sources'])) as progress:
        for index, item in enumerate(p['sources']):
            saved = tech/'sources'/(item['key']+'.json')
            if saved.exists():
                packet = read(saved)
            else:
                packet = source_audit(c, p, item); write_json(saved, packet)
            for name in records:
                records[name].extend(packet[name])
            evidence.append(packet['evidence'])
            progress.update(completed_sources=index+1, event_views=len(records['events']))
            print('audit', index+1, '/', len(p['sources']), item['source']['id'], len(packet['events']), flush=True)
        # Retain every view to explain the original rejection count. Unique-event
        # summaries use the parent's order: full/native before partial aliases.
        first = {}
        for row in records['events']:
            first.setdefault(row['event'], row['event_view'])
        for name, values in records.items():
            for row in values:
                row['primary_event_view'] = first[row['event']]==row['event_view']
            write_metric_rows(values, out, family=FAMILY, name=name)
        summaries = []
        for scope in ['all', 'pooled_test', 'original_test']:
            for unit in ['event_views', 'unique_events']:
                ee = [r for r in records['events'] if scope=='all' or (scope=='pooled_test' and r['pooled_role']=='test') or (scope=='original_test' and r['original_role']=='test')]
                if unit=='unique_events':
                    ee = [r for r in ee if r['primary_event_view']]
                rejected = [r for r in ee if r['rejected']]
                evs = {r['event_view'] for r in rejected}
                cc = [r for r in records['contacts'] if r['event_view'] in evs]
                distances = [r['nearest_atom_distance_A'] for r in cc]
                sizes = [r['component_atoms'] for r in cc]
                summaries.append(dict(scope=scope, unit=unit, candidates=len(ee), rejected=len(rejected), accepted=len(ee)-len(rejected),
                    rejected_with_disjoint=sum(r['any_disjoint_component'] for r in rejected),
                    rejected_with_shared_onset=sum(r['any_shared_onset_component'] for r in rejected),
                    rejected_shared_only=sum(r['any_shared_onset_component'] and not r['any_disjoint_component'] for r in rejected),
                    rejected_with_component64=sum(r['any_component64'] for r in rejected),
                    rejected_with_established=sum(r['any_established_component'] for r in rejected),
                    rejected_recoverable_by_other_onset_center=sum(r['eligible_all_onset_centers']>0 for r in rejected),
                    blocking_contacts=len(cc), contact_size_min=min(sizes) if sizes else None,
                    contact_size_median=float(np.median(sizes)) if sizes else None, contact_size_max=max(sizes) if sizes else None,
                    contact_distance_min_A=min(distances) if distances else None,
                    contact_distance_median_A=float(np.median(distances)) if distances else None,
                    contact_distance_max_A=max(distances) if distances else None,
                    contacts_missing_family=sum(r['family'] is None for r in cc)))
        write_metric_rows(summaries, out, family=FAMILY, name='summary')
        leads = []
        for scope in ['all', 'pooled_test', 'original_test']:
            evs = {r['event_view'] for r in records['events'] if scope=='all' or (scope=='pooled_test' and r['pooled_role']=='test') or (scope=='original_test' and r['original_role']=='test')}
            for lead in p['config']['leads_ps']:
                cc = [r for r in records['centers'] if r['event_view'] in evs and r['lead_ps']==lead]
                by_event = defaultdict(list)
                for r in cc:
                    by_event[r['event_view']].append(r)
                leads.append(dict(scope=scope, lead_ps=lead, sampled_centers=len(cc), blocked_centers=sum(r['blocked'] for r in cc),
                                  candidate_event_views=len(by_event), all_sampled_centers_blocked=sum(all(r['blocked'] for r in rr) for rr in by_event.values()),
                                  any_sampled_center_clear=sum(any(not r['blocked'] for r in rr) for rr in by_event.values())))
        write_metric_rows(leads, out, family=FAMILY, name='by-lead')
        write_json(tech/'complete.json', dict(parent_identity=p['identity'], event_views=len(records['events']), physical_events=len(first), evidence=evidence))
        lines = ['# Neighborhood rejection audit', '',
                 'Exact replay of the original sampled centers and periodic 8 Å veto at all three leads. No new simulation or PTM classification.',
                 'A blocking component has at least 8 connected PTM-crystalline atoms. Sharing future onset-member IDs is a precursor-overlap diagnostic, not proof of a common lineage. Disjointness is not proof of causal interference.',
                 'A component of 64 atoms at one frame is distinguished from establishment confirmed by the lineage history. Only the largest component of each labelled family appears in the trace; unmatched fragments have unknown establishment status.', '',
                 '| Scope | Unit | Candidates | Rejected | Rejected with shared atoms | Rejected with disjoint component | Rejected with ≥64 | Rejected with confirmed/pre-existing crystal |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            lines.append('| '+ ' | '.join(str(r[k]) for k in ['scope','unit','candidates','rejected','rejected_with_shared_onset','rejected_with_disjoint','rejected_with_component64','rejected_with_established'])+' |')
        lines += ['', 'Counts for shared and disjoint components can overlap. Contact size/distance distributions count sampled center–component contacts, not independent crystals.',
                  '[Events](tables/events.csv) · [Component contacts](tables/contacts.csv) · [Lead-specific exclusions](tables/by-lead.csv) · [Definitions](tables/METRICS.md)']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
