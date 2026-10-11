"""All available exact 0.10 ps views, saved-velocity eligibility and event aliases."""
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import numpy as np
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from .data import read, cache, output


def eligibility(c,s):
    if '010' not in s['views']:return 'outside_010_label_grid'
    if c.get('population')!='all_010_with_saved_velocities':
        if not s['dataset'].startswith('al-main-010ps-20260927-source'):return 'other_population'
        if s['origin']!='original':return 'historical_partial_or_duplicate_view'
        if s['role'] not in ['train','selection','calibration','test']:return 'no_fixed_role'
    for part in s['parts']:
        m=read(Path(part['path'])/'manifest.json')
        if 'velocities' not in m['arrays']:return 'no_saved_velocities'
        if m.get('velocity_units')!='angstrom_per_ps':raise ValueError('Unsupported saved velocity units: '+part['path'])
    return None


def bind_reuse(c,sources):
    if not c.get('reuse_prepared'):return None
    root=resolve_path(c['reuse_prepared']);old=read(root/'technical/plan.json')
    for key in ['campaign_identity','fixed_identity','seed','leads_ps','radius_A','control_followup_ps','centers_per_event','negative_candidates']:
        if c[key]!=old['config'][key]:raise ValueError('Prepared-source observation protocol changed: '+key)
    oldcache=resolve_path(old['config']['cache']);lookup={s['key']:s for s in old['sources']}
    for item in sources:
        previous=lookup.get(item['key']);folder=oldcache/'sources'/item['key'];receipt=folder/'complete.json'
        if previous is None or not receipt.exists():continue
        if item['source']!=previous['source'] or item['label_identity']!=previous['label_identity']:raise ValueError('Prepared source mapping differs')
        done=read(receipt)
        if done['identity']!=digest(dict(plan=old['identity'],source=item['key'])):raise ValueError('Wrong prepared source producer')
        item['reuse']=dict(folder=str(folder),receipt_sha256=sha(receipt),identity=done['identity'])
    return dict(plan=str(root/'technical/plan.json'),plan_sha256=sha(root/'technical/plan.json'),plan_identity=old['identity'])


def canonicalize(c,p,item,folder,access,rows,coverage,base):
    """Merge physical event identity while retaining every source-view receipt."""
    s=item['source'];events={};view_aliases={}
    positions=np.load(folder/'positions.npy',mmap_mode='r');velocities=np.load(folder/'velocities.npy',mmap_mode='r')
    for row in rows:
        view=row['event'];f=int(row['first_frame'])
        if view not in view_aliases:
            q,b,_,_=access.frame(f);pi,rf=s['frames'][f]
            v=np.asarray(access.velocities[pi][rf],np.float32)/np.float32(1000.)
            signature=dict(ancestor=s['ancestry_group'],time_ps=round(float(s['times_ps'][f]),8),
                positions=hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
                velocities=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest(),
                box=b.tolist(),first_members=base['event_members'][view])
            physical='birth-'+digest(signature)[:32];view_aliases[view]=physical;events[physical]=signature
        row.update(event=view_aliases[view],event_view=view,dataset=s['dataset'],source_origin=s['origin'],native_cadence_ps=s['native_cadence_ps'],label_cadence_ps=.1)
        row['input_hash']=hashlib.sha256(positions[row['indices']].tobytes()+velocities[row['indices']].tobytes()).hexdigest()
    for r in coverage:r['physical_event']=view_aliases.get(r['event'])
    write_json(folder/'rows.json',rows);write_json(folder/'coverage.json',coverage)
    write_json(folder/'event-identities.json',dict(views=view_aliases,physical=events))
    result={k:v for k,v in base.items() if k!='event_members'}
    result.update(identity=digest(dict(plan=p['identity'],source=item['key'])),source=s['id'],role=s['role'],ancestor=s['ancestry_group'],
        dataset=s['dataset'],source_origin=s['origin'],native_cadence_ps=s['native_cadence_ps'],label_cadence_ps=.1,
        rows=len(rows),accepted_events=len(view_aliases),physical_events=len(events),
        files={n:sha(folder/n) for n in ['positions.npy','velocities.npy','descriptors.npy','rows.json','coverage.json','event-identities.json']})
    write_json(folder/'complete.json',result);return result


def reuse(c,p,item,folder,access,events):
    spec=item['reuse'];old=Path(spec['folder']);receipt=old/'complete.json'
    if sha(receipt)!=spec['receipt_sha256']:raise ValueError('Changed prepared receipt')
    done=read(receipt)
    if any(sha(old/n)!=h for n,h in done['files'].items()):raise ValueError('Changed prepared input bytes')
    for name in ['positions.npy','velocities.npy','descriptors.npy']:
        target=folder/name
        if not target.exists():os.link(old/name,target)
        elif sha(target)!=done['files'][name]:raise ValueError('Changed reused input '+name)
    rows=read(old/'rows.json');coverage=read(old/'coverage.json')
    base={k:done[k] for k in ['candidate_events','observations','labels_sha256','velocity_manifest','velocity_provenance']}
    base['reused']=spec;base['event_members']={item['key']+':'+str(e['track']):e['first_members'] for e in events}
    return canonicalize(c,p,item,folder,access,rows,coverage,base)


def deduplicate(rows):
    pairs=defaultdict(list)
    for row in rows:pairs[row['pair']].append(row)
    retained={};aliases=[];result=[]
    for pair,members in pairs.items():
        positives=[r for r in members if r['label']==1]
        if len(positives)!=1 or len(members)!=5:raise ValueError('Malformed matched pair')
        case=positives[0];key=(case['event'],int(case['atom']),case['input_hash'])
        if key in retained:
            prior=retained[key]
            # Equal full-cell onset Q,V plus same event membership/time establishes
            # a physical event alias. Distinct local prehistories remain in scope.
            if case['input_hash']==prior['input_hash']:
                aliases.append(dict(event=case['event'],atom=case['atom'],dropped_pair=pair,retained_pair=prior['pair'],dropped_view=case['event_view'],reason='same physical event, center and all three Q/V input patches'))
                continue
        retained[key]=case;result.extend(members)
    return result,aliases
