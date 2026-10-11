"""Freeze available paired relaxed observations, preserving producer ancestry.

The unrelaxed member is an availability/identity requirement, not a model input.
Retained exact local pairs qualify even when their old full trajectory is gone.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.data import relaxation_index as index
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path

read = index.read


def role(original):
    return {'train': 'train', 'selection': 'evaluation', 'test': 'evaluation',
            'calibration': 'calibration', 'heldout_or_diagnostic': 'diagnostic'}[original]


def load_sources(c):
    fixed = read(resolve_path(c['fixed_dataset']['root'])/'plan.json')
    tda = read(resolve_path(c['ancestry_plan']))
    sources = {}
    for s in tda['sources']:
        original = 'train' if s['training_eligible'] else 'heldout_or_diagnostic'
        sources[s['manifest_sha256']] = dict(source=s['id'], lineage=s['lineage'],
            original_role=original, declared_role=s['declared_split'], role=role(original),
            material='Al', potential='al-lee2003-meam', path=s['trajectory'],
            manifest_sha256=s['manifest_sha256'])
    for s in fixed['sources']:
        h = s['manifest_sha256']
        old = sources.get(h, {})
        sources[h] = dict(source=str(s['id']), lineage=s['lineage'],
            original_role=s['role'], declared_role=s['role'], role=role(s['role']),
            material='Al', potential='al-lee2003-meam', path=old.get('path'), manifest_sha256=h)
    return sources, fixed


@lru_cache(maxsize=1024)
def original_source(path, expected):
    path = resolve_path(path)
    manifest = path/'manifest.json'
    if not manifest.is_file():
        return None
    m = read(manifest)
    names = ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high')
    if any(not (path/m['arrays'][n]['file']).is_file() for n in names):
        return None
    if sha(manifest) != expected:
        raise ValueError(f'Changed available original: {manifest}')
    return dict(path=str(path), manifest_sha256=expected, kind='dynamic')


def pair_arrays(t):
    """Read the actual three retained pair formats; units are physical Angstrom."""
    if t['kind'] == 'index_patch':
        e = read(t['entry'])
        ids, neighbors, cold = index._patch_values(e)
        hot_name = {'cold': 'hot', 'relaxed_clouds': 'observed_clouds'}.get(e['position_field'])
        with np.load(e['payload']) as z:
            hot = z[hot_name].reshape(cold.shape) if hot_name else None
        return ids, neighbors, hot, cold
    path = Path(t['pair_path'])
    names = ('center_ids', 'neighbor_ids', 'hot', 'cold') if t['kind'] == 'native_pair' else (
        'atom_ids', 'neighbor_ids', 'hot', 'cold')
    ids, near, hot, cold = [np.load(path/(n+'.npy'), mmap_mode='r') for n in names]
    if t['kind'] == 'fixed_pair':
        near, hot, cold = (a[t['frame_index']] for a in (near, hot, cold))
    return ids, near, hot, cold


def checked_frame(original, meta):
    from src.data.structural_pretraining.prepare import source_arrays
    arrays = source_arrays(original)
    frame = meta['source_frame']
    if int(arrays['timesteps'][frame]) != meta['source_timestep']:
        raise ValueError(f'Original frame identity changed: {original["path"]}/{frame}')
    if len(arrays['atom_ids']) != meta['atom_count']:
        raise ValueError('Original atom count differs from relaxation')
    for name in ('box_low', 'box_high'):
        np.testing.assert_array_equal(arrays[name][frame], meta[name])
    return arrays


def static_tasks(c, parent, parent_plan, parent_manifest):
    """User-confirmed inherent configurations; preserve existing native-unit patches."""
    from src.data.structural_pretraining.prepare import source_arrays
    cached={str(resolve_path(s['path']).resolve()):s for s in parent_plan['sources'] if s['kind']=='static'}
    tasks=[];summary=Counter()
    for spec in c['static_sources']:
        path=resolve_path(spec['path'])
        if sha(path)!=spec['file_sha256']:
            raise ValueError(f'Static inherent configuration changed: {path}')
        arrays=source_arrays(spec);x=arrays['positions']
        s=cached.get(str(path.resolve()))
        common=dict(source=spec['id'],lineage=spec['lineage'],role='train',original_role='train',declared_role='train',
            material=spec['material'],potential='unknown-static',source_manifest_sha256=spec['file_sha256'],
            frame=0,source_timestep=None,precision='static_float32',pairing='user_confirmed_static_exception',
            static_source=spec,relaxation_evidence='User explicitly confirmed Mg/Al/Zr/Ta static collections are already relaxed on 2026-10-08; minimizer/potential not recorded')
        if s is not None:
            if s['split']!='train' or s['lineage']!=spec['lineage'] or s['file_sha256']!=spec['file_sha256']:
                raise ValueError('Static cached ancestry or original identity changed')
            shards=[r for r in parent_manifest['shards'] if r['task']['source']==s['id']]
            for r in shards:
                folder=parent/'shards'/r['task']['id']
                if r['task']['paired'] or not r['task']['observed']:
                    raise ValueError('Unexpected static cache format')
                if any(not (folder/n).is_file() for n in ('hot.npy','neighbor_ids.npy','center_ids.npy')):
                    raise FileNotFoundError(f'Unavailable existing static cache: {folder}')
                tasks.append(dict(common,id='static-'+r['task']['id'],kind='static_cached',
                    pair_path=str(folder),input_files=r['files'],rows=r['rows']))
                summary[spec['material']]+=r['rows']
        else:
            # Static clouds have no supplied periodic box. Stay inside the same
            # native 8-A boundary convention used by the established producer.
            eligible=np.flatnonzero(((x>x.min(0)+8)&(x<x.max(0)-8)).all(-1))
            if len(eligible)<c['centers_per_source']:
                raise ValueError('Insufficient interior static centers: '+str(path))
            rng=np.random.default_rng(int(digest([c['seed'],spec['file_sha256'],'static-centers'])[:16],16))
            centers=np.sort(rng.choice(eligible,c['centers_per_source'],replace=False)).tolist()
            tasks.append(dict(common,id='static-'+spec['id'],kind='static_full',rows=len(centers),center_ids=centers))
            summary[spec['material']]+=len(centers)
    return tasks,dict(summary)


def plan(c):
    dest = resolve_path(c['cache']); dest.mkdir(parents=True, exist_ok=True)
    path = dest/'plan.json'
    if path.exists():
        p = read(path)
        if p['config'] != c:
            raise ValueError('Changed relaxed collection; use a new release')
        return p
    sources, fixed = load_sources(c)
    ni = read(resolve_path(c['ni_relaxation'])/'technical/plan.json')
    if ni['identity'] != c['ni_identity']:
        raise ValueError('Changed Ni relaxation population')
    ni_hashes = {part['manifest_sha256'] for s in ni['sources'].values() for part in s['parts']}
    entries = []; excluded = Counter(); raw = {}; by_archive = {}
    for f in sorted((index.root()/'entries').glob('*.json')):
        e = read(f); m = e['metadata']; h = m['source_manifest_sha256']
        if h in ni_hashes:
            continue  # all planned Ni frames are bound below, including pending cells
        if not index.available(e):
            excluded['indexed_product_unavailable'] += 1
            continue
        index.validate_metadata(m, e['path'])
        if h not in sources:
            raise ValueError(f'Unmapped relaxed-source ancestry: {h}: {m["source"]}')
        sources[h]['path'] = m['source']
        if h not in raw:
            raw[h] = original_source(m['source'], h)
        e['_entry_file'] = str(f); e['_entry_sha256'] = sha(f)
        entries.append(e)
        if e['precision'] == 'archived_float16_cell':
            by_archive[str(Path(e['path']).resolve())] = e
    candidates = []

    def common(h, frame):
        s = sources[h]
        return {**{k:s[k] for k in ('source','lineage','role','original_role','declared_role','material','potential')},
                'source_manifest_sha256': h, 'frame': int(frame)}

    def archive_fields(e):
        return dict(entry=e['_entry_file'], entry_sha256=e['_entry_sha256'],
                    relaxation_input_sha256=e['metadata']['input_sha256'],
                    source_timestep=e['metadata']['source_timestep'])

    # Highest fidelity observations are considered first during deduplication.
    for e in entries:
        if e['precision'] != 'saved_float32_patches':
            continue
        m = e['metadata']; h = m['source_manifest_sha256']
        if e['neighbors_per_patch'] != 80:
            raise ValueError('Expected the retained nearest80 patch contract')
        candidates.append(dict(common(h,m['source_frame']), **archive_fields(e), kind='index_patch',
            id='patch-'+e['id'], rank=0, precision='prequantization_float32',
            original=raw[h], pairing='embedded' if e['position_field'] in ('cold','relaxed_clouds') else 'lookup'))

    # Retained structural caches contain explicit paired observations, not just cold tensors.
    parent = resolve_path(c['paired_native']); manifest = read(parent/'manifest.json')
    parent_plan = read(parent/'plan.json'); native_sources = {s['id']:s for s in parent_plan['sources']}
    static,static_counts=static_tasks(c,parent,parent_plan,manifest)
    for shard in manifest['shards']:
        t = shard['task']
        if not t['paired']:
            continue
        s = native_sources[t['source']]; cell = next(v for v in s['cells'] if v['frame']==t['frame'])
        e = by_archive.get(str(resolve_path(cell['archive']).resolve()))
        folder = parent/'shards'/t['id']
        if e is None or any(not (folder/f).is_file() for f in ('hot.npy','cold.npy','neighbor_ids.npy','center_ids.npy')):
            excluded['paired_native_unavailable_or_removed_relaxation'] += shard['rows']
            continue
        if s['lineage'] != sources[s['manifest_sha256']]['lineage'] or s['split'] != sources[s['manifest_sha256']]['original_role']:
            raise ValueError('Native pair ancestry changed')
        candidates.append(dict(common(s['manifest_sha256'],t['frame']), **archive_fields(e),
            id='native-'+t['id'], kind='native_pair', pair_path=str(folder), input_files=shard['files'],
            rank=1, precision='float16_cell_derived_float32', pairing='embedded'))
    parent = resolve_path(c['fixed_dataset']['root']); manifest = read(parent/'benchmark/manifest.json')
    fixed_sources = {s['id']:s for s in fixed['sources']}
    for shard in manifest['sources']:
        s = fixed_sources[shard['id']]; folder = parent/'benchmark/sources'/str(s['id'])
        if any(not (folder/f).is_file() for f in ('hot.npy','cold.npy','neighbor_ids.npy','atom_ids.npy','frames.npy')):
            excluded['fixed_paired_source_unavailable'] += 1
            continue
        frames = np.load(folder/'frames.npy')
        for j, frame in enumerate(frames):
            cell = next(v for v in s['cells'] if v['frame']==int(frame))
            e = by_archive.get(str(resolve_path(cell['archive']).resolve()))
            if e is None:
                excluded['fixed_paired_frame_removed_relaxation'] += 1
                continue
            candidates.append(dict(common(s['manifest_sha256'],frame), **archive_fields(e),
                id=f'fixed-{s["id"]}-{frame}', kind='fixed_pair', pair_path=str(folder),
                input_files={k:v for k,v in shard['files'].items() if k in
                    ('hot.npy','cold.npy','neighbor_ids.npy','atom_ids.npy','frames.npy')}, frame_index=j,
                rank=2, precision='float16_cell_derived_float32', pairing='embedded'))

    # Index original observations by exact atom membership. This also recovers
    # originals for later relaxed-only patch caches without reviving raw data.
    originals = defaultdict(dict)
    for t in sorted(candidates,key=lambda v:(v['rank'],v['id'])):
        if t['pairing'] != 'embedded':
            continue
        ids, near, hot, cold = pair_arrays(t)
        if hot.shape != cold.shape or hot.shape != (len(ids),80,3) or not np.isfinite(hot).all() or np.any(hot[:,0]):
            raise ValueError(f'Invalid paired original {t["id"]}')
        key = (t['source_manifest_sha256'],t['frame'])
        for i, (center, members) in enumerate(zip(ids,near)):
            originals[key].setdefault((int(center),members.astype(np.int64).tobytes()),(t,i))

    tasks = static; seen = set(); covered_frames = set(); duplicates = 0
    for t in sorted(candidates,key=lambda v:(v['rank'],v['id'])):
        ids, near, hot, cold = pair_arrays(t)
        key = (t['source_manifest_sha256'],t['frame']); selected = []; paired_refs = []
        for i, center in enumerate(ids):
            observation = (*key,int(center))
            if observation in seen:
                duplicates += 1
                continue
            original = None
            if t['pairing']=='lookup' and t['original'] is None:
                original = originals[key].get((int(center),near[i].astype(np.int64).tobytes()))
                if original is None:
                    excluded['relaxed_patch_without_matching_original'] += 1
                    continue
                old, row = original
                paired_refs.append(dict(row=i, original_row=row, original_task=old))
            selected.append(i); seen.add(observation)
        if not selected:
            continue
        if t.get('original') is not None:
            checked_frame(t['original'],read(t['entry'])['metadata'])
        task = dict(t, selected_rows=selected, rows=len(selected), original_references=paired_refs)
        tasks.append(task); covered_frames.add(key)

    # Sample additional full cells only when their original coordinates exist
    # and that frame has no eligible cached neighborhoods.
    centers_by_source = {}
    for e in sorted(entries,key=lambda v:(v['precision']!='native_restart',v['id'])):
        if e['precision'] == 'saved_float32_patches':
            continue
        m=e['metadata'];h=m['source_manifest_sha256'];key=(h,m['source_frame'])
        if key in covered_frames:
            continue
        if raw[h] is None:
            excluded['relaxed_cell_without_original'] += 1
            continue
        arrays = checked_frame(raw[h],m)
        if h not in centers_by_source:
            rng=np.random.default_rng(int(digest([c['seed'],h,'fixed-centers'])[:16],16))
            centers_by_source[h]=np.sort(rng.choice(arrays['atom_ids'],c['centers_per_source'],replace=False)).tolist()
        tasks.append(dict(common(h,m['source_frame']),**archive_fields(e),id='cell-'+e['id'],
            kind='cell', original=raw[h], rows=c['centers_per_source'], center_ids=centers_by_source[h],
            precision=e['precision'], pairing='full_original'))
        covered_frames.add(key)

    # Freeze all 6,571 planned Ni cells, not only the subset finished today.
    for source in ni['sources'].values():
        first=source['parts'][0];raw_source=original_source(first['path'],first['manifest_sha256'])
        if raw_source is None:
            raise ValueError(f'Ni original unavailable: {source["id"]}')
        from src.data.structural_pretraining.prepare import source_arrays
        arrays=source_arrays(raw_source)
        rng=np.random.default_rng(int(digest([c['seed'],source['id'],'fixed-centers'])[:16],16))
        source_centers=np.sort(rng.choice(arrays['atom_ids'],c['centers_per_source'],replace=False)).tolist()
        for t in ni['tasks']:
            if t['source'] != source['id']:
                continue
            part=source['parts'][t['part']];original=original_source(part['path'],part['manifest_sha256'])
            if original is None:
                raise ValueError(f'Ni part unavailable: {part["path"]}')
            arrays=source_arrays(original)
            if int(arrays['timesteps'][t['frame']])!=t['source_timestep']:
                raise ValueError('Ni source timeline changed')
            observation=(part['manifest_sha256'],t['frame'])
            if observation in covered_frames:
                raise ValueError('Duplicate Ni frame')
            covered_frames.add(observation)
            tasks.append(dict(id='ni-'+t['id'],kind='ni',source=source['id'],lineage=source['ancestry_group'],
                original_role=source['role'],declared_role=source['role'],role=role(source['role']),
                material='Ni',potential='ni-mendelev2012-eam',frame=t['frame'],source_timestep=t['source_timestep'],
                source_manifest_sha256=part['manifest_sha256'], original=original,ni_task=t,
                rows=c['centers_per_source'],center_ids=source_centers,precision='native_restart',pairing='full_original'))
    train={t['lineage'] for t in tasks if t['role']=='train'}
    held={t['lineage'] for t in tasks if t['role']!='train'}
    if train&held:
        raise ValueError(f'Ancestry overlap: {train&held}')
    reference=read(resolve_path(c['reference_config']))
    if sha(resolve_path(c['reference_config']))!=c['reference_sha256']:
        raise ValueError('Small eSEN reference recipe changed')
    column_plan=read(resolve_path(reference['cache'])/'plan.json')
    from src.research.liquid_predictability.descriptors import descriptor_options
    if [v['name'] for v in column_plan['columns']]!=descriptor_options(c)['selected_columns']:
        raise ValueError('Target columns differ from last Small eSEN')
    counts=Counter();precision=Counter()
    for t in tasks:
        counts[t['role']+'/'+t['material']]+=t['rows'];precision[t['precision']]+=t['rows']
    p=dict(protocol=c['protocol'],config=c,tasks=tasks,columns=column_plan['columns'],
        normalization=reference['structural_dataset']['normalization'],counts=dict(counts),
        precision_rows=dict(precision),excluded=dict(excluded),duplicate_rows_removed=duplicates,
        ni_identity=ni['identity'],ancestry_overlap=0,descriptors=c['descriptors'],
        static_rows=static_counts,
        input_policy='Dynamic relaxations require exact retained unrelaxed originals, audit-only; user-confirmed already-relaxed static Mg/Al/Zr/Ta are an explicit exception.',
        deduplication='One source-manifest/frame/center; prefer prequantization float32 cached patches, then retained derived pairs; stable producer ID breaks ties.',
        membership='Preserve cached memberships; new cells use original nearest80 atom IDs in both hot and relaxed patches.',
        implementation={Path(__file__).name:sha(Path(__file__))})
    p['identity']=digest(p);write_json(path,p)
    summary={k:p[k] for k in ('identity','counts','precision_rows','excluded','duplicate_rows_removed','ni_identity','ancestry_overlap','static_rows')}
    summary.update(tasks=len(tasks),ni_frames=sum(t['kind']=='ni' for t in tasks),
        available_al_frames=len({(t['source_manifest_sha256'],t['frame']) for t in tasks if t['material']=='Al'}))
    write_json(resolve_path(c['output'])/'technical/paired-inventory.json',summary)
    return p
