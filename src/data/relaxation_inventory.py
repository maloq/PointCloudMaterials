"""Backfill retained relaxation products and plan explicit missing requests."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from src.data import relaxation_index as index
from src.data.relaxation_index import read, sha, digest, write_json
from src.project_runtime.paths import load_json, resolve_path, storage_path


def entry_id(path, product):
    return digest([str(Path(path).resolve()), product])


def engine_evidence(item):
    if item.get('plan'):
        path = resolve_path(item['plan']); p = read(path)
        return dict(binary=p['lammps'], sha256=p['lammps_sha256'],
                    path=str(path), file_sha256=sha(path))
    if item.get('implementation'):
        path = resolve_path(item['implementation']); p = read(path)
        return dict(binary=index.engine_command(p['relaxation'])[0],
                    sha256=p['implementation']['lammps_binary'], path=str(path), file_sha256=sha(path))
    return None


def historical_recipe(meta, evidence):
    if 'index_recipe' in meta:
        spec = meta['index_recipe']
        if spec != index.recipe(meta['settings'], meta['potential_checksums'], spec['engine_sha256']):
            raise ValueError(f'Recorded recipe differs from relaxation settings: {meta["source"]}')
        return spec
    accelerator = meta['settings'].get('accelerator')
    if accelerator:
        engine = accelerator['binary_sha256']
    elif evidence is not None:
        binary = index.engine_command(meta['settings'])[0]
        if Path(binary).resolve() != Path(evidence['binary']).resolve():
            raise ValueError(f'Engine evidence does not describe {binary}: {evidence["path"]}')
        engine = evidence['sha256']
    else:
        return None  # Never infer a historical binary hash from today's executable.
    return index.recipe(meta['settings'], meta['potential_checksums'], engine)


def import_archive(path, evidence=None, collection=None):
    path = Path(path).resolve()
    receipt = path/'archive_checksums.json'
    if not receipt.is_file():
        return []  # In-progress/failure directories are not ready products.
    meta = read(path/'metadata.json'); index.validate_metadata(meta, path)
    checks = read(receipt)
    metadata_hash = sha(path/'metadata.json')
    if metadata_hash != checks['metadata.json']:
        raise ValueError(f'Metadata differs from archive receipt: {path}')
    spec = historical_recipe(meta, evidence)
    files = {str(path/name):value for name,value in checks.items()}
    files[str(receipt)] = sha(receipt)
    entry = dict(id=entry_id(path, 'cell'), path=str(path), metadata=meta, recipe=spec,
        metadata_sha256=metadata_hash, precision='archived_float16_cell',
        payload=str(path/'relaxed_binary_float16'), files=files,
        collection=str(path.parent.relative_to(storage_path('archive'))),
        engine_evidence=None if 'index_recipe' in meta or 'accelerator' in meta['settings'] else evidence,
        verification='metadata_and_receipt_at_registration; all_payload_hashes_on_use')
    binary = path/'relaxed_binary_float16'
    # Conversion is required: a converged metadata file alone is insufficient.
    if (not (binary/'manifest.json').is_file() or not (path/'conversion.json').is_file()
            or read(path/'conversion.json')['state'] != 'complete'):
        return []
    if str(binary/'positions.npy') not in files:
        raise ValueError(f'No position checksum in relaxation archive: {path}')
    if sha(binary/'manifest.json') != files[str(binary/'manifest.json')]:
        raise ValueError(f'Changed archived trajectory manifest: {path}')
    data = np.load(binary/'positions.npy', mmap_mode='r', allow_pickle=False)
    if data.dtype != np.float16 or data.shape != (1, meta['atom_count'], 3):
        raise ValueError(f'Invalid archived cell schema: {path}')
    if not index.available(entry):
        raise FileNotFoundError(f'Incomplete sealed archive: {path}')
    index.register(entry)
    entries = [entry]
    if (path/'patches.npz').is_file():
        entries.append(register_patches(entry, path/'patches.npz', 'positions', 'center_atom_ids',
                                       checks['patches.npz']))
    return entries


def register_patches(cell, payload, position_field, center_field, expected_hash, receipt=None):
    payload = Path(payload).resolve()
    if sha(payload) != expected_hash:
        raise ValueError(f'Changed saved relaxed neighborhoods: {payload}')
    entry = dict(cell, id=entry_id(payload, 'patches'), precision='saved_float32_patches',
        payload=str(payload), position_field=position_field, center_field=center_field,
        files=dict(cell['files'], **{str(payload):expected_hash}))
    if receipt is not None:
        entry['files'][str(receipt)] = sha(receipt)
    ids, neighbors, _ = index._patch_values(entry)
    entry.update(patches=len(ids), neighbors_per_patch=neighbors.shape[1])
    return index.register(entry)


def import_cache(path, layout):
    """Adapters for the actual two historical producers, not guessed schemas."""
    receipt = read(path)
    if layout == 'paired_clouds':
        archive = Path(receipt['archive'])
        payload = path.with_name('clouds.npz')
        position, center, checksum = 'cold', 'query_atom_ids', receipt['clouds_sha256']
    elif layout == 'tda_targets':
        archive = Path(receipt['relaxation_archive'])
        payload = Path(receipt['targets'])
        position, center, checksum = 'relaxed_clouds', 'center_atom_ids', receipt['targets_sha256']
    else:
        raise ValueError(f'Unknown relaxed-cache producer: {layout}')
    # Receipts normally contain live absolute paths. Resolve declared machine
    # aliases only for moved data; rebuilding the whole catalog per live path
    # makes a large backfill unnecessarily expensive.
    if not archive.is_dir():
        archive = resolve_path(archive)
    if not payload.is_file():
        payload = resolve_path(payload)
    cell_path = index.root()/'entries'/(entry_id(archive, 'cell')+'.json')
    if not cell_path.is_file() or not payload.is_file():
        return None
    cell = read(cell_path)
    if receipt['relaxation']['input_sha256'] != cell['metadata']['input_sha256']:
        raise ValueError(f'Local patches reference a different full cell: {path}')
    return register_patches(cell, payload, position, center, checksum, receipt=path.resolve())


def scan(config=index.CONFIG):
    c = load_json(config); counts = Counter(); roots = []; errors = []
    with index.lock('scan'):
        for item in c['archives']:
            directory = resolve_path(item['path']); evidence = engine_evidence(item)
            cells = sorted(directory.glob('*/metadata.json')) if directory.is_dir() else []
            record = dict(collection=item['id'], path=str(directory), available=directory.is_dir(), cells=len(cells))
            roots.append(record)
            def import_one(path):
                try:
                    return import_archive(path.parent, evidence, item['id']), None
                except (ValueError, KeyError, FileNotFoundError) as e:
                    return [], dict(path=str(path), error=str(e))
            with ThreadPoolExecutor(max_workers=8) as workers:
                for i, (entries, error) in enumerate(workers.map(import_one, cells)):
                    if error: errors.append(error)
                    elif not entries: counts['unsealed_or_unconverted'] += 1
                    for entry in entries: counts[entry['precision']] += 1
                    if (i+1) % 250 == 0:
                        print(json.dumps(dict(collection=item['id'], completed=i+1, total=len(cells))), flush=True)
            print(json.dumps(record), flush=True)
        for item in c.get('patch_caches', []):
            directory = resolve_path(item['path'])
            receipts = sorted(directory.glob('*/complete.json')) if directory.is_dir() else []
            record = dict(collection=item['id'], path=str(directory), available=directory.is_dir(), receipts=len(receipts))
            roots.append(record)
            def import_one_patch(path):
                try:
                    return import_cache(path, item['layout']), None
                except (ValueError, KeyError, FileNotFoundError) as e:
                    return None, dict(path=str(path), error=str(e))
            with ThreadPoolExecutor(max_workers=8) as workers:
                for entry, error in workers.map(import_one_patch, receipts):
                    if error: errors.append(error)
                    else: counts['external_patch_artifacts' if entry else 'unavailable_patch_reference'] += 1
            print(json.dumps(record), flush=True)
        write_json(index.root()/'scan.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            counts=dict(counts), roots=roots, errors=errors, config_sha256=sha(resolve_path(config))))
        result = status(export=True)
    if errors:
        raise ValueError(f'{len(errors)} invalid relaxation records; details in {index.root()/"scan.json"}')
    return result


def status(*, export=False):
    with ThreadPoolExecutor(max_workers=8) as workers:
        entries = list(workers.map(read, sorted((index.root()/'entries').glob('*.json'))))
        presence = list(workers.map(index.available, entries))
    counts = Counter(); collections = {}; sources=set(); cells=set(); profiles={}; rows=[]
    for entry, present in zip(entries, presence, strict=True):
        m=entry['metadata']; precision=entry['precision']
        counts['entries'] += 1
        counts[precision] += 1
        counts['available' if present else 'unavailable'] += 1
        counts['recipe_bound' if entry['recipe'] else 'recipe_unverified'] += 1
        if present:
            sources.add(m['source_manifest_sha256']); cells.add(index.source_key(m))
        profile = digest(entry['recipe']) if entry['recipe'] else 'unknown'
        if entry['recipe']:
            profiles[profile] = entry['recipe']
        collection=entry.get('collection', 'shared_native')
        summary=collections.setdefault(collection, Counter())
        summary[precision]+=1; summary['available' if present else 'unavailable']+=1
        if precision=='saved_float32_patches':
            summary['saved_patch_rows'] += entry['patches']
        rows.append(dict(id=entry['id'], collection=collection, precision=precision,
            available=present, recipe_id=profile, source=m['source'],
            source_manifest_sha256=m['source_manifest_sha256'], frame=m['source_frame'],
            timestep=m['source_timestep'], atom_count=m['atom_count'],
            fmax_eV_per_A=m['fmax_eV_per_A'], patches=entry.get('patches',0), payload=entry['payload']))
    result = dict(schema_version=1, measured_at=datetime.now(timezone.utc).isoformat(),
        index_root=str(index.root()), counts=dict(counts), unique_sources=len(sources),
        unique_source_frames=len(cells), recipe_profiles=len(profiles),
        collections={k:dict(v) for k,v in sorted(collections.items())},
        note='Available means files exist; every selected payload is checksum-verified before reuse. Patch rows can repeat across releases.')
    if export:
        write_json(index.root()/'inventory.json', result)
        write_json(index.root()/'recipes.json', profiles)
        if rows:
            with (index.root()/'inventory.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    return result


def plan_requests(path, output):
    """JSON request contract: settings, requests with source identity, product.

Patch requests additionally provide exact center_atom_ids/neighbor_atom_ids.
The result keeps all rows, including unknown precision or recipe mismatches.
"""
    c=load_json(path); spec=index.recipe(c['settings']); rows=[]
    for request in c['requests']:
        product=request['product']; all_entries=index.candidates(request)
        matching=[e for e in all_entries if e['recipe']==spec and index.available(e)]
        selected=None; state='compute'; reason='no_compatible_saved_product'
        if product=='saved_float32_patches':
            hit=index.patch_lookup(request,c['settings'],request['center_atom_ids'],request['neighbor_atom_ids'])
            if hit is not None:
                _,selected=hit;state='reuse';reason='exact_saved_membership'
            else:
                selected=next((e for e in matching if e['precision']=='native_restart'),None)
                if selected:
                    index.verify(selected);state='extract';reason='native_checkpoint_no_minimization'
        elif product=='archived_float16_cell':
            selected=index.cell_lookup(request,c['settings'],precision=product)
            if selected:
                state='reuse';reason='declared_float16_product'
            else:
                selected=next((e for e in matching if e['precision']=='native_restart'),None)
                if selected:
                    index.verify(selected);state='extract';reason='native_checkpoint_no_minimization'
        else:
            raise ValueError(f'Unknown requested product: {product}')
        rows.append(dict(request=request, action=state, reason=reason,
            entry_id=selected['id'] if selected else None,
            other_saved_entries=[dict(id=e['id'], precision=e['precision'], available=index.available(e),
                recipe_matches=e['recipe']==spec) for e in all_entries]))
    result=dict(schema_version=1, request_sha256=sha(resolve_path(path)), recipe=spec,
                counts=dict(Counter(r['action'] for r in rows)), requests=rows,
                missing=[r['request'] for r in rows if r['action']=='compute'])
    write_json(output,result)
    return dict(output=str(output), counts=result['counts'])


def add_arguments(parser):
    parser.add_argument('action', choices=['scan','status','plan'])
    parser.add_argument('--config',default=index.CONFIG)
    parser.add_argument('--requests',type=Path)
    parser.add_argument('--output',type=Path)


def dispatch(args):
    if args.action=='scan':return scan(args.config)
    if args.action=='status':return status(export=True)
    if args.requests is None or args.output is None:
        raise ValueError('plan requires --requests and --output')
    return plan_requests(args.requests,args.output)


def main():
    parser=argparse.ArgumentParser(__doc__);add_arguments(parser)
    print(json.dumps(dispatch(parser.parse_args()),indent=2))


if __name__=='__main__':main()
