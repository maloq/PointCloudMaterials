"""Evidence-backed dataset inventory without loading large coordinate arrays.

The location catalog remains the sole registration source. This module observes
current producer records, headers and filesystem metadata; it never changes data.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import numpy as np

from .paths import REPO, machine


TRAJECTORY_FORMATS = {'pointcloudmaterials.shooting_trajectory', 'pointcloudmaterials.temporal_lammps_trajectory'}
DOCUMENTS = {'manifest.json', 'metadata.json', 'config.json', 'protocol.json', 'outcome.json',
    'status.json', 'complete.json', 'dataset_release.json', 'precision_audit.json',
    'conversion.json', 'preparation.json', 'selected_branches.json', 'source_progress.json',
    'source_plan.json', 'plan.json', 'cohort.json', 'release.json', 'DATA_CARD_ERRATA.md'}
SKIP_DIRS = {'tracking', '.git', '__pycache__', '.pytest_cache', 'node_modules'}
FACT_KEYS = {'material', 'element', 'materials', 'material_order', 'atomic_numbers', 'temperature_K',
    'temperatures_K', 'material_temperature_K', 'timestep_fs', 'timestep_ps', 'material_timestep_fs',
    'sample_interval_ps', 'cadence_ps', 'duration_ps', 'measurement_duration_ps', 'ensemble',
    'pressure_bar', 'thermostat', 'barostat', 'thermostat_damping_ps', 'barostat_damping_ps',
    'atom_count', 'frame_count', 'units', 'coordinate_convention', 'split', 'source_split',
    'split_unit', 'root_lineage', 'lineage', 'structural_independence', 'branch_semantics',
    'scientific_scope', 'scientific_contract', 'ptm_rmsd_cutoff', 'seed', 'velocity_seed',
    'melt_seed', 'preparation_seed', 'parent_id', 'storage_dtype', 'position_dtype', 'velocity_dtype'}
FACT_KEYS.update({'radius_A','radius','cutoff_A','r_max','history_ps','frame_offsets_ps',
    'history_offsets_ps','future_offsets_ps','future_lags_ps','horizons_ps','target_names',
    'target_columns','descriptor_names','neighbor_count','atoms_per_cloud','training_eligible',
    'source_run_id','parent_trajectory_id','trajectory_id','periodic','pbc',
    'model_name','model_path','checkpoint_path','potential_name','pair_style','pair_coeff',
    'lammps_version','simulator_version','thermostat_ps','barostat_ps','mass_g_mol',
    'melt_duration_ps','melt_temperature_K','equilibration_duration_ps'})
PROVENANCE_KEYS = {'potential', 'potentials', 'potential_files', 'potential_hashes', 'potential_sha256',
    'potential_citation', 'pair_commands', 'scientific_scope', 'scientific_contract',
    'protocol_provenance', 'protocol_source', 'quantization', 'precision_audit', 'zr_removal'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.building')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def slug(identifier):
    return re.sub(r'[^A-Za-z0-9_-]', '-', identifier)[:100]+'-'+hashlib.sha256(identifier.encode()).hexdigest()[:8]


def header(path):
    """Read only NPY metadata, including object arrays without unpickling them."""
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    return dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran)


def zip_headers(path):
    """Read compressed NPY headers without decompressing complete cached arrays."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as stream:
                shape, fortran, dtype = np.lib.format._read_array_header(stream, np.lib.format.read_magic(stream))
            result[name[:-4]] = dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran)
    return result


def facts(document):
    """Retain exact known scientific fields and their JSON paths; no name guessing."""
    result = defaultdict(dict)

    def visit(value, path=''):
        if isinstance(value, dict):
            for key, item in value.items():
                child = f'{path}.{key}' if path else key
                if key in FACT_KEYS or (key == 'protocol' and isinstance(item,str)):
                    encoded = json.dumps(item, sort_keys=True)
                    bucket = result[key]
                    if encoded not in bucket:
                        bucket[encoded] = dict(value=item, first_json_path=child, occurrences=0)
                    bucket[encoded]['occurrences'] += 1
                # Numeric arrays and checkpoint tensors are not metadata fields.
                if key not in ('arrays', 'checksums', 'files', 'positions', 'velocities',
                               'output_steps', 'center_atom_ids', 'pool_atom_ids', 'frames'):
                    visit(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    visit(item, f'{path}[{index}]')
    visit(document)
    return {key:list(values.values()) for key, values in result.items()}


def potential_checksums(value):
    """Do not confuse an encoder checkpoint with the trajectory-generating model."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('potential', 'potentials', 'potential_files', 'potential_hashes',
                       'potential_sha256', 'library_sha256', 'parameter_sha256'):
                result.update(re.findall(r'[0-9a-f]{64}', json.dumps(item)))
            elif isinstance(item, (dict, list)):
                result.update(potential_checksums(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                result.update(potential_checksums(item))
    return result


def trajectory_record(path, document):
    """A declared complete manifest alone is insufficient for an available binary."""
    problems, arrays = [], {}
    for name, description in document['arrays'].items():
        array_path = path.parent/description['file']
        if not array_path.is_file():
            problems.append(f'Missing array: {array_path}')
            continue
        observed = header(array_path)
        arrays[name] = dict(**observed, declared_sha256=description['sha256'], file=description['file'])
        if observed['shape'] != description['shape'] or observed['dtype'] != description['dtype']:
            problems.append(f'Header disagrees with manifest: {array_path}')
    required = {'positions','timesteps','box_low','box_high','atom_ids','atom_types'}
    if document['format'] == 'pointcloudmaterials.shooting_trajectory':
        required.add('velocities')
    if required-set(arrays):
        problems.append(f'Missing required fields: {sorted(required-set(arrays))}')
    if 'positions' in arrays and arrays['positions']['shape'] != [document['frame_count'], document['atom_count'], 3]:
        problems.append('Position shape disagrees with declared atom/frame count')
    timeline = None
    if 'timesteps' in arrays:
        steps = np.load(path.parent/'timesteps.npy', allow_pickle=False)
        if steps.ndim != 1 or len(steps) != document['frame_count'] or np.any(np.diff(steps) <= 0):
            problems.append('Invalid timeline: must be one strictly increasing step per frame')
        else:
            timeline = dict(first_step=int(steps[0]), last_step=int(steps[-1]),
                            step_intervals=np.unique(np.diff(steps)).tolist())
    fingerprint = hashlib.sha256(json.dumps({key:description['sha256']
        for key, description in document['arrays'].items()}, sort_keys=True).encode()).hexdigest()
    return dict(format=document['format'], recorded_state=document['state'],
        usable_binary=document['state']=='complete' and not problems,
        atom_count=document['atom_count'], frame_count=document['frame_count'], arrays=arrays,
        timeline=timeline, content_signature_from_manifest=fingerprint, issues=problems,
        verification='Array existence and NPY headers checked; large array checksums are producer-declared, not rehashed.')


def resolve_reference(value, roots, entries):
    """Resolve the registry's explicit tokens against one frozen machine snapshot."""
    def replace(match):
        kind, key = match.groups()
        if kind == 'storage':
            return roots[key]
        entry = entries[key]
        return str(Path(roots[entry['root']])/entry['path'])
    text = re.sub(r'\$\{(storage|dataset):([^{}]+)\}', replace, value)
    path = Path(text)
    return path if path.is_absolute() else REPO/path


def inspect_dataset(identifier, entry, root, child_roots, potential_hashes, references):
    records, trajectories, loose_arrays, links, issues = [], [], [], [], []
    counts, states, extensions = Counter(), Counter(), Counter()
    aggregate = defaultdict(set)
    matched_potentials = set(entry.get('metadata', {}).get('potential_ids', []))
    bytes_apparent = bytes_allocated = 0
    metadata = entry.get('metadata', {})
    if root.is_dir():
        for directory, dirs, names in os.walk(root, followlinks=False):
            current = Path(directory)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and
                not (current/d).is_symlink() and (current/d).resolve() not in child_roots)
            # A research-code archive is not a recursively scanned raw dataset.
            if entry['kind'] == 'research_archive':
                dirs[:] = []
            for name in sorted(names):
                path = current/name
                if path.is_symlink():
                    links.append(dict(path=str(path.relative_to(root)), target=str(path.resolve()), available=path.exists()))
                    continue
                stat = path.stat()
                bytes_apparent += stat.st_size
                bytes_allocated += stat.st_blocks*512
                counts['files'] += 1
                extensions[path.suffix.lower() or '(none)'] += 1
                if name.endswith('.npy'):
                    try:
                        loose_arrays.append(dict(path=str(path.relative_to(root)), **header(path)))
                    except (ValueError, EOFError) as error:
                        issues.append(f'Unreadable NPY header {path}: {error}')
                if name.endswith('.npz'):
                    try:
                        loose_arrays.append(dict(path=str(path.relative_to(root)), members=zip_headers(path)))
                    except (ValueError, EOFError, zipfile.BadZipFile) as error:
                        issues.append(f'Unreadable NPZ headers {path}: {error}')
                if name not in DOCUMENTS and not (name.startswith('source-') and name.endswith('.json')):
                    continue
                if path.suffix != '.json':
                    records.append(dict(path=str(path), relative_path=str(path.relative_to(root)), sha256=sha(path), kind='data_card'))
                    continue
                try:
                    raw = path.read_bytes()
                    document = json.loads(raw)
                except (ValueError, OSError) as error:
                    issues.append(f'Unreadable metadata {path}: {error}')
                    continue
                if not isinstance(document, dict):
                    records.append(dict(path=str(path), relative_path=str(path.relative_to(root)),
                        sha256=hashlib.sha256(raw).hexdigest(), kind='list_metadata', entries=len(document)))
                    continue
                record = dict(path=str(path), relative_path=str(path.relative_to(root)),
                    sha256=hashlib.sha256(raw).hexdigest(), kind='metadata', keys=sorted(document),
                    recorded_state=document.get('state'), facts=facts(document),
                    provenance={k:document[k] for k in PROVENANCE_KEYS if k in document})
                if document.get('format') in TRAJECTORY_FORMATS:
                    record['trajectory'] = trajectory_record(path, document)
                    record['kind'] = 'trajectory'
                    trajectories.append(record)
                    issues.extend(record['trajectory']['issues'])
                for field, values in record['facts'].items():
                    for value in values:
                        aggregate[field].add(json.dumps(value['value'], sort_keys=True))
                if name in ('outcome.json', 'metadata.json') and 'state' in document:
                    states[str(document['state'])] += 1
                for checksum in potential_checksums(document):
                    matched_potentials.update(potential_hashes.get(checksum, []))
                records.append(record)
    materials = set(metadata.get('materials', []))
    for key in ('material','element','materials','material_order'):
        for encoded in aggregate[key]:
            value = json.loads(encoded)
            if isinstance(value, str): materials.add(value)
            elif isinstance(value, list): materials.update(v for v in value if isinstance(v, str))
    unknown = []
    if not materials: unknown.append('materials')
    if not matched_potentials and metadata.get('role') not in ('synthetic','administrative','container'):
        unknown.append('generating potential identity')
    for key in ('temperature_K', 'timestep_fs', 'ensemble'):
        alternatives = {'temperature_K':('temperatures_K','material_temperature_K'),
                        'timestep_fs':('timestep_ps','material_timestep_fs'), 'ensemble':()}
        if entry['kind'] == 'simulation' and not any(aggregate[k] for k in (key,*alternatives[key])):
            unknown.append(key)
    usable = [r for r in trajectories if r['trajectory']['usable_binary']]
    result = dict(id=identifier, title=metadata.get('title',identifier), slug=slug(identifier),
        kind=entry['kind'], role=metadata.get('role',entry['kind']),
        classification=metadata.get('classification','unreviewed'), description=metadata.get('description',''),
        materials=sorted(materials), potential_ids=sorted(matched_potentials),
        metadata=metadata, location=dict(root_role=entry['root'], relative_path=entry['path'],
            resolved=str(root), available=root.is_dir()), dependencies=entry.get('dependencies',[]),
        aliases=entry.get('aliases',[]), references=references,
        observed=dict(files=counts['files'], apparent_bytes=bytes_apparent, allocated_bytes=bytes_allocated,
            file_extensions=dict(extensions), producer_state_record_counts=dict(states),
            metadata_records=len(records), binary_trajectory_records=len(trajectories),
            available_complete_binary_records=len(usable), stored_frames=sum(r['trajectory']['frame_count'] for r in usable)),
        facts={k:[json.loads(v) for v in sorted(values)] for k,values in aggregate.items() if values},
        loose_arrays=loose_arrays, symlinks=links, missing_metadata=unknown, issues=issues,
        trajectory_signatures=[dict(path=r['path'], signature=r['trajectory']['content_signature_from_manifest']) for r in usable])
    return result, records


def source_references(entries):
    result = defaultdict(list)
    for base in ('configs', 'src'):
        for path in (REPO/base).rglob('*'):
            if path.suffix not in ('.json','.yaml','.yml','.py') or path.name == 'datasets.json':
                continue
            text = path.read_text()
            for identifier in entries:
                explicit_id = re.search(r'["\'](?:dataset|dataset_id)["\']\s*:\s*["\']'+re.escape(identifier)+r'["\']', text)
                long_id = len(identifier)>10 and ('"'+identifier+'"' in text or "'"+identifier+"'" in text)
                if '${dataset:'+identifier+'}' in text or explicit_id or long_id:
                    result[identifier].append(str(path.relative_to(REPO)))
    return result


def build_registry(output, *, settings=None, catalog_path=None):
    settings = settings or machine()
    roots = settings['roots']
    path = Path(catalog_path or settings['catalog'])
    path = path if path.is_absolute() else REPO/path
    catalog = json.loads(path.read_text())
    entries = catalog['datasets']
    locations = {key:(Path(roots[value['root']])/value['path']).resolve() for key,value in entries.items()}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    potential_hashes = defaultdict(list)
    potentials = []
    for key, value in catalog.get('potential_registry', {}).items():
        observed = []
        for item in value['files']:
            location = resolve_reference(item['path'], roots, entries)
            checksum = sha(location) if location.is_file() else None
            if checksum is not None and checksum != item['sha256']:
                raise ValueError(f'Potential file changed: {location}; expected {item["sha256"]}, found {checksum}')
            potential_hashes[item['sha256']].append(key)
            observed.append(dict(**item, location=str(location), available=location.is_file(), sha256_verified=checksum is not None))
        potentials.append(dict(id=key, **{k:v for k,v in value.items() if k!='files'}, files=observed))
    references = source_references(entries)
    datasets, signature_groups = [], defaultdict(list)
    for identifier, entry in sorted(entries.items()):
        print(f'INDEX {identifier}', flush=True)
        root = locations[identifier]
        child_roots = {p for key,p in locations.items() if key!=identifier and p!=root and p.is_relative_to(root)}
        dataset, records = inspect_dataset(identifier, entry, root, child_roots, potential_hashes, references[identifier])
        dataset['contains_registered'] = [key for key,p in locations.items() if p in child_roots]
        dataset['same_location_as'] = [key for key,p in locations.items() if key!=identifier and p==root]
        dataset['records_file'] = f'records/{dataset["slug"]}.json'
        write_json(output/dataset['records_file'], dict(dataset_id=identifier, records=records))
        for item in dataset.pop('trajectory_signatures'):
            signature_groups[item['signature']].append(dict(dataset_id=identifier,path=item['path']))
        datasets.append(dataset)
    duplicates = [dict(signature=key, records=value) for key,value in signature_groups.items() if len(value)>1]
    duplicate_ids = Counter(record['dataset_id'] for group in duplicates for record in group['records'])
    for dataset in datasets:
        dataset['observed']['binary_records_in_duplicate_groups'] = duplicate_ids[dataset['id']]
    unregistered = []
    for reference in catalog.get('registry_discovery_roots', []):
        directory = resolve_reference(reference, roots, entries)
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if child.is_dir() and child.resolve() not in locations.values():
                owners = [key for key,root in locations.items() if child.resolve().is_relative_to(root)]
                if not owners:
                    unregistered.append(dict(path=str(child), symlink=child.is_symlink(),
                        metadata_files=sorted(p.name for p in child.glob('*.json')),
                        note='Unregistered directory; may be a dataset, staging area or operational artifact. Not counted as research data.'))
    registry = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
        catalog_sha256=sha(path), implementation_sha256=sha(Path(__file__)), datasets=datasets,
        potentials=potentials, duplicate_binary_groups=duplicates, unregistered_directories=unregistered,
        remote_holdings=catalog.get('remote_holdings',[]),
        methodology='Non-atomic filesystem inventory. Current binary headers override historical precision declarations. Metadata and small potential files are hashed; large coordinate arrays are not rehashed. Counts include descendants, converted copies and diagnostics, and are not independent samples. Registered nested roots are excluded from parent storage counts. Recorded producer states do not establish scheduler liveness.')
    write_json(output/'registry.json', registry)
    from .dataset_registry_render import render_registry
    render_registry(registry, output)
    return dict(datasets=len(datasets), materials=sorted({m for d in datasets for m in d['materials']}),
        potential_definitions=len(potentials), metadata_records=sum(d['observed']['metadata_records'] for d in datasets),
        duplicate_binary_groups=len(duplicates), unregistered_directories=len(unregistered),
        integrity_issues=sum(len(d['issues']) for d in datasets), output=str(output))
