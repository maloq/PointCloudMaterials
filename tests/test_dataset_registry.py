"""Registry evidence, physical schemas, duplicate groups and safe HTML output."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.project_runtime.dataset_registry import build_registry, facts, potential_checksums, trajectory_record, zip_headers


def binary(root, *, velocity=True, dtype='float16'):
    root.mkdir(parents=True)
    arrays = dict(positions=np.zeros((3,4,3),dtype),timesteps=np.array([0,100,200],np.int64),
        box_low=np.zeros((3,3),np.float32),box_high=np.ones((3,3),np.float32)*10,
        atom_ids=np.arange(1,5,dtype=np.int64),atom_types=np.ones(4,np.int32))
    if velocity: arrays['velocities'] = np.zeros((3,4,3),dtype)
    description = {}
    for name,value in arrays.items():
        np.save(root/f'{name}.npy',value)
        description[name] = dict(file=f'{name}.npy',shape=list(value.shape),dtype=str(value.dtype),
                                 sha256=hashlib.sha256(value.tobytes()).hexdigest())
    document = dict(format='pointcloudmaterials.shooting_trajectory' if velocity else 'pointcloudmaterials.temporal_lammps_trajectory',
        state='complete',arrays=description,frame_count=3,atom_count=4,storage_dtype=dtype)
    (root/'manifest.json').write_text(json.dumps(document))
    return document


def test_current_headers_override_stale_precision_and_missing_arrays(tmp_path):
    doc=binary(tmp_path/'binary')
    path=tmp_path/'binary/manifest.json'
    observed=trajectory_record(path,doc)
    assert observed['usable_binary'] and observed['arrays']['positions']['dtype']=='float16'
    assert observed['timeline']['step_intervals']==[100]
    doc['arrays']['positions']['dtype']='float32'
    stale=trajectory_record(path,doc)
    assert not stale['usable_binary'] and any('Header disagrees' in issue for issue in stale['issues'])
    (path.parent/'velocities.npy').unlink()
    assert any('Missing' in issue for issue in trajectory_record(path,doc)['issues'])


def test_positions_only_does_not_invent_velocities(tmp_path):
    doc=binary(tmp_path/'binary',velocity=False)
    observed=trajectory_record(tmp_path/'binary/manifest.json',doc)
    assert observed['usable_binary'] and 'velocities' not in observed['arrays']


def test_npz_header_schema_does_not_unpickle_payload(tmp_path):
    path=tmp_path/'targets.npz'
    np.savez_compressed(path, targets=np.zeros((64,144),np.float32), labels=np.array([{'a':1}],dtype=object))
    schema=zip_headers(path)
    assert schema['targets']['shape']==[64,144]
    assert schema['targets']['dtype']=='float32'
    assert schema['labels']['dtype']=='object'


def test_potential_identity_does_not_use_encoder_checkpoint():
    encoder='1'*64; generating='2'*64
    data=dict(encoder={'checkpoint_sha256':encoder}, sources=[{'potential_hashes':{'Al.meam':generating}}])
    assert potential_checksums(data)=={generating}
    parsed=facts({'protocol':{'temperature_K':500}, 'sources':[{'temperature_K':500,'source_split':'test'}]})
    assert parsed['temperature_K'][0]['occurrences']==2
    assert parsed['source_split'][0]['value']=='test'


def test_complete_browser_keeps_missing_and_duplicates_and_prunes_nested_roots(tmp_path):
    storage=tmp_path/'storage';storage.mkdir()
    binary(storage/'parent/child/runA')
    binary(storage/'copy/runB')
    (storage/'parent/notes.json').write_text('{}')
    (storage/'unregistered').mkdir()
    (storage/'parent/child/config.json').write_text(json.dumps(dict(material='Ti',temperature_K=1250,timestep_ps=.001)))
    entries={
        'parent':dict(root='datasets',path='parent',kind='simulation',metadata={'role':'container'}),
        'child':dict(root='datasets',path='parent/child',kind='simulation',metadata={'title':'Ti <script>alert(1)</script>','materials':['Ti'],'classification':'research'}),
        'copy':dict(root='datasets',path='copy',kind='simulation',metadata={'classification':'duplicate'}),
        'missing':dict(root='datasets',path='absent',kind='dataset')}
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(schema_version=1,datasets=entries,registry_discovery_roots=['${storage:datasets}'])))
    out=tmp_path/'registry'
    build_registry(out,settings={'roots':{'datasets':str(storage)}},catalog_path=catalog)
    r=json.loads((out/'registry.json').read_text());d={d['id']:d for d in r['datasets']}
    assert d['parent']['observed']['available_complete_binary_records']==0
    assert d['child']['observed']['available_complete_binary_records']==1
    assert len(r['duplicate_binary_groups'])==1
    assert not d['missing']['location']['available']
    assert r['unregistered_directories'][0]['path']==str(storage/'unregistered')
    assert '<script>alert(1)</script>' not in (out/'index.html').read_text()
    assert '&lt;script&gt;' in (out/'index.html').read_text()
    record=json.loads((out/d['child']['records_file']).read_text())
    assert any('trajectory' in value for value in record['records'])
    assert (out/'cards'/f'{d["child"]["slug"]}.md').is_file()


def test_changed_potential_fails_before_publication(tmp_path):
    (tmp_path/'Al.meam').write_text('changed')
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps(dict(schema_version=1,datasets={},potential_registry={
        'al':{'name':'Al MEAM','files':[{'path':'${storage:datasets}/Al.meam','sha256':'0'*64}]}})))
    with pytest.raises(ValueError,match='Potential file changed'):
        build_registry(tmp_path/'registry',settings={'roots':{'datasets':str(tmp_path)}},catalog_path=catalog)
