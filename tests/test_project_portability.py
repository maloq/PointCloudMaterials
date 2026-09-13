"""Portable paths, byte-preserving bundles, publication and fresh-run placement."""

import json
from pathlib import Path
import shutil

import pytest
import yaml
from omegaconf import OmegaConf

from src.project_runtime import paths, transfer


@pytest.fixture
def project(tmp_path, monkeypatch):
    repo = tmp_path / 'checkout'
    repo.mkdir()
    for name in ['src', 'scripts', 'configs', 'environments', 'docs']:
        (repo / name).mkdir()
    for name in ['README.md', 'requirements.txt']:
        (repo / name).write_text(name)
    (repo / 'configs/machines').mkdir()
    shutil.copy2(paths.REPO / 'configs/machines/local.yaml', repo / 'configs/machines/local.yaml')
    machine_path = repo / 'machine.local.yaml'
    machine_path.write_text(yaml.safe_dump(dict(roots={'cache': str(tmp_path/'cache'),
        'datasets': str(tmp_path/'datasets'), 'archive': str(tmp_path/'archive'),
        'simulation_runs': str(tmp_path/'scratch/simulations')},
        legacy_paths={'/old-machine/cache': '${storage:cache}'})))
    (repo / 'configs/datasets.json').write_text(json.dumps(dict(schema_version=1, datasets={})))
    monkeypatch.setattr(paths, 'REPO', repo)
    monkeypatch.setattr(transfer, 'REPO', repo)
    monkeypatch.setenv('PCM_MACHINE_CONFIG', str(machine_path))
    return repo


def test_simulation_index_keeps_attempt_evidence_without_following_aliases(project):
    import csv
    from src.project_runtime.simulation_inventory import export_simulations

    root = project / 'simulation'
    (root / 'branches/one').mkdir(parents=True)
    (root / 'interrupted/one').mkdir(parents=True)
    (root / 'branches/one/outcome.json').write_text('{"state":"complete"}')
    (root / 'interrupted/one/outcome.json').write_text('{"state":"failed"}')
    (root / 'duplicate-link').symlink_to(root / 'branches/one', target_is_directory=True)
    entries = {name: dict(root='repo', path=path, kind=kind, dependencies=[])
               for name, path, kind in [('campaign', 'simulation', 'simulation'),
                                        ('unavailable', 'missing', 'simulation'),
                                        ('cache', 'simulation', 'cache')]}
    (project / 'configs/datasets.json').write_text(json.dumps(dict(schema_version=1, datasets=entries)))
    out = project / 'docs/simulations'
    result = export_simulations(out)
    assert result['collections'] == 2
    with (out / 'run_records.csv').open() as stream:
        records = list(csv.DictReader(stream))
    assert {r['recorded_state'] for r in records} == {'complete', 'failed'}
    assert len(records) == 2
    with (out / 'collections.csv').open() as stream:
        collections = {r['dataset_id']: r for r in csv.DictReader(stream)}
    assert collections['unavailable']['available'] == 'False'


def test_shared_json_yaml_resolution_and_path_independent_identity(project):
    entries = dict(schema_version=1, datasets={'sample': dict(root='cache', path='sample',
        kind='cache', dependencies=[], aliases=['/recorded/cache/sample'])})
    (project/'configs/datasets.json').write_text(json.dumps(entries))
    payload = {'input': '${dataset:sample}/views.npy', 'output': '${storage:output}/test', 'seed': 71}
    (project/'config.json').write_text(json.dumps(payload))
    resolved = paths.load_json(project/'config.json')
    assert resolved == OmegaConf.to_container(OmegaConf.create(payload), resolve=True)
    assert paths.resolve_path('/recorded/cache/sample/views.npy') == Path(resolved['input'])
    old = dict(input='/old-machine/cache/sample/views.npy', output='output/test', seed=71)
    assert paths.portable_config(old) == paths.portable_config(resolved)
    assert paths.portable_config(dict(old, seed=72)) != paths.portable_config(resolved)
    with pytest.raises(KeyError, match='Unknown dataset ID'):
        paths.dataset_path('missing')


def test_bundle_survives_move_and_preserves_manifest_bytes(project, tmp_path):
    cache = paths.storage_path('cache')
    (cache/'parent').mkdir(parents=True)
    (cache/'child').mkdir()
    (cache/'parent/frames.npy').write_bytes(b'exact scientific payload')
    (cache/'child/frames.npy').symlink_to(cache/'parent/frames.npy')
    manifest = b'{"original_path":"/old-machine/cache/parent"}\n'
    (cache/'child/manifest.json').write_bytes(manifest)
    entries = {name: dict(root='cache', path=name, kind='cache', dependencies=deps, aliases=[])
               for name, deps in [('parent', []), ('child', ['parent'])]}
    (project/'configs/datasets.json').write_text(json.dumps(dict(schema_version=1, datasets=entries)))
    plan = project/'selection.json'
    plan.write_text(json.dumps(dict(datasets=['child'], files=[])))
    destination = tmp_path/'export'
    preview = transfer.bundle(plan, destination)
    assert preview['state']=='preview' and not destination.exists()
    transfer.bundle(plan, destination, apply=True)
    moved = tmp_path/'another-machine'
    destination.rename(moved)
    assert transfer.verify_bundle(moved)['state']=='verified'
    assert (moved/'data/bundle/child/frames.npy').read_bytes()==b'exact scientific payload'
    assert (moved/'data/bundle/child/manifest.json').read_bytes()==manifest
    (moved/'data/bundle/parent/frames.npy').write_bytes(b'damaged')
    with pytest.raises(RuntimeError, match='inventory changed'):
        transfer.verify_bundle(moved)


def test_copy_keeps_original_when_source_changes(project, tmp_path, monkeypatch):
    source = tmp_path/'source'; source.mkdir()
    (source/'state.bin').write_bytes(b'original restart')
    copy = shutil.copytree
    def changing_copy(src, dst, **kwargs):
        result = copy(src, dst, **kwargs)
        (source/'state.bin').write_bytes(b'new live restart')
        return result
    monkeypatch.setattr(transfer.shutil, 'copytree', changing_copy)
    with pytest.raises(RuntimeError, match='source changed'):
        transfer.verified_copy(source, tmp_path/'copy', tmp_path/'audit.json', move=True)
    assert source.is_dir() and not source.is_symlink()
    assert (source/'state.bin').read_bytes()==b'new live restart'


def test_publish_requires_completion_and_keeps_portable_alias(project, tmp_path):
    source = tmp_path/'scratch/run'; source.mkdir(parents=True)
    (source/'status.json').write_text(json.dumps(dict(state='running')))
    with pytest.raises(RuntimeError, match='Only complete'):
        transfer.publish_simulation(source, identifier='run', move=True)
    (source/'status.json').write_text(json.dumps(dict(state='complete')))
    (source/'final.restart.bin').write_bytes(b'exact restart state')
    potential = paths.storage_path('datasets')/'potential'
    potential.mkdir(parents=True)
    (potential/'Ti.meam').write_bytes(b'potential bytes')
    (source/'technical').mkdir()
    (source/'technical/launch_config.json').write_text(json.dumps(dict(
        potential_files=[dict(path=str(potential/'Ti.meam'))])))
    with pytest.raises(ValueError, match='Register the potential'):
        transfer.publish_simulation(source, identifier='run', move=True)
    assert not source.is_symlink()
    (project/'configs/datasets.json').write_text(json.dumps(dict(schema_version=1,
        datasets={'ti-potential': dict(root='datasets', path='potential',kind='potential',dependencies=[])})))
    transfer.publish_simulation(source, identifier='run', move=True)
    assert paths.catalog()['run']['dependencies']==['ti-potential']
    assert source.is_symlink()
    assert paths.dataset_path('run').joinpath('final.restart.bin').read_bytes()==b'exact restart state'
    source.unlink()  # Model the cluster's later SCRATCH purge.
    assert paths.resolve_path(str(source/'final.restart.bin')).read_bytes()==b'exact restart state'


def test_fresh_elemental_launch_uses_scratch_and_explicit_cpu_count(project, monkeypatch):
    from src.simulation.campaigns import elemental
    config = project/'ti.json'
    config.write_text(json.dumps(dict(output_root='/wrong/durable/output', material='Ti',
        lammps='/old-machine/lmp', mpiexec='/old-machine/mpiexec', cpus=[100,101],
        position_storage_dtype='float32', atom_count=100000)))
    monkeypatch.setenv('SLURM_JOB_ID','test-job')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','32')
    launch = elemental.prepare_fresh_launch(config,'new-ti',ranks=1)
    payload = json.loads(launch.read_text())
    elemental.bind_allocation(payload)
    assert len(payload['cpus'])==1
    assert Path(payload['output_root']).is_relative_to(paths.storage_path('simulation_runs'))
    assert payload['position_storage_dtype']=='float16'
    assert payload['atom_count']==100000
    assert 'mpiexec' not in payload and 'lammps' not in payload
    assert payload['execution']['lammps']=='lmp'


def test_checkpoint_resume_after_cache_relocation_matches_uninterrupted(tmp_path, monkeypatch):
    import runpy
    import torch
    from src.training_methods.embedding_forecast.run import train
    fixtures = runpy.run_path(str(paths.REPO / 'tests/test_embedding_forecast.py'))
    old_cache = tmp_path / 'old-cache'
    fixtures['cache_fixture'](old_cache)
    variant = fixtures['autoregressive_variant']()
    config = dict(data=dict(cache=str(old_cache)), output=str(tmp_path/'whole'),
        history_ps=1.5, anchor_history_ps=1.5, horizons_ps=[0.75,1.5,2.25], stride_ps=0.75,
        seeds=[3], training=dict(epochs=4, patience=4, cpu_threads=1, scale_floor_fraction=0.05,
        batch_size=13, workers=0, learning_rate=0.01, weight_decay=0.0001,
        minimum_lr_fraction=0.2, gradient_clip=5.0, warmup_epochs=1,
        augmentation=dict(noise_std=0.01, frame_dropout=0.15)), variants=[variant],
        comparisons=[], bin_comparisons=[])
    train(config, variant, 3, 'cpu')
    chunked = dict(config, output=str(tmp_path/'chunked'))
    train(chunked, variant, 3, 'cpu', epochs_per_invocation=2)
    new_cache = tmp_path/'new-machine/cache'
    new_cache.parent.mkdir()
    old_cache.rename(new_cache)
    profile = tmp_path/'machine.yaml'
    profile.write_text(yaml.safe_dump(dict(roots=dict(cache=str(new_cache)),
        legacy_paths={str(old_cache): '${storage:cache}'})))
    monkeypatch.setenv('PCM_MACHINE_CONFIG', str(profile))
    train(chunked, variant, 3, 'cpu', resume=True, epochs_per_invocation=2)
    suffix = f"{variant['name']}-seed3/technical/last.pt"
    whole = torch.load(tmp_path/'whole'/suffix, weights_only=False)
    resumed = torch.load(tmp_path/'chunked'/suffix, weights_only=False)
    for name in whole['model']:
        torch.testing.assert_close(whole['model'][name], resumed['model'][name], rtol=0, atol=0)
    assert whole['scheduler']==resumed['scheduler']
    torch.testing.assert_close(whole['sampler_rng'], resumed['sampler_rng'], rtol=0, atol=0)
    with pytest.raises(ValueError, match='exact scientific config'):
        train(dict(chunked, stride_ps=1.5), variant, 3, 'cpu', resume=True)


def test_storage_transition_requires_exact_cache_and_model(tmp_path):
    from src.training_methods.embedding_forecast.runtime import implementation_hashes, check_resume_implementation
    current = implementation_hashes()
    previous = dict(current, **{'data.py': '0'*64})
    checkpoint = dict(implementation_sha256=previous, epoch=2, cache_manifest_sha256='1'*64)
    transition = tmp_path/'transition.json'
    payload = dict(kind='storage-relocation', previous=previous, replacement=current,
                   cache_manifest_sha256=['2'*64])
    transition.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='exact verified cache'):
        check_resume_implementation(checkpoint,current,transition,tmp_path)
    payload['cache_manifest_sha256']=['1'*64]
    transition.write_text(json.dumps(payload))
    check_resume_implementation(checkpoint,current,transition,tmp_path)
    changed = dict(current, **{'model.py': '3'*64})
    payload['replacement']=changed
    transition.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='preserve'):
        check_resume_implementation(checkpoint,changed,transition,tmp_path)
