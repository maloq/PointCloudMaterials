import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from src.analysis.inference_cache import discard_inference_cache
from src.analysis.report import publish_report
from src.experiment_runner.cache_storage import relocate_caches
from src.training_methods.train_entrypoint import _run_registered_post_training_analysis


def test_flat_gallery_is_portable_and_rejects_another_checkpoint(tmp_path):
    source = tmp_path/'artifacts'
    source.mkdir()
    metrics = dict(clustering=dict(primary_k=7), topology=dict(checkpoint_sha256='selected'))
    (source/'analysis_metrics.json').write_text(json.dumps(metrics))
    (source/'latent_umap_clusters.png').write_bytes(b'figure')
    (source/'analysis_inference_cache.npz').write_bytes(b'large cache')
    output = tmp_path/'reports'/'anchor'
    publish_report(source, output)
    (source/'latent_umap_clusters.png').unlink()
    assert (output/'umap.png').read_bytes() == b'figure'
    assert not (output/'analysis_inference_cache.npz').exists()
    assert 'umap.png' in (output/'index.html').read_text()
    assert 'anchor/index.html' in (output.parent/'index.html').read_text()
    metrics['topology']['checkpoint_sha256'] = 'different'
    (source/'analysis_metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(FileExistsError, match='already belongs'):
        publish_report(source, output)


def test_cache_relocation_preserves_bytes_links_and_old_loader_path(tmp_path):
    source = tmp_path/'cache'
    source.mkdir()
    (source/'views.npy').write_bytes(b'physical positions')
    external = tmp_path/'base.npy'
    external.write_bytes(b'shared base views')
    (source/'base.npy').symlink_to(external)
    destination = tmp_path/'ids'/'cache'
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(audit=str(tmp_path/'audit.json'), moves=[dict(
        source=str(source), destination=str(destination), producer='test producer')])))
    relocate_caches(plan, apply=True)
    assert source.is_symlink() and source.resolve() == destination
    assert (source/'views.npy').read_bytes() == b'physical positions'
    assert (source/'base.npy').resolve() == external
    assert json.loads((tmp_path/'audit.json').read_text())[0]['state'] == 'complete'
    assert relocate_caches(plan, apply=True)[0]['state'] == 'already_relocated'
    recorded_verification = (tmp_path/'audit.json').read_bytes()
    relocate_caches(plan)
    assert (tmp_path/'audit.json').read_bytes() == recorded_verification


def test_discard_only_inference_arrays_and_matching_metadata(tmp_path):
    for name in ['cache.npz','cache.npz.meta.json','test_predictions.npz']:
        (tmp_path/name).write_bytes(b'preserved prediction or removable cache')
    discard_inference_cache(tmp_path, 'cache.npz')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['test_predictions.npz']


def test_recovery_checkpoint_is_removed_only_after_successful_analysis(tmp_path, monkeypatch):
    best, last = tmp_path/'best.ckpt', tmp_path/'last.ckpt'
    best.write_bytes(b'selected'); last.write_bytes(b'recovery')
    cfg = OmegaConf.create(dict(data=dict(kind='relaxed_histories'), checkpoint_keep_last_after_analysis=False))
    def failed(*args):
        raise RuntimeError('analysis failed')
    monkeypatch.setattr('src.training_methods.train_entrypoint.run_post_training_analysis_safe', failed)
    kwargs = dict(checkpoint_callbacks=[SimpleNamespace(best_model_path=str(best))], enabled_by_default=True, requested=True)
    with pytest.raises(RuntimeError, match='analysis failed'):
        _run_registered_post_training_analysis(cfg, **kwargs)
    assert last.exists()
    monkeypatch.setattr('src.training_methods.train_entrypoint.run_post_training_analysis_safe', lambda *args:None)
    _run_registered_post_training_analysis(cfg, **kwargs)
    assert best.exists() and not last.exists()
