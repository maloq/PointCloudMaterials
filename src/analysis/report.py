"""Structured publication of standard analyses; rendering never recalculates scores."""
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.experiment_runner.artifacts import analysis_artifacts, write_json
from src.experiment_runner.metric_docs import write_metric_table
from src.experiment_runner.result_records import file_hash, location
from .publication import publish_bundle


def report_directory(cfg, analysis_cfg):
    root = OmegaConf.select(analysis_cfg, 'report.root')
    if root is None:
        return None
    variant = OmegaConf.select(cfg, 'experiment_variant')
    name = (f'{variant}-seed{cfg.seed_everything}' if variant is not None
            else OmegaConf.select(analysis_cfg, 'report.model_name', default=cfg.experiment_name))
    return Path(root)/str(name).lower().replace('_', '-')


def publish_report(source, destination, *, checkpoint_sha256=None, update_index=True,
                   computed=False, analysis_cfg=None):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if not (source/'analysis_metrics.json').is_file():
        source = analysis_artifacts(source)
    metrics = json.loads((source/'analysis_metrics.json').read_text())
    if computed:
        previous = destination/'analyses/standard-v1/analysis.json'
        if previous.exists() and json.loads(previous.read_text())['numerical_evidence']['sha256']!=file_hash(source/'analysis_metrics.json'):
            raise ValueError(f'{previous}: numerical evidence changed; export to a new analysis revision')
        # Only the numerical producer may create a new metric export. Publication
        # calls retain the original calculation definitions.
        calculation_root = source.parent if source.name == 'data' else destination/'analyses/standard-v1'
        write_metric_table(metrics, calculation_root, family='analysis', name='scores')
    metadata = dict(protocol='standard-analysis',
        inputs={'evidence': metrics.get('inference_cache', {}),
                'note': 'Native input/representation contract is recorded in retained cache metadata.'},
        population={'static_sources': metrics.get('analysis_source_names'),
                    'temporal_inputs': metrics.get('temporal_real_inputs'),
                    'note': 'Descriptive unless the linked scientific protocol establishes held-out sampling.'},
        selection={'note': 'Checkpoint supplied by caller; analysis does not select models.'})
    provenance_path = source/'encoder-provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        metadata['selection'] = dict(rule=provenance['selection'],method=provenance['method'],
            epoch=provenance['epoch'],source_checkpoint_sha256=provenance['source_sha256'],
            receipt=location(provenance_path))
    native_path = source/'structural-inference-protocol.json'
    if native_path.exists():
        native = json.loads(native_path.read_text())
        metadata['inputs']['native_receipt'] = dict(path=location(native_path),sha256=file_hash(native_path))
        metadata['inputs']['representation'] = native['representation']
        metadata['inputs']['protocol'] = native['protocol']
        if native['protocol'] == 'native_capacity_mace_static_v1':
            frame = native['frames'][0]
            metadata['inputs'].update({k:frame[k] for k in ('encoder_inputs','predictor_inputs',
                'history','motion','explicit_conditions','observation','support_radius_A','candidate_atoms')})
    stages = None
    if analysis_cfg is not None:
        stages = {'numerical_analysis':dict(state='complete', evidence=str(source/'analysis_metrics.json'))}
        for name, option, key in [('representatives','figure_set.enabled','cluster_figure_set'),
                                 ('real_md','real_md.enabled','real_md_qualitative')]:
            enabled = bool(OmegaConf.select(analysis_cfg,option,default=False))
            recorded = key in metrics or key+'_by_k' in metrics or (name=='representatives' and 'cluster_figure_sets_by_snapshot' in metrics)
            stages[name] = dict(state=('complete' if recorded else 'unavailable') if enabled else 'disabled',
                                evidence=str(source/'analysis_metrics.json') if recorded else None,
                                note=f'{option}={enabled}; saved result block present={recorded}.')
        equivariance_enabled = (bool(OmegaConf.select(analysis_cfg,'equivariance.enabled',default=True))
                                and not metrics['runtime_profile'].get('equivariance_skipped',False))
        equivariance_recorded = 'equivariance' in metrics
        stages['equivariance'] = dict(
            state=('complete' if equivariance_recorded else 'unavailable') if equivariance_enabled else 'disabled',
            evidence=str(source/'analysis_metrics.json') if equivariance_recorded else None,
            note='State follows the requested option, runtime profile and saved result block.')
    analysis = publish_bundle(source,destination,checkpoint_sha256=checkpoint_sha256,
        metadata=metadata,stages=stages,refresh=update_index,
        include_paper_svg=bool(OmegaConf.select(analysis_cfg,'real_md.time_series.paper_enabled',default=False)) if analysis_cfg is not None else False,
        execution=dict(state='analysis_complete',evidence=location(source/'analysis_metrics.json'),
                       note='Numerical analysis completed; this is not a new encoder fit.') if computed else None)
    # The maintained topology collector discovers numerical results through this
    # exact producer receipt. Keep that interface without recreating flat plots.
    manifest = destination/'technical/source.json'
    if not manifest.exists():
        write_json(manifest,dict(analysis_directory=str(source),files={},
                                checkpoint_sha256=analysis['checkpoint_sha256']))
    numerical_alias = destination/'technical/metrics.json'
    if not numerical_alias.exists():
        numerical_alias.symlink_to(os.path.relpath(source/'analysis_metrics.json',numerical_alias.parent))
    return analysis
