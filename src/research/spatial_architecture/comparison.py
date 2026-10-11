"""Fixed-final-epoch MACE order ablation, using a pinned existing reference."""
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.structural_response.data import read, root, digest, sha, write_json


def validate_reference(c):
    reference = c['comparison_reference']
    previous = resolve_path(reference['output'])
    plan = read(previous/'technical/plan.json')
    expected = digest(dict(plan=plan['identity'], architecture='mace'))
    if expected != reference['identity']:
        raise ValueError('Changed MACE order-ablation reference identity')
    original = plan['config']
    for field in ('fixed_dataset', 'cache', 'structural_dataset', 'data', 'loader',
                  'prepared_identity', 'derived_cache', 'seed', 'methods', 'spatial_training'):
        if c[field] != original[field]:
            raise ValueError(f'Unmatched MACE order ablation: {field}')
    if c['execution_variants']['mace'] != original['execution_variants']['mace']:
        raise ValueError('MACE execution variant differs from the reference')
    changed = {k for k in set(c['reference_encoder']) | set(original['reference_encoder'])
               if c['reference_encoder'].get(k) != original['reference_encoder'].get(k)}
    if changed != {'max_ell', 'correlation'}:
        raise ValueError(f'Expected only angular/correlation changes, found {changed}')
    return previous, plan


def publish(c):
    previous, plan = validate_reference(c)
    rows, sources = [], []
    for output, config, expected in (
            (previous, plan['config'], c['comparison_reference']['identity']),
            (root(c), c, digest(dict(plan=read(root(c)/'technical/plan.json')['identity'], architecture='mace')))):
        p = output/'analyses/mace/technical'
        result = read(p/'complete.json')
        if result['identity'] != expected or result['update'] != c['spatial_training']['updates']:
            raise ValueError(f'Incomplete or changed order-ablation model: {p}')
        if sha(p/'final.pt') != result['checkpoint_sha256']:
            raise ValueError(f'Changed final checkpoint: {p}')
        parameters = read(p/'parameters.json')
        encoder = config['reference_encoder']
        seconds = result['timing']['training_seconds']
        rows.append(dict(model=f"mace-l{encoder['max_ell']}-c{encoder['correlation']}",
            angular_order=encoder['max_ell'], correlation=encoder['correlation'],
            encoder_parameters=parameters['encoder'], total_parameters=parameters['total'],
            gpu=parameters['gpu'], updates=result['update'], epochs=result['epochs'],
            training_seconds=seconds, mean_update_seconds=seconds/result['update'],
            **result['validation']))
        sources.append(dict(output=str(output), identity=expected,
            checkpoint_sha256=result['checkpoint_sha256'], completion_sha256=sha(p/'complete.json')))
    if rows[0]['gpu'] != rows[1]['gpu']:
        raise ValueError('Order-ablation timing comparison requires the same GPU model')
    analysis = root(c)/'analyses/order-comparison'
    write_metric_rows(rows, analysis, family='spatial_architecture', name='final_models')
    write_json(analysis/'technical/sources.json', sources)
    return dict(analysis=str(analysis),
        l2_c2_over_l3_c3_training_time=rows[1]['training_seconds']/rows[0]['training_seconds'],
        l2_c2_over_l3_c3_descriptor_error=rows[1]['descriptor_half_mse']/rows[0]['descriptor_half_mse'])
