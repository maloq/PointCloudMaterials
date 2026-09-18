"""Train-only constants and float64 checks for finite but uninformative fits."""
from collections import defaultdict
import numpy as np


def spread(values):
    values=np.asarray(values,dtype=np.float64)
    return float(values.std(axis=0).mean())


def training_means(release):
    """Match the structural target producer's endpoint/valid-label populations."""
    sums=defaultdict(dict);counts=defaultdict(dict)
    for shard in release.manifest['shards']:
        if shard['task']['split']!='train':continue
        key=(shard['material'],shard['potential'],shard['static'])
        arrays=release.arrays[shard['task']['id']]
        views=arrays['views'][:,[2,4] if shard['static'] else [2,3,4]].ravel()
        if (views<0).any():raise ValueError(f'Missing structural training endpoint: {key}')
        targets=dict(physical=arrays['physical'][views],tda=arrays['tda'][arrays['tda_valid']])
        for name,values in targets.items():
            if not len(values):continue
            sums[key][name]=sums[key].get(name,0)+values.sum(0,dtype=np.float64)
            counts[key][name]=counts[key].get(name,0)+len(values)
    return {key:{name:total/counts[key][name] for name,total in parts.items()} for key,parts in sums.items()}


def check_learning(metrics,baseline,best_present,step,config):
    """Stop collapsed fits; require useful selection decoding after a grace period."""
    for key in ('state_std_mean','projector_std_mean','physical_prediction_std_mean','tda_prediction_std_mean'):
        value=metrics[key]
        if not np.isfinite(value) or value<=config['minimum_std']:
            raise FloatingPointError(f'Learning health failed at update {step}: {key}={value}; '
                f'requires > {config["minimum_std"]}. Check saved selection diagnostics.')
    if step>=config['baseline_after_updates'] and best_present>=baseline:
        raise RuntimeError(f'Learning health failed at update {step}: best present selection '
            f'{best_present:.6g} has not beaten training-only group-mean baseline {baseline:.6g}.')


def check_regression(scores,baseline,step,config):
    """A good old checkpoint does not make persistent later degradation healthy."""
    if step<config['baseline_after_updates']:return
    required=baseline*(1-config['minimum_relative_gain'])
    if min(scores)>=required:
        raise RuntimeError(f'Learning health at update {step}: best score {min(scores):.6g} '
            f'must be below {required:.6g} ({config["minimum_relative_gain"]:.1%} gain over baseline)')
    patience=config['regression_patience']
    if len(scores)<=patience:return
    previous=min(scores[:-patience]);limit=min(required,previous*config['maximum_regression_ratio'])
    if all(value>limit for value in scores[-patience:]):
        raise RuntimeError(f'Persistent selection regression at update {step}: last {patience} '
            f'scores {scores[-patience:]} exceed {limit:.6g}; previous best {previous:.6g}')
