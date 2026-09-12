"""Common local-topology scores, linear probes and whole-source uncertainty."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.data_utils.topology_targets import BLOCKS


def score(prediction, target, contexts, temperatures, scaling):
    error = prediction.astype(np.float64)-target.astype(np.float64)
    context_mean = np.empty_like(target)
    for context in np.unique(contexts):
        selected = contexts == context
        context_mean[selected] = target[selected].mean(0)
    blocks = {}
    for d, block in enumerate(BLOCKS):
        mse = np.mean(error[:, block]**2)
        variance = np.mean((target[:, block]-target[:, block].mean(0))**2)
        local_variance = np.mean((target[:, block]-context_mean[:, block])**2)
        blocks[f'H{d}'] = dict(mse=float(mse), r2=float(1-mse/variance),
            within_frame_r2=float(1-mse/local_variance), scaled_mse=float(mse/scaling['block_scale'][d]**2))
    per_row = np.mean(np.stack([np.mean(error[:, block]**2, axis=1)/scaling['block_scale'][d]**2
                               for d, block in enumerate(BLOCKS)]), axis=0)
    return dict(balanced_mse=float(per_row.mean()), raw_mse=float(np.mean(error**2)),
        mean_block_r2=float(np.mean([b['r2'] for b in blocks.values()])),
        mean_within_frame_r2=float(np.mean([b['within_frame_r2'] for b in blocks.values()])),
        blocks=blocks, by_temperature={str(int(t)): float(per_row[temperatures==t].mean()) for t in np.unique(temperatures)}), per_row


def ridge_predictions(features, targets, train, rows, alpha):
    scaler = StandardScaler().fit(features[train])
    ridge = Ridge(alpha=alpha).fit(scaler.transform(features[train]), targets[train])
    return ridge.predict(scaler.transform(features[rows]))


def paired_source_gain(reference, candidate, sources, seed):
    groups = np.unique(sources)
    a = np.array([reference[sources==s].mean() for s in groups])
    b = np.array([candidate[sources==s].mean() for s in groups])
    draw = np.random.default_rng(seed).integers(0, len(groups), size=(4000, len(groups)))
    gains = 1-b[draw].mean(1)/a[draw].mean(1)
    return dict(relative_mse_reduction=float(1-b.mean()/a.mean()),
        source_bootstrap_95_percent_interval=np.quantile(gains, [.025, .975]).tolist(),
        source_count=len(groups), per_source_reduction=(1-b/a).tolist(),
        limitation='Resamples whole held-out source trajectories; few test sources limit precision. Seeds are averaged first.')

