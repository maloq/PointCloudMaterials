"""Training-only transforms and balanced losses for the repository 144D TDA target."""

import numpy as np
from sklearn.decomposition import PCA
import torch


BLOCKS = (slice(0, 16), slice(16, 80), slice(80, 144))


def fit_targets(targets, components, floor_fraction):
    pca = PCA(n_components=components, svd_solver='full').fit(targets)
    block_std = np.array([np.sqrt(targets[:, block].var(0).mean()) for block in BLOCKS])
    floor = floor_fraction * block_std.max()
    block_scale = np.sqrt(block_std**2 + floor**2)
    pixel_scale = np.concatenate([np.full(block.stop-block.start, scale) for block, scale in zip(BLOCKS, block_scale)])
    return dict(tda_mean=pca.mean_.astype(np.float32), tda_components=pca.components_.astype(np.float32),
        tda_std=np.maximum(np.sqrt(pca.explained_variance_), 1e-5).astype(np.float32),
        pixel_scale=pixel_scale.astype(np.float32), block_std=block_std, block_scale=block_scale,
        pca_variance_ratio=pca.explained_variance_ratio_)


def transform_target(target, scaling, kind):
    centered = target - scaling['tda_mean']
    if kind == 'pca':
        return (centered @ scaling['tda_components'].T) / scaling['tda_std']
    if kind == 'blocks':
        return centered / scaling['pixel_scale']
    raise ValueError(f'Unknown topology target transform: {kind}')


def raw_prediction(prediction, scaling, kind):
    if kind == 'pca':
        return (prediction * scaling['tda_std']) @ scaling['tda_components'] + scaling['tda_mean']
    if kind == 'blocks':
        return prediction * scaling['pixel_scale'] + scaling['tda_mean']
    raise ValueError(f'Unknown topology target transform: {kind}')


def topology_loss(prediction, target, kind):
    errors = (prediction-target).square()
    if kind == 'pca':
        return errors.mean()
    return torch.stack([errors[:, block].mean() for block in BLOCKS]).mean()

