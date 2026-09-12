"""Training-only corruption of observed embeddings, leaving future targets intact."""

import torch


def augment_history(history, settings):
    """Jitter past observations and simulate missing intermediate measurements.

    Missing frames carry the preceding available observation forward. The oldest
    observation and the clean anchor cannot be dropped; the anchor is never jittered.
    Noise is expressed in the training-standardized embedding coordinates.
    """
    augmented = history.clone()
    augmented[:, :-1] += settings['noise_std'] * torch.randn_like(augmented[:, :-1])
    missing = torch.rand(history.shape[:2], device=history.device) < settings['frame_dropout']
    for frame in range(1, history.shape[1] - 1):
        augmented[:, frame] = torch.where(missing[:, frame, None],
                                         augmented[:, frame - 1], augmented[:, frame])
    return augmented
