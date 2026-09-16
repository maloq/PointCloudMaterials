"""Shared inference and exact encoder-gradient replay for variable-size graphs."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from src.models.encoders.mace_context import context_features, make_context_graph
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


def load_model(config):
    torch.set_num_threads(config['cpu_threads'])
    torch.cuda.set_device(config['device'])
    torch.manual_seed(config['seed'])
    directory, name = resolve_config_path(config['checkpoint'])
    cfg = OmegaConf.load(Path(directory)/f'{name}.yaml')
    cfg.encoder.kwargs.performance.compile_radial_mlp = False
    cfg.encoder.kwargs.activation_checkpointing = False
    model = load_model_from_checkpoint(config['checkpoint'], cfg, device=config['device'], module=VICRegModule)
    if model.tda_head is not None or cfg.encoder.name != 'PretrainedMACEGeometry':
        raise ValueError('Context experiment requires the unsupervised snapshot forecast encoder')
    return model, cfg


def graph(config, clouds, mode):
    return make_context_graph(clouds, mode, device=config['device'],
            inner=config['inner_radius_A'], outer=config['outer_radius_A'])


def encode(config, model, clouds, mode):
    pieces = []
    with torch.no_grad():
        for start in range(0, len(clouds), config['micro_batch_size']):
            g = graph(config, clouds[start:start+config['micro_batch_size']], mode)
            pieces.append(context_features(model.encoder.mace, g))
    z = torch.cat(pieces)
    if z.shape != (len(clouds), 256) or not torch.isfinite(z).all():
        raise FloatingPointError(f'Invalid context embedding: {mode}, {z.shape}')
    return z


def augment(config, model, views, mode):
    """Use the actual VICReg mirror and jitter settings, without its 80-node crop."""
    clouds = []
    radius = model.encoder.reference_radius_A
    for view in views:
        # Draw the same full-candidate augmentation for all modes, including
        # the 80-atom baseline, so common atoms receive identical perturbations.
        original = view
        points = torch.from_numpy(original).to(config['device'])[None]/radius
        points = model.vicreg.apply_view_postprocessing(points, use_neighbor=False,
                   apply_occlusion=False, view_points=None)[0]*radius
        # Mirroring leaves radii unchanged; track the realized radial movement.
        movement = max(float(torch.linalg.vector_norm(points[0])),
                       float(np.max(np.abs(torch.linalg.vector_norm(points-points[0], dim=1).cpu().numpy()
                                 -np.linalg.norm(original-original[0], axis=1)))))
        margin = config['candidate_radius_A']-config['outer_radius_A']-10.
        if mode != 'mean80' and movement >= margin:
            raise ValueError(f'Augmentation exceeds complete-halo margin: {movement} >= {margin} A')
        clouds.append((points[:80] if mode == 'mean80' else points).cpu().numpy())
    return clouds


def loss_from_features(model, z):
    features = model._shared_invariant(z, None).chunk(3, dim=0)
    return model.vicreg.compute_spatiotemporal_loss(features=features,
                                                  temporal_weight=model.temporal_weight)


def replay_step(config, model, clouds, mode):
    """Full-batch VICReg/projector; replay only the BatchNorm-free MACE encoder."""
    z = encode(config, model, clouds, mode).detach().requires_grad_(True)
    loss, metrics, _ = loss_from_features(model, z)
    loss.backward()
    for start in range(0, len(clouds), config['micro_batch_size']):
        g = graph(config, clouds[start:start+config['micro_batch_size']], mode)
        actual = context_features(model.encoder.mace, g)
        actual.backward(z.grad[start:start+len(actual)])
    return float(loss.detach()), {key:float(value.detach()) for key,value in metrics.items()}
