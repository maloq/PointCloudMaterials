"""Read-only hooks on the native encoder, preserving its scalar output."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.analysis.structural_adapter import _training_contraction_order
from src.data.structural_pretraining.batches import collate, move
from src.data.structural_pretraining.prepare import file_hash
from src.models.encoders.structural import StructuralGATr, ARCHITECTURE_REVISION
from src.research.trajectory_stability.encode import observation

STAGES = ('block1', 'block2_mlp_input', 'block2_output')
SECTORS = ('plane_normal', 'ideal_bivector', 'axial_bivector', 'point_numerator')
VECTOR_INDICES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


def vector_parts(mv):
    """Four SO(3) vector triplets in the pinned PGA basis; no point division.

    Only plane_normal is translation invariant without a chosen local origin.
    All inputs here are centered on the tracked atom; only proper rotations
    are tested. No arbitrary multivector is interpreted as a point or rotor.
    """
    return torch.stack((mv[..., 2:5], mv[..., 5:8],
        torch.stack((-mv[..., 10], mv[..., 9], -mv[..., 8]), -1),
        torch.stack((-mv[..., 13], mv[..., 12], -mv[..., 11]), -1)), -2)


class Capture(nn.Module):
    def __init__(self, path, expected_sha256):
        super().__init__()
        if file_hash(path) != expected_sha256:
            raise ValueError(f'Checkpoint changed: {path}')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved['architecture'], saved['input_frames'], saved['identity']['architecture_revision']) != (
                'gatr', 1, ARCHITECTURE_REVISION):
            raise ValueError('Requires the audited native snapshot GATr')
        for name, expected in saved['identity']['implementation']['files'].items():
            if name.startswith('src/models/') and file_hash(name) != expected:
                raise ValueError(f'Checkpoint implementation differs: {name}')
        _training_contraction_order()
        self.encoder = StructuralGATr()
        self.encoder.load_state_dict(saved['encoder'], strict=True)
        self.precision = saved['identity']['config']['precision']
        self.scale = saved['scales']['Al']
        self.stage_values = {}
        self.centers = None
        self.handles = [
            self.encoder.spatial[0].register_forward_hook(self._after('block1')),
            self.encoder.spatial[1].mlp.register_forward_pre_hook(self._before, with_kwargs=True),
            self.encoder.spatial[1].register_forward_hook(self._after('block2_output')),
        ]

    def _select(self, value):
        return value[torch.arange(len(value), device=value.device), self.centers]

    def _after(self, name):
        def hook(module, args, output):
            self.stage_values[name] = self._select(output[0])
        return hook

    def _before(self, module, args, kwargs):
        self.stage_values['block2_mlp_input'] = self._select(args[0])

    def forward(self, batch):
        self.centers = batch['centers']
        self.stage_values = {}
        z = self.encoder(batch)
        mv = torch.stack([self.stage_values[s] for s in STAGES], 1)
        return z, mv

    def make_batch(self, positions, centers, device='cuda:0'):
        return move(collate([observation(p, c, self.scale, 'gatr')
            for p, c in zip(positions, centers, strict=True)], 'gatr'), device)

    def evaluate(self, batch):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            z, mv = self(batch)
        if not torch.isfinite(z).all() or not torch.isfinite(mv).all():
            raise FloatingPointError('Nonfinite native GATr state or multivector')
        return z.float(), mv.float()


def geometry_baselines(local, scale):
    """Centered density dipoles and quadrupole axis, in physical coordinates."""
    from src.data.predictive_memory.targets import taper
    from src.data.structural_pretraining.prepare import REFERENCE_RADIUS
    x = local.astype(np.float64)
    r = np.linalg.norm(x, axis=-1)
    w7 = taper(r, 5., 7.)*(r > 0)
    w17 = taper(r*REFERENCE_RADIUS/scale, 15., 17.)*(r > 0)
    if min(w7.sum(), w17.sum()) <= 0:
        raise ValueError('Empty geometry baseline support')
    c7 = w7@x/w7.sum(); c17 = w17@x/w17.sum()
    shape = (x.T*w7)@x/w7.sum()
    values, vectors = np.linalg.eigh(shape)
    gap = (values[-1]-values[-2])/values.sum()
    return np.stack((c7, c17, vectors[:, -1])).astype(np.float32), float(gap)
