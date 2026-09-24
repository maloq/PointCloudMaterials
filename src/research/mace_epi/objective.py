"""Full-batch two-view objectives on the exported invariant state itself."""
import torch
from torch import nn

from src.training_methods.neighborhood_jepa.regularization.objective import epiplexity
from src.training_methods.structural_pretraining.objective import vicreg


class Objective(nn.Module):
    def __init__(self, treatment, epi_weight):
        super().__init__()
        if treatment not in ('vicreg', 'epi', 'epi-variance'):
            raise ValueError(f'Unknown paired-MACE treatment: {treatment}')
        self.treatment = treatment
        self.epi_weight = epi_weight
        self.register_buffer('epi_initial_scale', torch.tensor(1.))
        self.diagnostics = {}

    def forward(self, model, encoded, target):
        # The native packer interleaves [current, next] within each anchor.
        n = len(target['index'])
        if len(encoded) != 2*n or n < 2:
            raise ValueError(f'Expected two views for each of {n} anchors: {encoded.shape}')
        z = encoded[:, :128].float().reshape(n, 2, 128)
        _, details = vicreg(z[:, 0], z[:, 1])
        terms = {'alignment': 25/51 * details['invariance']}
        if self.treatment in ('vicreg', 'epi-variance'):
            terms['variance'] = 25/51 * details['variance']
        if self.treatment == 'vicreg':
            terms['covariance'] = details['covariance']/51
        else:
            if target['reservoir'].shape != (n, 2, 64):
                raise ValueError(f'Wrong frozen reservoir shape: {target["reservoir"].shape}')
            score = torch.stack([epiplexity(z[:, i], target['reservoir'][:, i]) for i in (0, 1)]).mean()
            terms['epi'] = -self.epi_weight * score / self.epi_initial_scale
            details['epi_score'] = score
        self.diagnostics = {k: float(v.detach()) for k, v in details.items()}
        self.diagnostics['export_rms'] = float(z.detach().square().mean().sqrt())
        self.diagnostics['epi_initial_scale'] = float(self.epi_initial_scale)
        return sum(terms.values()), terms
