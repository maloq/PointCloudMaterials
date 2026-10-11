"""Autonomous latent dynamics, with m = 0.1 ps * D E(Q)[V].

No time, temperature, age, species or physical reconstruction inputs/targets.
The structural graph is the established native local80 MACE graph. The local
atom list is fixed during each short simulator sensitivity observation.
"""
import math
import torch
from torch import nn
from src.research.local_response.model import Predictor as LocalGeometry


class Geometry(LocalGeometry):
    def __init__(self, config):
        super().__init__(config)
        del self.head

    def forward(self, q):
        return self.encode(q)


class MotionModel(nn.Module):
    def __init__(self, config, arm):
        super().__init__()
        self.geometry = Geometry(config)
        self.arm = arm
        self.time_scale = config['model']['time_scale_ps']
        self.step_ps = config['model']['step_ps']
        d = config['encoder']['code_dim']
        hidden = config['model']['hidden']
        self.d = d
        # Identical parameterization for discrete and continuous comparisons.
        self.acceleration = nn.Sequential(nn.Linear(2*d, hidden), nn.SiLU(), nn.Linear(hidden, d))
        self.diffusion = nn.Sequential(nn.Linear(2*d, hidden), nn.SiLU(), nn.Linear(hidden, d))
        self.readout = nn.Sequential(nn.Linear(2*d, hidden), nn.SiLU(), nn.Linear(hidden, 48))
        nn.init.normal_(self.acceleration[-1].weight, std=.001)
        nn.init.zeros_(self.acceleration[-1].bias)
        nn.init.normal_(self.diffusion[-1].weight, std=.001)
        nn.init.constant_(self.diffusion[-1].bias, -3.)
        nn.init.normal_(self.readout[-1].weight, std=.001)
        nn.init.zeros_(self.readout[-1].bias)

    def encode(self, state, *, create_graph):
        if state.shape[1:] != (80, 6):
            raise ValueError(f'Expected local80 Q,V in A and A/ps: {state.shape}')
        micro=getattr(self,'inference_microbatch',len(state))
        if not create_graph and len(state)>micro:
            return torch.cat([self.encode(x,create_graph=False) for x in state.split(micro)])
        q, v = state[..., :3], state[..., 3:]
        # Center subtraction is inside geometry.graph, so this also subtracts
        # the center velocity. autograd.functional.jvp supports higher grads.
        with torch.enable_grad():
            s, m = torch.autograd.functional.jvp(self.geometry, q, v * self.time_scale,
                                                create_graph=create_graph, strict=True)
        return torch.cat((s, m), -1)

    def coefficients(self, z):
        s, m = z.split(self.d, -1)
        # A bounded residual and positive damping avoid an unconstrained
        # unstable initialization. These are learned effective dynamics.
        a = 2 * torch.tanh(self.acceleration(z)) - m
        b = .001 + .5 * torch.sigmoid(self.diffusion(z))
        return m, a, b

    def advance(self, z, dt_ps, noise):
        dt = dt_ps / self.time_scale
        m, a, b = self.coefficients(z)
        s = z[..., :self.d]
        # Kinetic Euler scheme: only latent velocity receives Brownian noise.
        # The dt^1.5 position term consistently uses that same increment.
        kick = b * math.sqrt(dt) * noise
        return torch.cat((s + dt*m + .5*dt*(dt*a + kick), m + dt*a + kick), -1)

    def path(self, z, horizons, *, particles, seed, step_ps=None, deterministic=False,
             random_batch_size=None, sample_offset=0):
        """Numerically evaluate an autonomous rule at requested elapsed times.

        Horizons organize integration only; they never enter a neural layer.
        The discrete baseline supports multiples of its fixed 0.1 ps step.
        """
        horizons = [float(h) for h in horizons]
        if not horizons or any(h <= 0 for h in horizons) or horizons != sorted(set(horizons)):
            raise ValueError('Require positive increasing horizons')
        step = .1 if self.arm == 'discrete' else (self.step_ps if step_ps is None else step_ps)
        if self.arm == 'discrete' and any(abs(h/.1-round(h/.1)) > 1e-7 for h in horizons):
            raise ValueError('Discrete baseline has no sub-0.1 ps prediction')
        rng = torch.Generator(device=z.device).manual_seed(int(seed))
        full_batch=len(z) if random_batch_size is None else random_batch_size
        if sample_offset<0 or sample_offset+len(z)>full_batch:
            raise ValueError('Invalid slice of the effective-batch noise stream')
        z = z.unsqueeze(1).expand(-1, particles, -1)
        clock = 0.
        result = []
        for end in horizons:
            while clock < end - 1e-9:
                dt = min(step, end-clock)
                eps = torch.zeros_like(z[..., :self.d]) if deterministic else torch.randn(
                    (full_batch,particles,self.d), generator=rng, dtype=z.dtype, device=z.device
                    )[sample_offset:sample_offset+len(z)]
                z = self.advance(z, dt, eps)
                clock += dt
            result.append(z)
        return torch.stack(result, 2)  # sample, particle, horizon, state

    def prediction(self, path):
        mean, raw = self.readout(path).split(24, -1)
        logvar = -5. + 8. * torch.sigmoid(raw)
        return mean, logvar


def mixture_nll(mean, logvar, target):
    """Joint 24-D diagonal-Gaussian mixture NLL, divided by dimension."""
    logp = -.5 * ((target[:, None]-mean).square()*torch.exp(-logvar) + logvar + math.log(2*math.pi))
    return -(torch.logsumexp(logp.sum(-1), 1)-math.log(mean.shape[1])) / mean.shape[-1]


def noncollapse(s):
    std = (s.var(0, unbiased=False)+1e-4).sqrt()
    variance = torch.relu(1-std).mean()
    centered = s-s.mean(0)
    cov = centered.T @ centered / max(1, len(s)-1)
    cov = cov-torch.diag_embed(cov.diagonal())
    return variance + .01*cov.square().mean()
