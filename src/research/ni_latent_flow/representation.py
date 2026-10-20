"""Label-free temporal representations of the measured Ni state.

SelfTICA uses Eq. 4 of https://arxiv.org/html/2606.15495v3, with a shared
encoder and a learned, bias-free linear predictor. There is no spectral
gradient, feature normalization, target detachment, or reconstruction loss.

FMRC uses the forward PLUS retrospective conditional vector-field regression
in Eqs. 20--21 of https://arxiv.org/html/2408.17139v2. The authors' implementation
is https://github.com/Mingyuan00/Flow_Matching_for_RC/blob/main/1-CLN025/FMRC_training.ipynb.
The Gaussian prior is fitted on training observations, path noise has SD .001,
and the flow heads use three 256-wide ReLU layers. Each directional loss sums
squared errors over coordinates and averages over samples, exactly as published.
``fmrc_bidirectional`` is the main objective; ``fmrc_forward`` is a named ablation.

Our explicit domain/protocol adaptations are the existing matched-capacity
two-layer SiLU encoder, 32 latents, train-normalized central244 measurement
targets (including velocity measurements), no optional TICA preprocessing,
lag-indexed heads, source-separated train/test, and a fixed external budget.
With regional331 inputs, the target remains central244: the objective preserves
conditional central-state information, not the entire regional state. Reverse
conditioning means inferring earlier observations, not time-reversing MD or
reversing velocities. Nonstationary, partially observed Ni data do not establish
the paper's reversible/Markov spectral guarantees. There is no detailed-balance
constraint, equilibrium reweighting, or static reconstruction objective.

Every configured physical lag has its own auxiliary head. The scalar ``s``
inside a flow head is a dimensionless noise-to-data interpolation coordinate,
not physical time. The caller chooses native train pairs at that lag, controls
the random seed/checkpoint RNG state, and freezes the encoder before fitting a
fresh diagnostic decoder. Auxiliary heads are never the autonomous propagator.
"""
from itertools import chain

import torch
from torch import nn

from src.research.ni_kinetic_latent.model import Model


MEASUREMENT_DIMENSION = 244


class ConditionalMeasurementFlow(nn.Module):
    """Vector field conditioned only on encoded state and algorithmic flow s."""

    def __init__(self, latent_dimension, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(MEASUREMENT_DIMENSION + latent_dimension + 1, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, MEASUREMENT_DIMENSION))

    def forward(self, interpolated, s, latent):
        return self.network(torch.cat((interpolated, s, latent), -1))


def attach_heads(model, settings):
    """Register auxiliary modules before constructing an optimizer or loading.

    Required setting: ``lags`` (native-frame integer offsets). Optional flow
    settings: ``flow_width`` (256) and ``flow_path_sigma`` (0.001). Fit the
    empirical Gaussian prior with ``fit_flow_prior`` before fresh FMRC training.
    """
    keys = [f'lag{lag}' for lag in settings['lags']]
    if model.kind == 'selftica':
        heads = {'predictor': nn.ModuleDict({
            key: nn.Linear(model.dimension, model.dimension, bias=False)
            for key in keys})}
    elif model.kind in ('fmrc_forward', 'fmrc_bidirectional'):
        directions = ('future', 'past') if model.kind == 'fmrc_bidirectional' else ('future',)
        heads = {direction: nn.ModuleDict({
            key: ConditionalMeasurementFlow(model.dimension,
                                            settings.get('flow_width', 256))
            for key in keys}) for direction in directions}
        model.register_buffer('flow_prior_mean', torch.zeros(MEASUREMENT_DIMENSION))
        model.register_buffer('flow_prior_scale', torch.ones(MEASUREMENT_DIMENSION))
        model.register_buffer('flow_prior_fitted', torch.tensor(False))
    else:
        raise ValueError(f'Unknown temporal representation objective: {model.kind}')
    model.representation_heads = nn.ModuleDict(heads)
    return model


def make_model(c, kind, input_dimension=MEASUREMENT_DIMENSION):
    """Existing 244/331-input encoder and central244 decoder, plus train heads."""
    return attach_heads(Model(c, kind, input_dimension=input_dimension),
                        c['representation'])


def representation_parameters(model):
    """Optimize only the encoder and auxiliary heads, never the decoder."""
    return chain(model.encoder.parameters(), model.location.parameters(),
                 model.representation_heads.parameters())


@torch.no_grad()
def fit_flow_prior(model, training_targets):
    """Fit the published diagonal Gaussian prior on supplied TRAIN observations.

    Accept [states,244] or [sequences,frames,244]; all supplied observations
    contribute equally. Population standard deviation matches the authors'
    NumPy estimator. Constant measurements retain zero prior variance; the
    separate .001 path noise is still applied. Moments are checkpoint buffers.
    """
    values = training_targets.reshape(-1, MEASUREMENT_DIMENSION).double()
    scale, mean = torch.std_mean(values, dim=0, unbiased=False)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise FloatingPointError('Nonfinite training measurements in FMRC Gaussian prior')
    model.flow_prior_mean.copy_(mean)
    model.flow_prior_scale.copy_(scale)
    model.flow_prior_fitted.fill_(True)


def _selftica(model, present_latent, future_latent, key):
    # Evaluate the contrastive fourth-order reduction in float32 under AMP.
    # Masking the diagonal avoids subtracting nearly equal squared sums.
    with torch.autocast(device_type=present_latent.device.type, enabled=False):
        predicted = model.representation_heads['predictor'][key](present_latent.float())
        similarities = predicted @ future_latent.float().T
        positive = similarities.diagonal().mean()
        off_diagonal = similarities.square()
        off_diagonal.fill_diagonal_(0)
        n = len(present_latent)
        if n < 2:
            raise ValueError('SelfTICA contrastive minibatches require at least two pairs')
        negative = off_diagonal.sum() / (n * (n - 1))
        loss = negative - 2 * positive
    return loss, dict(selftica_positive=positive, selftica_negative=negative)


def _flow_loss(model, head, latent, target, s, settings):
    """Independent Gaussian endpoint coupling and constant-width noisy path."""
    target = target.float()
    noise = model.flow_prior_mean + model.flow_prior_scale * torch.randn_like(target)
    sigma = settings.get('flow_path_sigma', .001)
    interpolated = (1 - s) * noise + s * target
    if sigma:
        interpolated = interpolated + sigma * torch.randn_like(target)
    velocity = target - noise
    predicted = head(interpolated, s, latent)
    squared_error = (predicted.float() - velocity).square()
    loss = squared_error.sum(-1).mean()
    zero_mse = velocity.square().sum(-1).mean()
    return loss, zero_mse


def representation_loss(model, present, future, settings):
    """Return scalar loss and detached tensor metrics for one native-lag batch.

    ``present`` and ``future`` are [B,244] or [B,331] train-normalized inputs.
    Their first 244 columns are the central measurement state. ``settings``
    must include the currently sampled ``lag``, one of the model's configured
    ``lags``. Pair construction, source ancestry, and lag validity belong to
    the existing data/training producer. No birth or phase labels are read.
    """
    key = f"lag{settings['lag']}"
    latent = model.encode(present)
    if model.kind == 'selftica':
        loss, metrics = _selftica(model, latent, model.encode(future), key)
    elif model.kind in ('fmrc_forward', 'fmrc_bidirectional'):
        if not model.flow_prior_fitted:
            raise ValueError('Fit the FMRC Gaussian prior on TRAIN targets before training')
        s = torch.rand((len(present), 1), device=present.device, dtype=torch.float32)
        loss, zero_mse = _flow_loss(model, model.representation_heads['future'][key],
                                    latent, future[:, :MEASUREMENT_DIMENSION], s, settings)
        metrics = dict(flow_forward_squared_error=loss, flow_forward_zero_squared_error=zero_mse)
        if model.kind == 'fmrc_bidirectional':
            reverse, reverse_zero = _flow_loss(model, model.representation_heads['past'][key],
                                               model.encode(future),
                                               present[:, :MEASUREMENT_DIMENSION], s, settings)
            loss = loss + reverse
            metrics.update(flow_reverse_squared_error=reverse,
                           flow_reverse_zero_squared_error=reverse_zero)
    else:
        raise ValueError(f'Unknown temporal representation objective: {model.kind}')
    metrics.update(representation_loss=loss,
                   latent_variance=latent.float().var(0, unbiased=False).mean(),
                   latent_mean_square=latent.float().square().mean())
    return loss, {key: value.detach() for key, value in metrics.items()}
