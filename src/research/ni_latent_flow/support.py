"""Opt-in coordinate supports for the 244 fixed Ni measurements.

Bounds follow liquid_predictability/descriptors.py, descriptors/rich.py and
alpha3d.cpp, measurement_sim/features.py, and ni_latent_sim/measurements.py.
This enforces necessary scalar constraints, not atomistic realizability: Gram
positivity, ordered quantiles, discrete Betti counts and second-moment relations
are not enforced jointly. Signed angular cross powers and rates remain signed.

Logistic/softplus maps are bijective only on open supports. Observed boundaries
are moved inward by the explicitly recorded epsilon before fitting. Generated
values are never clipped. Softplus has a linear positive tail, avoiding the
exponential growth of the historical asinh inverse for positive measurements.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.research.ni_kinetic_latent.data import inverse, transform


def bounds(columns):
    """Exact per-column necessary bounds; no empirical extrema are used."""
    lower = torch.full((len(columns),), -torch.inf)
    upper = torch.full((len(columns),), torch.inf)
    for index, name in enumerate(columns):
        if name.startswith('bond_order/'):
            suffix = name.split('/', 1)[1].split('_', 1)[1]
            if suffix in ('q', 'qbar', 'neighbor_q_mean', 'coherence_std'):
                lower[index], upper[index] = 0., 1.
            elif suffix == 'neighbor_q_std':
                # Population SD of values in [0,1], as used by order_invariants.
                lower[index], upper[index] = 0., .5
            elif suffix in ('coherence_mean', 'coherence_q10', 'coherence_q50', 'coherence_q90'):
                lower[index], upper[index] = -1., 1.
            elif suffix not in ('w', 'wbar'):
                raise ValueError('Unknown bond-order measurement: ' + name)
        elif name.startswith('tda/') or name.startswith('density/radial'):
            lower[index] = 0.
        elif name.startswith('density/l'):
            ell, left, right = name.split('_')
            # l=0 is a product of nonnegative radial sums; l>0 cross terms
            # can be negative. Every diagonal is a squared harmonic magnitude.
            if ell == 'density/l0' or left == right:
                lower[index] = 0.
        elif name.startswith('velocity/'):
            if name.endswith(('second', 'squared')):
                lower[index] = 0.
        else:
            raise ValueError('Unknown measurement family: ' + name)
    return lower, upper


class MeasurementSupport(nn.Module):
    """Old standardized244 <-> standardized unconstrained support coordinates.

    ``bank`` is the existing producer's normalization dictionary. ``fit`` accepts
    TRAIN observations only. Buffers contain both normalizations for checkpoint
    restoration; construction does not fit anything. There are no parameters.
    Epsilon is a fraction of interval width for bounded channels and of the
    pinned training raw standard deviation for nonnegative channels.
    """

    def __init__(self, bank, epsilon=1e-6, tolerance=1e-5, scale_floor=1e-4):
        super().__init__()
        self.columns = list(bank['columns'])
        if not 0 < epsilon < .5 or tolerance < 0 or scale_floor <= 0:
            raise ValueError('Invalid support epsilon, tolerance or scale floor')
        self.epsilon, self.tolerance, self.scale_floor = epsilon, tolerance, scale_floor
        lower, upper = bounds(self.columns)
        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)
        self.register_buffer('bounded', torch.where(torch.isfinite(upper))[0])
        self.register_buffer('positive', torch.where(torch.isfinite(lower) & ~torch.isfinite(upper))[0])
        for key in ('raw_mean', 'raw_scale', 'mean', 'scale'):
            self.register_buffer('old_' + key, bank[key].detach().float().clone())
        self.register_buffer('coordinate_mean', torch.zeros(len(self.columns)))
        self.register_buffer('coordinate_scale', torch.ones(len(self.columns)))
        self.register_buffer('fitted', torch.tensor(False))

    def _bank(self):
        return {key: getattr(self, 'old_' + key) for key in
                ('raw_mean', 'raw_scale', 'mean', 'scale')}

    def _raw(self, x):
        # FP64 here limits inverse-normalization roundoff around exact zeros.
        return inverse(x.double(), self._bank())

    def _violations(self, raw):
        tolerance = self.tolerance * self.old_raw_scale.to(raw).clamp_min(1.)
        return (~torch.isfinite(raw) | (raw < self.lower.to(raw) - tolerance)
                | (raw > self.upper.to(raw) + tolerance))

    def _coordinates(self, raw):
        bad = self._violations(raw)
        if bad.any():
            indices = torch.where(bad.reshape(-1, len(self.columns)).any(0))[0].tolist()
            raise ValueError('Measurements outside declared support: ' + ', '.join(self.columns[i] for i in indices))
        values = raw / self.old_raw_scale.to(raw)
        bounded = self.bounded
        fraction = (raw[..., bounded] - self.lower[bounded]) / (self.upper[bounded] - self.lower[bounded])
        # This is observed-boundary handling, never generated-sample clipping.
        fraction = fraction.clamp(self.epsilon, 1 - self.epsilon)
        values[..., bounded] = fraction.log() - torch.log1p(-fraction)
        positive = values[..., self.positive].clamp_min(self.epsilon)
        values[..., self.positive] = positive + torch.log(-torch.expm1(-positive))
        return values

    def _coordinates_to_raw(self, values):
        raw = values * self.old_raw_scale.to(values)
        bounded = self.bounded
        raw[..., bounded] = (self.lower[bounded] + (self.upper[bounded] - self.lower[bounded])
                             * torch.sigmoid(values[..., bounded]))
        raw[..., self.positive] = (F.softplus(values[..., self.positive])
                                  * self.old_raw_scale[self.positive])
        return raw

    @torch.no_grad()
    def fit(self, train_x, chunk_size=8192):
        """Streaming FP64 training moments; the caller declares training rows."""
        values = train_x.reshape(-1, len(self.columns))
        total = torch.zeros(len(self.columns), dtype=torch.float64, device=self.coordinate_mean.device)
        square = torch.zeros_like(total)
        for chunk in values.split(chunk_size):
            coordinate = self._coordinates(self._raw(chunk.to(total.device)))
            total += coordinate.sum(0)
            square += coordinate.square().sum(0)
        mean = total / len(values)
        variance = (square / len(values) - mean.square()).clamp_min(0)
        self.coordinate_mean.copy_(mean)
        self.coordinate_scale.copy_(variance.sqrt().clamp_min(self.scale_floor))
        self.fitted.fill_(True)
        return self

    def encode(self, x):
        """Transform actual observed targets, including declared boundary handling."""
        if not self.fitted:
            raise RuntimeError('Support normalization must be fitted on train first')
        coordinates = self._coordinates(self._raw(x))
        return ((coordinates - self.coordinate_mean) / self.coordinate_scale).to(x.dtype)

    def to_raw(self, coordinates):
        """Differentiable constrained measurement values, without inverse sinh."""
        # Small raw fluctuations around large means (e.g. TDA entropy) need
        # FP64 scalar algebra; neural layers/parameters remain mixed precision.
        values = coordinates.double() * self.coordinate_scale + self.coordinate_mean
        return self._coordinates_to_raw(values)

    def decode(self, coordinates):
        """Return the existing normalized244 score space, preserving old metrics."""
        return transform(self.to_raw(coordinates), self._bank()).to(coordinates.dtype)

    @torch.no_grad()
    def audit(self, x, chunk_size=8192):
        """Per-channel support/boundary and round-trip audit; fits no statistics."""
        values = x.reshape(-1, len(self.columns))
        device = self.coordinate_mean.device
        count = len(values)
        minimum = torch.full_like(self.coordinate_mean, torch.inf, dtype=torch.float64)
        maximum = torch.full_like(minimum, -torch.inf)
        bad_count = torch.zeros(len(self.columns), dtype=torch.int64, device=device)
        adjusted_count = torch.zeros_like(bad_count)
        adjustment = torch.zeros_like(minimum)
        roundtrip = torch.zeros_like(minimum)
        for chunk in values.split(chunk_size):
            chunk = chunk.to(device)
            raw = self._raw(chunk)
            minimum = torch.minimum(minimum, raw.amin(0))
            maximum = torch.maximum(maximum, raw.amax(0))
            bad_count += self._violations(raw).sum(0)
            coordinates = self._coordinates(raw)
            adjusted = self._coordinates_to_raw(coordinates)
            difference = (adjusted - raw).abs()
            bounded_fraction = ((raw[..., self.bounded] - self.lower[self.bounded]) /
                                (self.upper[self.bounded] - self.lower[self.bounded]))
            adjusted_count[self.bounded] += ((bounded_fraction < self.epsilon) |
                                             (bounded_fraction > 1 - self.epsilon)).sum(0)
            adjusted_count[self.positive] += (raw[..., self.positive] /
                                              self.old_raw_scale[self.positive] < self.epsilon).sum(0)
            adjustment = torch.maximum(adjustment, difference.amax(0))
            reconstructed = self.decode(self.encode(chunk))
            roundtrip = torch.maximum(roundtrip, (reconstructed - chunk).double().abs().amax(0))
        return dict(states=count, epsilon=self.epsilon, tolerance=self.tolerance,
            boundary_rule='Bounded: inward epsilon*width; nonnegative: epsilon*pinned train raw std; input-only',
            support='Necessary scalar constraints only; no joint atomistic feasibility claim',
            channels=[dict(column=name, lower=None if not torch.isfinite(self.lower[i]) else float(self.lower[i]),
                upper=None if not torch.isfinite(self.upper[i]) else float(self.upper[i]),
                minimum=float(minimum[i]), maximum=float(maximum[i]),
                invalid_count=int(bad_count[i]), boundary_adjusted_count=int(adjusted_count[i]),
                maximum_raw_adjustment=float(adjustment[i]), maximum_normalized_roundtrip_error=float(roundtrip[i]))
                for i, name in enumerate(self.columns)])


@torch.no_grad()
def validity_metrics(raw, columns, raw_scale, tolerance=1e-5):
    """Scalar and joint producer identities, retaining every supplied state.

    Leading state/trajectory dimensions are preserved. Coordinate diagnostics
    end in 244, density diagnostics in seven angular orders, and shell moment
    diagnostics in three shells. Nonfinite inputs are violations; their excess
    is infinite, never a dropped observation. Matrix-zero substitution below
    only permits numerical eigendecomposition of the remaining valid matrices.
    These necessary conditions do not establish atomistic realizability.
    """
    raw = raw.double()
    lower, upper = [value.to(raw) for value in bounds(columns)]
    unit = raw_scale.to(raw).clamp_min(1.)
    finite = torch.isfinite(raw)
    excess = torch.maximum((lower - raw).clamp_min(0), (raw - upper).clamp_min(0)) / unit
    excess = torch.where(finite, excess, torch.full_like(excess, torch.inf))
    result = dict(scalar_violation=(~finite) | (excess > tolerance), scalar_excess=excess)
    lookup = {name: index for index, name in enumerate(columns)}
    indices = torch.tensor([[lookup[f'density/l{ell}_{a}_{b}'] for a in range(4)
                             for b in range(a, 4)] for ell in range(7)], device=raw.device)
    powers = raw[..., indices]
    gram = torch.zeros((*raw.shape[:-1], 7, 4, 4), dtype=raw.dtype, device=raw.device)
    a, b = torch.triu_indices(4, 4, device=raw.device)
    gram[..., a, b] = powers
    gram[..., b, a] = powers
    gram_finite = torch.isfinite(gram).all((-2, -1))
    computational_gram = torch.where(gram_finite[..., None, None], gram, 0.)
    # Bound cuSOLVER workspace independently of the number of sampled paths.
    # Each 4x4 matrix is independent; chunking preserves the same PSD test.
    matrices = computational_gram.reshape(-1, 4, 4)
    minimum = torch.cat([torch.linalg.eigvalsh(chunk)[:, 0]
                         for chunk in matrices.split(1024)]).reshape(gram_finite.shape)
    gram_unit = computational_gram.abs().amax((-2, -1)).clamp_min(1.)
    psd_excess = (-minimum / gram_unit).clamp_min(0)
    psd_excess = torch.where(gram_finite, psd_excess, torch.full_like(psd_excess, torch.inf))
    result.update(density_min_eigenvalue=torch.where(gram_finite, minimum,
                                                    torch.full_like(minimum, torch.nan)),
                  density_psd_violation=(~gram_finite) | (psd_excess > tolerance),
                  density_psd_excess=psd_excess)
    radial = raw[..., [lookup[f'density/radial{i}'] for i in range(4)]]
    outer = radial[..., :, None] * radial[..., None, :]
    l0_finite = torch.isfinite(outer).all((-2, -1)) & gram_finite[..., 0]
    l0_error = (gram[..., 0, :, :] - outer).abs().amax((-2, -1)) / outer.abs().amax((-2, -1)).clamp_min(1.)
    l0_error = torch.where(l0_finite, l0_error, torch.full_like(l0_error, torch.inf))
    result.update(density_l0_identity_error=l0_error,
                  density_l0_identity_violation=(~l0_finite) | (l0_error > tolerance))
    mean = raw[..., [lookup[f'velocity/shell{i}/radial_mean'] for i in range(3)]]
    second = raw[..., [lookup[f'velocity/shell{i}/radial_second'] for i in range(3)]]
    moment_finite = torch.isfinite(mean) & torch.isfinite(second)
    margin = second - mean.square()
    moment_unit = torch.maximum(second.abs(), mean.square()).clamp_min(1.)
    moment_excess = (-margin / moment_unit).clamp_min(0)
    moment_excess = torch.where(moment_finite, moment_excess, torch.full_like(moment_excess, torch.inf))
    result.update(velocity_moment_margin=margin,
                  velocity_moment_violation=(~moment_finite) | (moment_excess > tolerance),
                  velocity_moment_excess=moment_excess)
    return result
