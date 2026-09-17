"""Version 1: 128 continuous geometric/motion features on fixed 5--7 A support.

No PTM, crystallization, topology, basin, or future-derived membership is used.
"""
import numpy as np
from scipy.special import eval_legendre

BLOCKS = {'radial': (0, 32), 'pair': (32, 64), 'angular': (64, 80),
          'speed': (80, 96), 'radial_velocity': (96, 112), 'moments': (112, 128)}


def taper(x, inner, outer):
    a = np.clip((x-inner)/(outer-inner), 0, 1)
    return (1-a)**3*(1+3*a+6*a*a)


def rbf(values, weights, low, high, bins):
    centers = np.linspace(low, high, bins)
    width = (high-low)/(bins-1)
    return (weights[:, None]*np.exp(-.5*((values[:, None]-centers)/width)**2)).sum(0)/weights.sum()


def physical_packet(relative_positions, relative_velocities):
    x, u = np.asarray(relative_positions, dtype=np.float64), np.asarray(relative_velocities, dtype=np.float64)
    radius = np.linalg.norm(x, axis=-1)
    keep = (radius > 0) & (radius < 7)
    x, u, radius = x[keep], u[keep], radius[keep]
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(u).all():
        raise ValueError('Physical target requires at least two finite neighbors within 7 A')
    support = taper(radius, 5., 7.)
    w = support/support.sum()
    direction = x/radius[:, None]
    speed2 = np.sum(u*u, -1)
    radial_v = np.sum(direction*u, -1)
    a, b = np.triu_indices(len(x), 1)
    pair_weight = support[a]*support[b]
    pair_distance = np.linalg.norm(x[a]-x[b], axis=-1)
    cosine = np.clip(np.sum(direction[a]*direction[b], -1), -1, 1)
    angular = np.array([np.dot(pair_weight, eval_legendre(l, cosine))/pair_weight.sum()
                        for l in range(1, 17)])
    # Ridge makes this local affine fit continuous even near degenerate geometry.
    affine = np.linalg.solve(x.T@(w[:, None]*x)+1e-3*np.eye(3), x.T@(w[:, None]*u))
    strain = (affine+affine.T)/2
    deviator = strain-np.trace(strain)*np.eye(3)/3
    rotation = (affine-affine.T)/2
    mean_x, mean_u = w@x, w@u
    moments = [support.sum(), taper(radius, 0, 3).sum(), w@radius, w@radius**2,
               w@radius**3, w@speed2, w@speed2**2, w@radial_v, w@radial_v**2,
               mean_u@mean_u, mean_x@mean_u, np.trace(affine), np.sum(deviator**2),
               np.sum(rotation**2), w@np.sum((u-x@affine)**2, -1), w@radial_v**3]
    result = np.concatenate((rbf(radius, support, 0, 7, 32),
        rbf(pair_distance, pair_weight, 0, 14, 32), angular,
        rbf(np.sqrt(speed2), support, 0, 24, 16), rbf(radial_v, support, -16, 16, 16), moments))
    if result.shape != (128,) or not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite 128-dimensional physical target')
    return result.astype(np.float32)
