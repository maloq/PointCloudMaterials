"""Compiled float64 reductions for local rich spatial-training targets.

These kernels retain the original complex spherical-harmonic convention and
finite alpha-persistence intervals. No fast-math, filtration truncation or
neighborhood approximation is used.
"""
from functools import lru_cache
from math import factorial

import numpy as np
from numba import njit


@lru_cache(None)
def harmonic_normalization():
    out = np.zeros((9, 9))
    for ell in range(9):
        for m in range(ell + 1):
            out[ell, m] = np.sqrt((2 * ell + 1) / (4 * np.pi)
                                  * factorial(ell - m) / factorial(ell + m))
    return out


@njit(cache=True, nogil=True)
def harmonics(vectors, normalization):
    """Average Y_lm, l=2,4,6,8, sharing the Legendre recurrence across orders."""
    q = np.zeros((4, len(vectors), 17), np.complex128)
    for i in range(len(vectors)):
        for j in range(vectors.shape[1]):
            x, y, z = vectors[i, j]
            radius = np.sqrt(x*x + y*y + z*z)
            cosine = min(1., max(-1., z / radius))
            sine = np.sqrt(max(0., 1. - cosine*cosine))
            phi = np.arctan2(y, x)
            phase = complex(np.cos(phi), np.sin(phi))
            phase_m = complex(1., 0.)
            pmm = 1.
            for m in range(9):
                if m:
                    pmm *= -(2*m - 1)*sine
                    phase_m *= phase
                previous, current = 0., pmm
                for ell in range(m, 9):
                    if ell > m:
                        following = ((2*ell - 1)*cosine*current
                                     - (ell + m - 1)*previous) / (ell - m)
                        previous, current = current, following
                    if ell >= 2 and ell % 2 == 0:
                        k = ell // 2 - 1
                        value = normalization[ell, m]*current*phase_m
                        q[k, i, ell + m] += value
                        if m:
                            q[k, i, ell - m] += (-1 if m % 2 else 1)*value.conjugate()
    return q / vectors.shape[1]


@njit(cache=True, nogil=True)
def order_invariants(q, ix4, weight4, ix6, weight6):
    out = np.empty(40)
    offset = 0
    for k in range(4):
        ell = 2*(k + 1)
        field = q[k, :, :2*ell + 1]
        norm = np.sqrt((field.real**2 + field.imag**2).sum(axis=1))
        mean = np.zeros(2*ell + 1, np.complex128)
        for i in range(len(field)):
            mean += field[i] / len(field)
        mean_norm = np.sqrt((mean.real**2 + mean.imag**2).sum())
        factor = np.sqrt(4*np.pi/(2*ell + 1))
        out[offset:offset + 4] = np.array([factor*norm[0], factor*mean_norm,
                                          factor*norm.mean(), factor*norm.std()])
        offset += 4
        if ell == 4 or ell == 6:
            indices, weights = (ix4, weight4) if ell == 4 else (ix6, weight6)
            for which in range(2):
                values = field[0] if which == 0 else mean
                magnitude = norm[0] if which == 0 else mean_norm
                total = complex(0., 0.)
                for j in range(len(weights)):
                    a, b, c = indices[j]
                    total += values[a]*values[b]*values[c]*weights[j]
                out[offset] = total.real / max(magnitude**3, 1e-14)
                offset += 1
        coherence = np.zeros(len(field) - 1)
        for i in range(1, len(field)):
            for m in range(2*ell + 1):
                coherence[i-1] += (field[0, m].conjugate()*field[i, m]).real
            coherence[i-1] /= max(norm[0], 1e-14)*max(norm[i], 1e-14)
        out[offset], out[offset + 1] = coherence.mean(), coherence.std()
        ordered = np.sort(coherence)
        for j, probability in enumerate((.1, .5, .9)):
            at = (len(ordered) - 1)*probability
            lo, hi = int(np.floor(at)), int(np.ceil(at))
            out[offset + 2 + j] = ordered[lo] + (ordered[hi] - ordered[lo])*(at - lo)
        offset += 5
    return out


@njit(cache=True, nogil=True)
def persistence_packet(pairs, dim, norm):
    """Original log moments, entropy, Gaussian images and smooth Betti curves."""
    out = np.zeros(27 if dim == 0 else 51)
    finite = pairs[np.isfinite(pairs[:, 1])]
    radii = np.sqrt(np.maximum(finite, 0.))
    life = radii[:, 1] - radii[:, 0]
    total_life = max(life.sum(), 1e-14)
    spectrum_size = 12 if dim == 0 else 36
    for i in range(len(radii)):
        birth, death = radii[i]
        lifetime = life[i]
        loglife = np.log1p(lifetime)
        out[1] += loglife / norm
        out[2] += loglife*loglife / norm
        out[3] = max(out[3], loglife)
        p = lifetime / total_life
        if p > 0:
            out[4] -= p*np.log(p)
        out[5] += np.log1p(birth) / norm
        out[6] += np.log1p(death) / norm
        if dim == 0:
            for j in range(12):
                out[7+j] += np.exp(-.5*((death - (.6 + 2.4*j/11))/.15)**2) / norm
        else:
            birth_kernel = np.empty(6)
            life_kernel = np.empty(6)
            for j in range(6):
                birth_kernel[j] = np.exp(-.5*((birth - (.6 + 3.4*j/5))/.25)**2)
                life_kernel[j] = np.exp(-.5*((lifetime - 1.5*j/5)/.15)**2)
            weight = np.tanh(lifetime)/norm
            for j in range(6):
                for k in range(6):
                    out[7 + 6*j + k] += birth_kernel[j]*life_kernel[k]*weight
        for j in range(8):
            alpha = .75 + .5*j
            enter = .5*(1 + np.tanh((alpha - birth)/.1))
            leave = .5*(1 + np.tanh((death - alpha)/.1))
            out[7 + spectrum_size + j] += enter*leave/norm
    out[0] = len(radii)/norm
    return out
