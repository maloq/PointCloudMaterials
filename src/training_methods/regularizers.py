"""Scale-controlled Epi scores shared by encoder training protocols."""

import math
import torch


def ridge_map(h, rho=3.0):
    with torch.no_grad():
        h = h.double()
        h = (h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(h.shape[1])
        eye = torch.eye(h.shape[1], device=h.device, dtype=h.dtype)
        q, r = torch.linalg.qr(torch.cat((h, math.sqrt(rho) * eye)), mode='reduced')
        return torch.linalg.solve_triangular(r, q[: len(h)].T, upper=True)


def epiplexity(z, h, normalize=True):
    """Blog's ridge/logdet score; common RMS normalization removes scale inflation.

    Reservoir is frozen random MACE, not an image CNN. This is an adaptation,
    not a reproduction of the blog's image-view alignment experiment.
    """
    centered = z.double() - z.double().mean(0)
    if normalize:
        centered = centered / torch.sqrt(centered.square().mean() + 1e-4)
    w = ridge_map(h) @ centered
    # Sylvester determinant identity evaluates the smaller reservoir-side matrix.
    matrix = torch.eye(w.shape[0], device=w.device, dtype=w.dtype) + 30 * (w @ w.T)
    factor = torch.linalg.cholesky(matrix)
    return (torch.log(torch.diagonal(factor)).sum() / math.log(2)).float()
