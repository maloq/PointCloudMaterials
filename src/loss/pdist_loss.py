import torch
import math
from typing import Tuple, Optional

# ----------------------------
# Utilities
# ----------------------------

def _upper_tri_mask(n: int, device, dtype=torch.bool):
    m = torch.ones((n, n), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1).to(torch.bool)  # i < j

def _soft_histogram(values: torch.Tensor,
                    num_bins: int,
                    vmin: torch.Tensor,
                    vmax: torch.Tensor,
                    sigma: Optional[torch.Tensor],
                    normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Smooth (Gaussian) histogram per batch.

    values: (B, M)
    vmin, vmax: scalar or (B,) tensors (per-batch range)
    sigma: scalar or (B,) bandwidth
    returns: (hist[B, num_bins], bin_centers[B, num_bins])
    """
    assert values.ndim == 2, "values should be (B, M)"
    B = values.shape[0]
    device = values.device
    dtype = values.dtype

    # Ensure (B,) tensors for vmin/vmax
    if not torch.is_tensor(vmin):
        vmin = torch.tensor([vmin]*B, device=device, dtype=dtype)
    if not torch.is_tensor(vmax):
        vmax = torch.tensor([vmax]*B, device=device, dtype=dtype)
    if vmin.ndim == 0:
        vmin = vmin.expand(B)
    if vmax.ndim == 0:
        vmax = vmax.expand(B)

    # Create bin centers: (B, num_bins)
    t = torch.linspace(0, 1, num_bins, device=device, dtype=dtype).view(1, num_bins)
    rnge = (vmax - vmin).view(B, 1)
    bin_centers = vmin.view(B, 1) + rnge * t  # (B, num_bins)

    # Prepare sigma with broadcast shape (B, 1, 1)
    if sigma is None:
        sigma = 0.02 * rnge.squeeze(1).clamp_min(1e-6)  # (B,)
    elif not torch.is_tensor(sigma):
        sigma = torch.tensor([sigma]*B, device=device, dtype=dtype)  # (B,)
    if sigma.ndim == 0:
        sigma = sigma.expand(B)
    sigma_b = sigma.view(B, 1, 1)  # (B,1,1)

    # Compute weights: (B, M, num_bins)
    diffs = values.unsqueeze(-1) - bin_centers.unsqueeze(1)  # (B, M, num_bins)
    weights = torch.exp(-0.5 * (diffs / sigma_b) ** 2)

    # Sum across samples -> (B, num_bins)
    hist = weights.sum(dim=1)

    if normalize:
        hist = hist / (hist.sum(dim=-1, keepdim=True) + 1e-12)

    return hist, bin_centers


def _pairwise_dists(x: torch.Tensor) -> torch.Tensor:
    """
    Batched intra-set distances.
    x: (B, N, 3)
    returns: D (B, N, N)
    """
    # torch.cdist is stable and differentiable
    return torch.cdist(x, x, p=2)


def _gather_upper_tri(dmat: torch.Tensor) -> torch.Tensor:
    """
    Extract upper-triangular (i<j) entries per batch and flatten.
    dmat: (B, N, N)
    returns: (B, P) where P = N*(N-1)/2
    """
    B, N, _ = dmat.shape
    mask = _upper_tri_mask(N, dmat.device)
    # (B, P)
    return dmat[:, mask].contiguous()


# ----------------------------
# 1) Pairwise distance loss
# ----------------------------

def pairwise_distance_loss(x: torch.Tensor,
                           y: torch.Tensor,
                           mode: str = "sorted",
                           hist_bins: int = 64,
                           hist_sigma: Optional[float] = None,
                           reduction: str = "mean") -> torch.Tensor:
    """
    Permutation-invariant distance-structure loss between two point clouds.

    x, y: (B, N, 3)
    mode:
      - "sorted": sort all i<j distances and L2 between sorted vectors
      - "hist":   compare smooth histograms of distances
    hist_bins: number of bins if mode="hist"
    hist_sigma: Gaussian bandwidth for soft hist; default=2% of max range
    """
    assert x.ndim == 3 and y.ndim == 3 and x.shape == y.shape
    B, N, _ = x.shape
    Dx = _pairwise_dists(x)  # (B,N,N)
    Dy = _pairwise_dists(y)

    dx = _gather_upper_tri(Dx)  # (B,P)
    dy = _gather_upper_tri(Dy)

    if mode == "sorted":
        dx_sorted, _ = torch.sort(dx, dim=-1)
        dy_sorted, _ = torch.sort(dy, dim=-1)
        loss_vec = (dx_sorted - dy_sorted).pow(2).mean(dim=-1)  # (B,)
    elif mode == "hist":
        # per-batch vmin/vmax
        vmin = torch.minimum(dx.min(dim=-1).values, dy.min(dim=-1).values)
        vmax = torch.maximum(dx.max(dim=-1).values, dy.max(dim=-1).values)
        # pad a tiny margin
        pad = 1e-6 + 0.01 * (vmax - vmin)
        vmin = vmin - pad
        vmax = vmax + pad
        # sigma default: 2% of range
        if hist_sigma is None:
            hist_sigma = 0.02 * (vmax - vmin).clamp(min=1e-6)
        # Broadcast friendly
        if isinstance(hist_sigma, torch.Tensor) and hist_sigma.ndim == 1:
            # (B,) -> (B,1) via autograd-friendly add
            hist_sigma_use = hist_sigma
        else:
            # scalar float -> tensor
            hist_sigma_use = (0.02 * (vmax - vmin)).clamp(min=1e-6)

        hx, _ = _soft_histogram(dx, hist_bins, vmin, vmax, sigma=hist_sigma_use, normalize=True)
        hy, _ = _soft_histogram(dy, hist_bins, vmin, vmax, sigma=hist_sigma_use, normalize=True)
        loss_vec = (hx - hy).pow(2).mean(dim=-1)  # (B,)
    else:
        raise ValueError("mode must be 'sorted' or 'hist'")

    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec  # (B,)


# ----------------------------
# 2) Angle / triad loss
# ----------------------------

def _knn_indices(base: torch.Tensor, k: int) -> torch.Tensor:
    """
    base: (B,N,3)  -> returns neighbor idx (B,N,k), excluding self
    """
    B, N, _ = base.shape
    D = torch.cdist(base, base)  # (B,N,N)
    # exclude self by adding +inf to diagonal
    inf_diag = torch.full((B, N), float('inf'), device=base.device, dtype=base.dtype)
    D = D + torch.diag_embed(inf_diag)
    idx = torch.topk(D, k=k, largest=False, dim=-1).indices  # (B,N,k)
    return idx


def _angle_distribution(x: torch.Tensor,
                        idx_knn: torch.Tensor,
                        mode: str = "hist",
                        bins: int = 72,
                        sigma: Optional[float] = None,
                        return_cos: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute per-batch angle distributions using neighbor triplets centered at each point.
    x: (B,N,3)
    idx_knn: (B,N,k) indices of neighbors per center (fixed graph)
    mode: "hist" or "sorted"
    bins: for histogram if mode="hist"
    sigma: Gaussian bandwidth for hist over angles (in radians)
    return_cos: if True, returns cos(theta) values (for debugging)

    Returns:
      - If mode="hist": (B,bins), bin_centers (B,bins)
      - If mode="sorted": (B,M) sorted angle (or cos) values, None
    """
    B, N, _ = x.shape
    k = idx_knn.shape[-1]
    device = x.device
    dtype = x.dtype

    # Neighbor vectors u relative to center j
    # centers: (B,N,1,3)
    centers = x.unsqueeze(2)
    # Use advanced indexing to gather neighbors
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    nbrs = x[batch_idx, idx_knn]  # (B,N,k,3)
    u = nbrs - centers  # (B,N,k,3)
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-12)
    u_hat = u / u_norm  # (B,N,k,3)

    # Cosines via Gram matrix over neighbor dimension
    # G[b,n] = u_hat[b,n] @ u_hat[b,n]^T  -> (k,k)
    # Extract upper triangle p<q to avoid duplicates/self
    G = torch.einsum('bnkd,bnmd->bnkm', u_hat, u_hat)  # (B,N,k,k)
    triu_mask = torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), diagonal=1)
    cos_vals = G[:, :, triu_mask].reshape(B, -1)  # (B, N*k*(k-1)/2)

    if mode == "sorted":
        if return_cos:
            # Return sorted cosines
            vals_sorted, _ = torch.sort(cos_vals, dim=-1)
            return vals_sorted, None
        else:
            # angles in [0, pi]
            angles = torch.acos(cos_vals.clamp(-1.0, 1.0))
            angles_sorted, _ = torch.sort(angles, dim=-1)
            return angles_sorted, None
    elif mode == "hist":
        if sigma is None:
            sigma = 0.02 * math.pi  # default smoothness

        if return_cos:
            # Histogram in cos-space [-1,1]
            vmin = torch.full((B,), -1.0, device=device, dtype=dtype)
            vmax = torch.full((B,), 1.0, device=device, dtype=dtype)
            hist, centers = _soft_histogram(cos_vals, bins, vmin, vmax, sigma=sigma, normalize=True)
            return hist, centers
        else:
            # angles in [0, pi]
            angles = torch.acos(cos_vals.clamp(-1.0, 1.0))
            vmin = torch.zeros((B,), device=device, dtype=dtype)
            vmax = torch.full((B,), math.pi, device=device, dtype=dtype)
            hist, centers = _soft_histogram(angles, bins, vmin, vmax, sigma=sigma, normalize=True)
            return hist, centers
    else:
        raise ValueError("mode must be 'sorted' or 'hist'")


def angle_triad_loss(x: torch.Tensor,
                     y: torch.Tensor,
                     k: int = 8,
                     mode: str = "hist",
                     bins: int = 72,
                     sigma: Optional[float] = None,
                     base_graph: str = "x",
                     use_cos: bool = False,
                     reduction: str = "mean") -> torch.Tensor:
    """
    Compare angle (or cosine) distributions from neighbor triplets.

    x, y: (B,N,3)
    k: neighbors per center (typ. 6-12 for 3D local geometry)
    mode: "hist" (smooth hist loss) or "sorted" (sort & L2)
    bins: number of bins if mode="hist"
    sigma: bandwidth for soft histogram (radians if use_cos=False). If None, auto.
    base_graph: 'x' uses GT for the KNN graph (robust), 'y' uses predicted.
    use_cos: if True, work in cos(theta) space in [-1,1]; else angles in [0,pi]
    """
    assert x.shape == y.shape and x.ndim == 3
    base = x if base_graph == "x" else y
    idx = _knn_indices(base, k=k)  # (B,N,k)

    if mode == "sorted":
        a_x, _ = _angle_distribution(x, idx, mode="sorted", return_cos=use_cos)
        a_y, _ = _angle_distribution(y, idx, mode="sorted", return_cos=use_cos)
        loss_vec = (a_x - a_y).pow(2).mean(dim=-1)
    elif mode == "hist":
        h_x, _ = _angle_distribution(x, idx, mode="hist", bins=bins, sigma=sigma, return_cos=use_cos)
        h_y, _ = _angle_distribution(y, idx, mode="hist", bins=bins, sigma=sigma, return_cos=use_cos)
        loss_vec = (h_x - h_y).pow(2).mean(dim=-1)
    else:
        raise ValueError("mode must be 'sorted' or 'hist'")

    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec


# ----------------------------
# 3) Pair Distribution Function (RDF / g(r))
# ----------------------------

def pair_distribution_function(x: torch.Tensor,
                               bins: int = 64,
                               r_max: Optional[torch.Tensor] = None,
                               sigma: Optional[torch.Tensor] = None,
                               volume: Optional[torch.Tensor] = None,
                               normalize_mode: str = "pdf") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a smooth radial distribution function per batch.

    x: (B,N,3)
    bins: number of radial bins
    r_max: scalar tensor or (B,) tensor of max radius for histogram. If None -> per-batch max pair distance.
    sigma: Gaussian bandwidth for radial soft histogram (absolute units). If None -> 2% of r_max.
    volume: optional scalar or (B,) for the occupied volume. If provided, we compute a density-normalized g(r).
    normalize_mode:
      - "pdf": histogram normalized to integrate to 1 (probability density over r)
      - "gr":  if volume provided: return g(r) ≈ counts / (rho * N * 4π r^2 Δr)
               if volume None: returns shape-corrected proxy dividing by r^2 then L1-normalized.

    Returns:
      g: (B, bins)  (PDF or g(r) depending on normalize_mode)
      r_centers: (B, bins)
    """
    assert x.ndim == 3
    B, N, _ = x.shape
    device = x.device
    dtype = x.dtype

    D = _pairwise_dists(x)  # (B,N,N)
    # Extract i<j, then mirror counts by factor 2 to treat i!=j pairs
    r = _gather_upper_tri(D)  # (B, P), P=N*(N-1)/2

    # r_max per batch
    if r_max is None:
        r_max = r.max(dim=-1).values.detach()  # (B,)
    elif not isinstance(r_max, torch.Tensor):
        r_max = torch.tensor([r_max]*B, device=device, dtype=dtype)

    # Add a small margin
    rmax_pad = r_max + 1e-6

    # sigma per batch
    if sigma is None:
        sigma = 0.02 * r_max.clamp_min(1e-6)
    elif not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor([sigma]*B, device=device, dtype=dtype)

    # Smooth histogram in [0, rmax]
    vmin = torch.zeros((B,), device=device, dtype=dtype)
    h, rc = _soft_histogram(r, bins, vmin, rmax_pad, sigma, normalize=False)  # (B,bins)

    dr = (rmax_pad - vmin) / (bins - 1 + 1e-12)  # (B,)
    dr = dr.clamp_min(1e-12)

    if normalize_mode == "pdf":
        # Normalize to integrate to 1 over r
        h_pdf = h / (h.sum(dim=-1, keepdim=True) + 1e-12)
        return h_pdf, rc

    # g(r) normalization
    # counts per bin -> divide by shell area 4π r^2 and density
    r_centers = rc  # (B,bins)
    shell_area = 4.0 * math.pi * (r_centers**2)  # (B,bins)
    counts_per_length = h / dr.unsqueeze(-1)     # approximate density along r

    if volume is not None:
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor([volume]*B, device=device, dtype=dtype)
        rho = (N / volume).unsqueeze(-1)  # (B,1)
        # each unordered pair counted once; for i!=j counts, factor 2 cancels in density-normalization
        g = counts_per_length / (rho * N * shell_area + 1e-12)  # (B,bins)
        return g, r_centers
    else:
        # shape-corrected proxy: divide by r^2 then L1-normalize for comparability
        g_proxy = counts_per_length / (shell_area + 1e-12)
        g_proxy = g_proxy / (g_proxy.sum(dim=-1, keepdim=True) + 1e-12)
        return g_proxy, r_centers


def rdf_loss(x: torch.Tensor,
             y: torch.Tensor,
             bins: int = 64,
             r_max: Optional[float] = None,
             sigma: Optional[float] = None,
             volume: Optional[float] = None,
             normalize_mode: str = "pdf",
             reduction: str = "mean") -> torch.Tensor:
    """
    Compare RDF/PDF between x and y via L2.
    """
    gx, rcx = pair_distribution_function(
        x, bins=bins, r_max=r_max, sigma=sigma, volume=volume, normalize_mode=normalize_mode
    )
    gy, _ = pair_distribution_function(
        y, bins=bins, r_max=r_max, sigma=sigma, volume=volume, normalize_mode=normalize_mode
    )
    loss_vec = (gx - gy).pow(2).mean(dim=-1)
    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec


# ----------------------------
# Example usage in a training step
# ----------------------------
if __name__ == "__main__":
    B, N = 4, 80
    x = torch.randn(B, N, 3, device='cpu')  # ground truth
    y = x + 0.05 * torch.randn_like(x)      # prediction (noisy)

    # Pairwise distance loss
    l_pair = pairwise_distance_loss(x, y, mode="hist", hist_bins=64)

    # Angle/triad loss (k=8 neighbors, histogram over angles)
    l_angle = angle_triad_loss(x, y, k=8, mode="hist", bins=72, base_graph="x")

    # RDF loss (PDF-style)
    l_rdf = rdf_loss(x, y, bins=64, normalize_mode="pdf")

    total = l_pair + l_angle + l_rdf
    print(l_pair.item(), l_angle.item(), l_rdf.item(), total.item())
