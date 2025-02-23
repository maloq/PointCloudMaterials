import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from src.utils.logging_config import setup_logging
from src.loss.reconstruction_loss import kl_divergence_loss, wasserstein_distance_loss
logger = setup_logging()



def mse_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss between predictions and targets.

    Args:
        pred (Tensor): Predicted point cloud.
        target (Tensor): Ground truth point cloud.
        **kwargs: Additional keyword arguments (currently ignored).

    Returns:
        Tensor: Chamfer distance loss.
    """
    loss = nn.functional.mse_loss(pred, target)
    return loss, []



def chamfer_wasserstein_loss(pred, target, **kwargs):
    loss, _ = mse_loss(pred, target)
    rdf1, r_mid1 = calculate_rdf_spherical(pred, density=kwargs['density'], dr=kwargs['dr'], sphere_radius=kwargs['sphere_radius'])
    rdf2, r_mid2 = calculate_rdf_spherical(target, density=kwargs['density'], dr=kwargs['dr'], sphere_radius=kwargs['sphere_radius'])
    rec_loss = wasserstein_distance_loss(rdf1, rdf2)
    total_loss = loss + rec_loss * kwargs['reconstruction_loss_scale']
    return total_loss, [rec_loss,]


def chamfer_kl_divergence_loss(pred, target, **kwargs):
    loss, _ = mse_loss(pred, target)
    rdf1, r_mid1 = calculate_rdf_spherical(pred, density=kwargs['density'], dr=kwargs['dr'], sphere_radius=kwargs['sphere_radius'])
    rdf2, r_mid2 = calculate_rdf_spherical(target, density=kwargs['density'], dr=kwargs['dr'], sphere_radius=kwargs['sphere_radius'])
    rec_loss = kl_divergence_loss(rdf1, rdf2)
    total_loss = loss + rec_loss * kwargs['reconstruction_loss_scale']
    return total_loss, [rec_loss,]



def calculate_rdf_spherical(spherical_points, sphere_radius, dr, drop_first_n_bins=1, density=None):
    """
    Calculates the Radial Distribution Function (RDF) for each point cloud in a batch, where
    the point clouds are provided in spherical coordinates. It assumes that the input has the form
    (r, theta, phi) and that the points are already sorted by their radial coordinate.

    The function performs the following operations:
      1. Extracts the radial distances (r) from the input, which already represent the distance 
         from the (normalized) center.
      2. Bins these radial distances into shells defined by sphere_radius and dr.
      3. Computes the histogram of points per radial shell.
      4. Normalizes the histogram counts by the ideal (expected) counts for a uniform distribution,
         where the expected count for a shell is given by its volume:
            shell volume = 4 * π * (r_mid)^2 * dr
         and
            density = (number of points) / ((4/3)*π*(sphere_radius)^3).
      5. Drops the first few bins (e.g., the one centered at zero) if requested.

    Args:
        spherical_points (torch.Tensor): Batched point clouds in spherical coordinates of shape (B, N, 3),
                                         where each point is represented as (r, theta, phi) and B is the batch size.
        sphere_radius (float): Maximum radius within which to calculate the RDF.
        dr (float): Bin width for constructing the RDF histogram.
        drop_first_n_bins (int): Number of initial bins to drop (default is 1) to possibly remove the zero-distance bin.
        density (float, optional): Precomputed density (points per unit volume). If not provided,
                                   it is computed as N / ((4/3)*π*(sphere_radius)^3).

    Returns:
        rdf (torch.Tensor): RDF values for each sample, of shape (B, num_bins_after_drop).
        r_mid (torch.Tensor): Midpoints of the radial bins after dropping the initial bins, of shape (num_bins_after_drop,).
    """
    B, N, _ = spherical_points.shape

    # Extract the radial values. Assumes the spherical ordering: (r, theta, phi)
    r_values = spherical_points[..., 0]  # Shape: (B, N)
    
    # Define radial bins from 0 up to sphere_radius in steps of dr.
    device = spherical_points.device
    bins = torch.arange(0, sphere_radius + dr, dr, device=device)  # (num_bins + 1,) edges
    num_bins = len(bins) - 1
    r_mid = (bins[:-1] + bins[1:]) / 2.0  # Bin midpoints, shape: (num_bins,)

    # Bucketize each radial value into its corresponding bin.
    # torch.bucketize returns indices in [1, num_bins+1], so subtract 1 for 0-based indices.
    bin_indices = torch.bucketize(r_values, bins, right=True) - 1   # Shape: (B, N)
    bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)

    # Create a one-hot representation of the bins and sum over the points dimension to obtain per-bin counts.
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins).float()  # Shape: (B, N, num_bins)
    hist = one_hot.sum(dim=1)  # Shape: (B, num_bins)

    # Compute the density if not provided.
    if density is None:
        volume = (4.0 / 3.0) * torch.pi * (sphere_radius ** 3)
        density = N / volume

    # Calculate the ideal (expected) counts per radial shell:
    # shell_volumes = 4 * π * (r_mid)^2 * dr
    shell_volumes = 4 * torch.pi * (r_mid ** 2) * dr  # Shape: (num_bins,)
    ideal_counts = shell_volumes * density

    # Normalize the histogram counts by the expected counts to obtain the RDF.
    rdf = hist / (ideal_counts.unsqueeze(0) + 1e-10)

    # Optionally drop the first few bins (e.g., the one centered at zero).
    rdf = rdf[:, drop_first_n_bins:]
    r_mid = r_mid[drop_first_n_bins:]
    
    return rdf, r_mid



def rdf_wasserstein_loss(pred, target):
    rdf1, r_mid1 = calculate_rdf_spherical(pred, sphere_radius=5, dr=0.05, r_max = 5)
    rdf2, r_mid2 = calculate_rdf_spherical(target, sphere_radius=5, dr=0.05, r_max = 5)
    loss = wasserstein_distance_loss(rdf1, rdf2)
    return loss

def rdf_kl_divergence_loss(pred, target):
    rdf1, r_mid1 = calculate_rdf_spherical(pred, sphere_radius=5, dr=0.05)
    rdf2, r_mid2 = calculate_rdf_spherical(target, sphere_radius=5, dr=0.05)
    loss = kl_divergence_loss(rdf1, rdf2)
    return loss