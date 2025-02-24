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
    return loss, [0]



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
      1. Extracts the radial distances (r) from the input.
      2. Bins these radial distances into spherical shells defined by the sphere_radius and dr.
      3. Computes the histogram of points per radial shell.
      4. Calculates the ideal (expected) counts for a uniform distribution by computing 
         the exact volume of each spherical shell:
            shell volume = (4/3)*π*((r_out)^3 - (r_in)^3)
         and using the density which is given by:
            density = (number of points) / ((4/3)*π*(sphere_radius)^3).
      5. Normalizes histogram counts by the expected counts and optionally drops the first few bins.
    
    Args:
        spherical_points (torch.Tensor): Batched point clouds in spherical coordinates of shape (B, N, 3),
                                         where each point is represented as (r, theta, phi) 
                                         and B is the batch size.
        sphere_radius (float): Maximum radius within which to calculate the RDF.
        dr (float): Bin width for constructing the RDF histogram.
        drop_first_n_bins (int): Number of initial bins to drop (default is 1) to remove the
                                 near-zero-distance bin that may yield numerical artifacts.
        density (float, optional): Precomputed density (points per unit volume). If not provided,
                                   it is computed as N / ((4/3)*π*(sphere_radius)^3).

    Returns:
        rdf (torch.Tensor): RDF values for each sample, of shape (B, num_bins_after_drop).
        r_mid (torch.Tensor): Midpoints of the radial bins after dropping the initial bins, of shape (num_bins_after_drop,).
    """
    B, N, _ = spherical_points.shape

    # Extract the radial values. Assumes the spherical ordering: (r, theta, phi)
    r_values = spherical_points[..., 0]  # Shape: (B, N)
    
    # Define radial bins (edges) from 0 up to sphere_radius in steps of dr.
    device = spherical_points.device
    bins = torch.arange(0, sphere_radius + dr, dr, device=device)  # edges: (num_bins+1,)
    num_bins = len(bins) - 1

    # Compute the midpoints (for plotting or reference)
    r_mid = (bins[:-1] + bins[1:]) / 2.0  # Shape: (num_bins,)

    # Bucketize each radial value into its corresponding bin.
    # torch.bucketize returns indices in [1, num_bins+1], so we subtract one for 0-based indices.
    bin_indices = torch.bucketize(r_values, bins, right=True) - 1   # Shape: (B, N)
    bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)

    # Create one-hot representation of bins and sum over points dimension to obtain per-bin counts.
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins).float()  # Shape: (B, N, num_bins)
    hist = one_hot.sum(dim=1)  # Shape: (B, num_bins)

    # Compute the density if not provided.
    if density is None:
        volume = (4.0 / 3.0) * torch.pi * (sphere_radius ** 3)
        density = N / volume

    # Compute exact shell volumes for each bin:
    # shell volume = (4/3) * π * ((bins[i+1])^3 - (bins[i])^3)
    shell_volumes = (4.0 / 3.0) * torch.pi * (bins[1:]**3 - bins[:-1]**3)  # Shape: (num_bins,)
    ideal_counts = shell_volumes * density

    # Normalize histogram counts by the ideal counts to obtain the RDF.
    rdf = hist / (ideal_counts.unsqueeze(0) + 1e-10)

    # Optionally drop the first few bins (e.g., the one near zero).
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

def sample_uniform_spherical_points(B, N, sphere_radius, device=None, dtype=torch.float):
    """
    Samples points uniformly within a sphere of given radius in spherical coordinates (r, theta, phi).
    
    Args:
        B (int): Batch size.
        N (int): Number of points per batch.
        sphere_radius (float): The radius of the sphere.
        device (torch.device, optional): The device on which to create the tensor. Defaults to CPU.
        dtype (torch.dtype, optional): Data type of the output tensor.
        
    Returns:
        torch.Tensor: A tensor of shape (B, N, 3) where each point is in spherical coordinates (r, theta, phi).
                      'r' is sampled as r = sphere_radius * u^(1/3) with u ~ U(0,1), 
                      'theta' is in [0, π] with proper weighting, and 
                      'phi' is in [0, 2π).
    """
    if device is None:
        device = torch.device('cpu')
    
    # Sample radial component using the cube root transform for uniform volume.
    u = torch.rand(B, N, 1, device=device, dtype=dtype)
    r = sphere_radius * torch.pow(u, 1.0/3.0)
    
    # Sample the polar angle theta: sample cos(theta) uniformly from [-1, 1]
    cos_theta = 2 * torch.rand(B, N, 1, device=device, dtype=dtype) - 1
    theta = torch.acos(cos_theta)  # theta in [0, pi]
    
    # Sample the azimuthal angle phi uniformly from [0, 2π)
    phi = 2 * torch.pi * torch.rand(B, N, 1, device=device, dtype=dtype)
    
    # Concatenate to form (r, theta, phi)
    spherical_points = torch.cat([r, theta, phi], dim=-1)
    return spherical_points