import sys,os
sys.path.append(os.getcwd())
import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



def chamfer_regularized_encoder_decoder(pred, target, trans_feat_list, feature_transform_loss_scale=0.001):
    loss, _ = chamfer_distance(pred, target) 
    trans_feat_encoder = trans_feat_list[0]
    trans_feat_decoder = trans_feat_list[1]
    feature_transform_loss = feature_transform_reguliarzer(trans_feat_encoder) + feature_transform_reguliarzer(trans_feat_decoder)
    total_loss = loss + feature_transform_loss * feature_transform_loss_scale
    return total_loss



def chamfer_regularized_encoder(pred, target, trans_feat_list, feature_transform_loss_scale=0.001):
    loss, _ = chamfer_distance(pred, target) 
    trans_feat_encoder = trans_feat_list[0]
    feature_transform_loss = feature_transform_reguliarzer(trans_feat_encoder)
    total_loss = loss + feature_transform_loss * feature_transform_loss_scale
    return total_loss



def chamfer(pred, target, **kwargs):
    loss, _ = chamfer_distance(pred, target) 
    return loss






def calculate_rdf_old(point_cloud, sphere_radius, dr, num_points=None, drop_first_n_bins=2):
    """
    Calculates the Radial Distribution Function (RDF) for batched points within a sphere.

    Args:
        point_cloud (torch.Tensor): Point cloud of shape (B, 3, N) or (B, N, 3), 
                                  where B is batch size and N is number of points.
        sphere_radius (float): Radius of the sphere containing the points.
        dr (float): Bin width for the RDF histogram.
        num_points (int, optional): Number of points to expect in the sphere.

    Returns:
        torch.Tensor: RDF values for each radial bin, shape (B, num_bins).
        torch.Tensor: Radial distances corresponding to the RDF bins.
    """
    # Ensure point cloud is in (B, N, 3) format
    if point_cloud.shape[1] == 3:
        point_cloud = point_cloud.transpose(1, 2)
    
    batch_size = point_cloud.shape[0]
    if num_points is None:
        num_points = point_cloud.shape[1]

    # Create radial bins
    r_bins = torch.arange(0, sphere_radius + dr, dr, device=point_cloud.device)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2
    num_bins = len(r_bins) - 1

    # Initialize batch RDF tensor
    batch_rdf = torch.zeros((batch_size, num_bins), device=point_cloud.device)

    for b in range(batch_size):
        # Calculate pairwise distances for each batch
        distances = torch.cdist(point_cloud[b], point_cloud[b])
        
        # Get upper triangle and remove self-interactions
        distances_triu = distances[torch.triu_indices(num_points, num_points, offset=1)]

        # Histogram distances
        hist = torch.histc(distances_triu, bins=num_bins, min=0, max=sphere_radius)

        # Calculate density for sphere
        volume = (4/3) * torch.pi * (sphere_radius**3)
        density = num_points / volume

        # Calculate ideal counts for normalization
        shell_volumes = 4 * torch.pi * r_mid**2 * dr
        ideal_counts = shell_volumes * density * num_points

        # Normalize to get RDF
        rdf = hist / ideal_counts
        rdf[torch.isnan(rdf)] = 0.0
        batch_rdf[b] = rdf

    batch_rdf = batch_rdf[:, drop_first_n_bins:]
    r_mid = r_mid[drop_first_n_bins:]

    return batch_rdf, r_mid



def calculate_rdf(point_cloud, sphere_radius, dr, num_points=None, drop_first_n_bins=2):
    """
    Calculates the Radial Distribution Function (RDF) for batched points within a sphere faster using vectorized operations.

    Args:
        point_cloud (torch.Tensor): Point cloud of shape (B, 3, N) or (B, N, 3), 
                                    where B is batch size and N is number of points.
        sphere_radius (float): Radius of the sphere containing the points.
        dr (float): Bin width for the RDF histogram.
        num_points (int, optional): Number of points to expect in the sphere.
        drop_first_n_bins (int): Number of initial bins to drop.

    Returns:
        torch.Tensor: RDF values for each radial bin, shape (B, num_bins_after_drop).
        torch.Tensor: Radial distances corresponding to the RDF bins after dropping.
    """
    # Ensure point cloud is in (B, N, 3) format
    if point_cloud.shape[1] == 3:
        point_cloud = point_cloud.transpose(1, 2)
    
    B, N = point_cloud.shape[0], point_cloud.shape[1]
    if num_points is None:
        num_points = N


    # Create radial bins and compute r_mid
    r_bins = torch.arange(0, sphere_radius + dr, dr, device=point_cloud.device)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2
    num_bins = len(r_bins) - 1

    # Compute all pairwise distances in a batched manner.
    # distances: shape (B, N, N)
    distances = torch.cdist(point_cloud, point_cloud)

    # Get the same upper triangular indices for all batches to avoid double counting
    triu_idx = torch.triu_indices(N, N, offset=1)
    # distances_upper: shape (B, M) where M = number of upper triangular elements per batch
    distances_upper = distances[:, triu_idx[0], triu_idx[1]]

    # Use bucketize to assign each distance to a bin
    # Use right=True so that values exactly equal to a bin edge are handled like torch.histc.
    bin_indices = torch.bucketize(distances_upper, r_bins, right=True) - 1
    # Clamp bin indices to ensure they fall into the valid range [0, num_bins-1]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # Create one-hot encoding of bin indices and sum along the M dimension per batch.
    # shape: (B, M, num_bins)
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins).to(torch.float)
    # Compute histograms for each batch; shape: (B, num_bins)
    hist = one_hot.sum(dim=1)
    
    # Calculate density for sphere: assume each point counts in the density estimation
    volume = (4/3) * torch.pi * (sphere_radius**3)
    density = num_points / volume
    # Calculate ideal counts per shell for normalization
    shell_volumes = 4 * torch.pi * (r_mid ** 2) * dr
    ideal_counts = shell_volumes * density * num_points

    # Normalize histogram counts to get rdf per batch (broadcast division over bins)
    rdf = hist / (ideal_counts.unsqueeze(0) + 1e-10)
    rdf[rdf != rdf] = 0.0  # Replace any NaNs with 0

    # Drop the first few bins if requested
    rdf = rdf[:, drop_first_n_bins:]
    r_mid = r_mid[drop_first_n_bins:]

    return rdf, r_mid



def kl_divergence_loss(rdf1, rdf2):
    """
    Calculate symmetric KL divergence (Jensen-Shannon without 0.5 factor) between two RDFs
    
    Args:
        rdf1, rdf2: torch tensors of RDF values
    Returns:
        torch.Tensor: KL divergence loss
    """
    epsilon = 1e-10
    rdf1_norm = rdf1 / (torch.sum(rdf1) + epsilon)
    rdf2_norm = rdf2 / (torch.sum(rdf2) + epsilon)
    
    m = 0.5 * (rdf1_norm + rdf2_norm)
    loss = torch.sum(rdf1_norm * torch.log(rdf1_norm / (m + epsilon) + epsilon)) + \
           torch.sum(rdf2_norm * torch.log(rdf2_norm / (m + epsilon) + epsilon))
    return loss



def wasserstein_distance_loss(rdf1, rdf2):
    """
    Calculate batched approximate 1D Wasserstein distance between two RDFs
    
    Args:
        rdf1, rdf2: torch tensors of RDF values, shape (B, num_bins)
    Returns:
        torch.Tensor: Wasserstein distance loss per batch
    """
    epsilon = 1e-10
    rdf1_norm = rdf1 / (torch.sum(rdf1, dim=1, keepdim=True) + epsilon)
    rdf2_norm = rdf2 / (torch.sum(rdf2, dim=1, keepdim=True) + epsilon)
    
    cdf1 = torch.cumsum(rdf1_norm, dim=1)
    cdf2 = torch.cumsum(rdf2_norm, dim=1)
    loss = torch.sum(torch.abs(cdf1 - cdf2), dim=1)
    return loss.mean()


def rdf_wasserstein_loss(pred, target):
    rdf1, r_mid1 = calculate_rdf(pred, sphere_radius=5, dr=0.05, r_max = 5)
    rdf2, r_mid2 = calculate_rdf(target, sphere_radius=5, dr=0.05, r_max = 5)
    loss = wasserstein_distance_loss(rdf1, rdf2)
    return loss


def chamfer_wasserstein_loss(pred, target, reconstruction_loss_scale=0.005, feature_transform_loss_scale=0.0001, trans_feat_list=None):
    loss, _ = chamfer_distance(pred, target)
    rdf1, r_mid1 = calculate_rdf(pred, sphere_radius=5, dr=0.05)
    rdf2, r_mid2 = calculate_rdf(target, sphere_radius=5, dr=0.05)
    rec_loss = wasserstein_distance_loss(rdf1, rdf2)

    feature_transform_loss = feature_transform_reguliarzer(trans_feat_list[0])
    total_loss = loss + rec_loss * reconstruction_loss_scale + feature_transform_loss * feature_transform_loss_scale

    return total_loss, rec_loss, feature_transform_loss