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


def chamfer_distance(pred, target, trans_feat_list):
    loss, _ = chamfer_distance(pred, target) 
    return loss



def calculate_rdf(point_cloud, sphere_radius, dr, num_points=None, drop_first_n_bins=2, ):
    """
    Calculates the Radial Distribution Function (RDF) for points within a sphere.

    Args:
        point_cloud (torch.Tensor): Point cloud of shape (N, 3), where N is the number of points.
        sphere_radius (float): Radius of the sphere containing the points.
        dr (float): Bin width for the RDF histogram.
        num_points (int, optional): Number of points to expect in the sphere. 
                                  If None, uses all points in point_cloud.

    Returns:
        torch.Tensor: RDF values for each radial bin.
        torch.Tensor: Radial distances corresponding to the RDF bins.
    """
    if num_points is None:
        num_points = point_cloud.shape[0]

    # Calculate pairwise distances
    distances = torch.cdist(point_cloud, point_cloud)
    
    # Create radial bins
    r_bins = torch.arange(0, sphere_radius + dr, dr)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2

    # Get upper triangle of distance matrix to avoid double counting
    # and remove self-interactions (diagonal elements)
    distances_triu = distances[torch.triu_indices(num_points, num_points, offset=1)]

    # Histogram distances
    hist = torch.histc(distances_triu, bins=len(r_bins) - 1, min=0, max=sphere_radius)

    # Calculate density for sphere
    volume = (4/3) * torch.pi * (sphere_radius**3)
    density = num_points / volume

    # Calculate ideal counts for normalization
    shell_volumes = 4 * torch.pi * r_mid**2 * dr
    ideal_counts = shell_volumes * density * num_points

    # Normalize to get RDF
    rdf = hist / ideal_counts

    # Handle potential division by zero
    rdf[torch.isnan(rdf)] = 0.0
    rdf = rdf[drop_first_n_bins:]
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
    Calculate approximate 1D Wasserstein distance between two RDFs using cumulative distributions
    
    Args:
        rdf1, rdf2: torch tensors of RDF values
    Returns:
        torch.Tensor: Wasserstein distance loss
    """
    epsilon = 1e-10
    rdf1_norm = rdf1 / (torch.sum(rdf1) + epsilon)
    rdf2_norm = rdf2 / (torch.sum(rdf2) + epsilon)
    
    cdf1 = torch.cumsum(rdf1_norm, dim=0)
    cdf2 = torch.cumsum(rdf2_norm, dim=0)
    loss = torch.sum(torch.abs(cdf1 - cdf2))
    return loss