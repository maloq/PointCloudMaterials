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


def chamfer_distance(pred, target):
    loss, _ = chamfer_distance(pred, target) 
    return loss





def calculate_rdf(point_cloud, r_max, dr, density=None):
    """
    Calculates the Radial Distribution Function (RDF) for a point cloud.

    Args:
        point_cloud (torch.Tensor): Point cloud of shape (N, 3), where N is the number of points.
        r_max (float): Maximum radius to consider for RDF calculation.
        dr (float): Bin width for the RDF histogram.
        density (float, optional): Density of the point cloud. If None, it's estimated from the point cloud volume.

    Returns:
        torch.Tensor: RDF values for each radial bin.
        torch.Tensor: Radial distances corresponding to the RDF bins.
    """
    num_points = point_cloud.shape[0]

    # Calculate pairwise distances (efficiently using torch.cdist)
    distances = torch.cdist(point_cloud, point_cloud)

    # Create radial bins
    r_bins = torch.arange(0, r_max + dr, dr)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2  # Midpoint of each bin

    # Histogram distances to get pair counts in each bin
    hist = torch.histc(distances.flatten(), bins=len(r_bins) - 1, min=0, max=r_max)

    # Normalize to get RDF
    if density is None:
        # Estimate density (assuming points are in a sphere of radius r_max - this is a simplification)
        volume = (4/3) * torch.pi * (r_max**3)  # Volume of a sphere with radius r_max
        density = num_points / volume

    ideal_counts = (4 * torch.pi * r_mid**2 * dr * density * num_points)  # Expected counts in each shell for uniform distribution
    rdf = hist / ideal_counts

    # Handle potential division by zero (if ideal_counts is zero for some bins)
    rdf[torch.isnan(rdf)] = 0.0

    return rdf, r_mid


class RDFLoss(nn.Module):
    def __init__(self, target_rdf, r_mid):
        super(RDFLoss, self).__init__()
        self.target_rdf = target_rdf.detach() # Detach target_rdf to prevent gradients from flowing back
        self.r_mid = r_mid.detach() # Detach r_mid as well

    def forward(self, reconstructed_pc, r_max, dr, density=None):
        """
        Calculates the RDF loss between the RDF of the reconstructed point cloud
        and the target RDF.

        Args:
            reconstructed_pc (torch.Tensor): Reconstructed point cloud.
            r_max (float): Maximum radius for RDF calculation.
            dr (float): Bin width for RDF calculation.
            density (float, optional): Density for RDF calculation.

        Returns:
            torch.Tensor: RDF loss value.
        """
        reconstructed_rdf, _ = calculate_rdf(reconstructed_pc, r_max, dr, density=density)

        # Interpolate target_rdf to match the bins of reconstructed_rdf if needed
        # (if r_mid from target and reconstructed RDF calculations are slightly different)
        # In this simple example, we assume they are calculated with the same bins.
        # If bins are different, use interpolation like torch.nn.functional.interpolate

        # Ensure both RDFs have the same length (handle cases where r_max or dr might slightly differ)
        min_len = min(len(self.target_rdf), len(reconstructed_rdf))
        loss = nn.MSELoss()(reconstructed_rdf[:min_len], self.target_rdf[:min_len])
        return loss