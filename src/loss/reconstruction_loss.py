import sys,os
sys.path.append(os.getcwd())
import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
from src.utils.logging_config import setup_logging
logger = setup_logging()



def chamfer_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss between predictions and targets.

    Args:
        pred (Tensor): Predicted point cloud.
        target (Tensor): Ground truth point cloud.
        **kwargs: Additional keyword arguments (currently ignored).

    Returns:
        Tensor: Chamfer distance loss.
    """
    loss, _ = chamfer_distance(pred, target)
    return loss, []


def chamfer_regularized_encoder_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss with feature transform regularization for the encoder.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        trans_feat_list (list or tuple): Contains at least the encoder's feature transform as the first element.
        feature_transform_loss_scale (float): Scale factor for the feature transform regularization loss.

    Returns:
        Tensor: Total loss.
    """
    loss, _ = chamfer_distance(pred, target)
    encoder_trans = kwargs['trans_feat_list'][0]
    reg_loss = feature_transform_regularizer(encoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * reg_loss
    return total_loss, [reg_loss]


def chamfer_regularized_encoder_decoder_loss(pred, target, **kwargs ):
    """
    Computes the Chamfer distance loss with feature transform regularization for both encoder and decoder.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        trans_feat_list (list or tuple): Contains [encoder_feature_transform, decoder_feature_transform].
        feature_transform_loss_scale (float): Scale factor for the feature transform regularization loss.

    Returns:
        Tensor: Total loss.
    """
    if len(kwargs['trans_feat_list']) < 2:
        raise ValueError("trans_feat_list must contain both encoder and decoder transformation features.")

    loss, _ = chamfer_distance(pred, target)
    encoder_trans, decoder_trans = kwargs['trans_feat_list']
    encoder_reg_loss = feature_transform_regularizer(encoder_trans)
    decoder_reg_loss = feature_transform_regularizer(decoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * (encoder_reg_loss + decoder_reg_loss)
    return total_loss, [encoder_reg_loss, decoder_reg_loss]


def chamfer_wasserstein_loss(pred, target, **kwargs):
    loss, _ = chamfer_distance(pred, target)
    rdf1, r_mid1 = calculate_rdf_central(pred, density=kwargs['density'], dr=kwargs['dr'], num_points=kwargs['num_points'])
    rdf2, r_mid2 = calculate_rdf_central(target, density=kwargs['density'], dr=kwargs['dr'], num_points=kwargs['num_points'])
    rec_loss = wasserstein_distance_loss(rdf1, rdf2)
    feature_transform_loss = feature_transform_regularizer(kwargs['trans_feat_list'][0])    
    total_loss = loss + rec_loss * kwargs['reconstruction_loss_scale'] + feature_transform_loss * kwargs['feature_transform_loss_scale']
    return total_loss, [rec_loss, feature_transform_loss]



def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def calculate_rdf_pairvise(point_cloud, sphere_radius, dr, num_points=None, drop_first_n_bins=2, density=None):
    """
    Calculates the Radial Distribution Function (RDF) for batched points within a sphere using vectorized operations.
    
    If a precomputed density is provided (from config where sphere_radius and num_points are constants),
    the function will use it rather than recalculating the density.

    Args:
        point_cloud (torch.Tensor): Point cloud of shape (B, 3, N) or (B, N, 3), 
                                    where B is batch size and N is number of points.
        sphere_radius (float): Radius of the sphere containing the points.
        dr (float): Bin width for the RDF histogram.
        num_points (int, optional): Number of points to expect in the sphere.
                                    If None, it is inferred from the point cloud.
        drop_first_n_bins (int): Number of initial bins to drop.
        density (float, optional): Precomputed density (num_points / volume). If provided,
                                   density calculation is skipped.

    Returns:
        torch.Tensor: RDF values for each radial bin, shape (B, num_bins_after_drop).
        torch.Tensor: Radial distances corresponding to the RDF bins after dropping.
    """
    # Ensure point cloud is in (B, N, 3) format.
    if point_cloud.shape[1] == 3:
        point_cloud = point_cloud.transpose(1, 2)

    B, N = point_cloud.shape[0], point_cloud.shape[1]
    if num_points is None:
        num_points = N

    # Compute density only if not provided.
    if density is None:
        volume = (4/3) * torch.pi * (sphere_radius**3)
        density = num_points / volume

    # Create radial bins and compute r_mid.
    r_bins = torch.arange(0, sphere_radius + dr, dr, device=point_cloud.device)
    r_mid = (r_bins[:-1] + r_bins[1:]) / 2
    num_bins = len(r_bins) - 1

    # Compute all pairwise distances in a batched manner.
    distances = torch.cdist(point_cloud, point_cloud)

    # Get the upper triangular indices for all batches to avoid double counting.
    triu_idx = torch.triu_indices(N, N, offset=1)
    distances_upper = distances[:, triu_idx[0], triu_idx[1]]

    # Bucketize to assign each distance to a bin.
    bin_indices = torch.bucketize(distances_upper, r_bins, right=True) - 1
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # Create one-hot encoding of bin indices and sum along the M dimension per batch.
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins).to(torch.float)
    hist = one_hot.sum(dim=1)
    
    # Calculate ideal counts per shell for normalization.
    shell_volumes = 4 * torch.pi * (r_mid ** 2) * dr
    ideal_counts = shell_volumes * density * num_points

    # Normalize histogram counts to get rdf per batch.
    rdf = hist / (ideal_counts.unsqueeze(0) + 1e-10)
    rdf[rdf != rdf] = 0.0  # Replace any NaNs with 0

    # Drop the first few bins if requested.
    rdf = rdf[:, drop_first_n_bins:]
    r_mid = r_mid[drop_first_n_bins:]

    return rdf, r_mid


def calculate_rdf_central(point_cloud, sphere_radius, dr, drop_first_n_bins=1, density=None):
    """
    Calculates the Radial Distribution Function (RDF) for each point cloud in a batch, using the point 
    closest to the sample center as the reference.

    The function works as follows:
      1. Computes the centroid for each point cloud sample.
      2. Finds the point closest to this centroid (the “central point”).
      3. Computes Euclidean distances from this central point to all points in the sample.
      4. Bins these distances into radial bins defined by sphere_radius and dr.
      5. Normalizes the histogram counts by the ideal count expected for each shell.
      6. Drops the first few bins (typically dropping the zero distance) as requested.

    Args:
        point_cloud (torch.Tensor): Batched point clouds of shape (B, N, 3), where B is the batch size.
        sphere_radius (float): Maximum radius within which to calculate the RDF.
        dr (float): Bin width for the RDF histogram.
        drop_first_n_bins (int): Number of initial bins to drop (default is 1 to exclude the zero-distance bin).
        density (float, optional): Precomputed density (points per unit volume). If not provided, it is computed 
                                   as (N / volume) where volume = (4/3)*pi*(sphere_radius)^3.
      
    Returns:
        rdf (torch.Tensor): Radial Distribution Function values for each sample, shape (B, num_bins_after_drop).
        r_mid (torch.Tensor): Midpoints of the radial bins after dropping the initial bins, shape (num_bins_after_drop,).
    """
    B, N, _ = point_cloud.shape
    
    # Compute the centroid for each sample.
    centroid = point_cloud.mean(dim=1)  # Shape: (B, 3)
    
    # For each sample, find the point closest to the centroid.
    dists_to_centroid = torch.norm(point_cloud - centroid.unsqueeze(1), dim=2)  # Shape: (B, N)
    central_idx = torch.argmin(dists_to_centroid, dim=1)  # Shape: (B,)
    batch_indices = torch.arange(B, device=point_cloud.device)
    central_points = point_cloud[batch_indices, central_idx, :]  # Shape: (B, 3)
    
    # Compute distances from the central point to all points within the same sample.
    distances_from_center = torch.norm(point_cloud - central_points.unsqueeze(1), dim=2)  # Shape: (B, N)
    
    # Define radial bins from 0 up to sphere_radius in steps of dr.
    device = point_cloud.device
    bins = torch.arange(0, sphere_radius + dr, dr, device=device)  # (num_bins + 1,) edges
    num_bins = len(bins) - 1
    r_mid = (bins[:-1] + bins[1:]) / 2.0  # Bin midpoints, shape: (num_bins,)
    
    # Bucketize each distance into its corresponding radial bin.
    # Subtract 1 to convert to 0-based indexing.
    bin_indices = torch.bucketize(distances_from_center, bins, right=True) - 1   # Shape: (B, N)
    bin_indices = torch.clamp(bin_indices, min=0, max=num_bins - 1)
    
    # Create one-hot representation and sum over the points dimension to obtain histogram counts per bin.
    one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins).float()  # Shape: (B, N, num_bins)
    hist = one_hot.sum(dim=1)  # Shape: (B, num_bins)
    
    # Compute density if not provided.
    if density is None:
        volume = (4.0 / 3.0) * torch.pi * (sphere_radius ** 3)
        density = N / volume
    
    # Calculate the ideal (expected) counts per radial shell.
    shell_volumes = 4 * torch.pi * (r_mid ** 2) * dr  # Volume of spherical shells, shape: (num_bins,)
    ideal_counts = shell_volumes * density  # Expected count for a uniform distribution, shape: (num_bins,)
    
    # Normalize the histogram counts by the ideal counts to obtain the RDF.
    rdf = hist / (ideal_counts.unsqueeze(0) + 1e-10)
    
    # Optionally drop the first few bins (e.g., the bin centered at zero distance).
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
    rdf1, r_mid1 = calculate_rdf_central(pred, sphere_radius=5, dr=0.05, r_max = 5)
    rdf2, r_mid2 = calculate_rdf_central(target, sphere_radius=5, dr=0.05, r_max = 5)
    loss = wasserstein_distance_loss(rdf1, rdf2)
    return loss

