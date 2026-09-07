"""Topology-aware attraction and continuous geometry matching; no cluster labels."""
import torch
from torch.nn import functional as F


def weighted_distance(a,b,weights):
    return ((a-b).square()*weights).sum(-1)/weights.sum()


def attraction(a,b,material,weights,medians,minimum):
    scales=a.new_tensor(medians)[material]
    # At the training median distance the attraction is one half.
    return torch.exp(-0.6931471805599453*weighted_distance(a,b,weights)/scales).clamp_min(minimum).detach()


def density_matched_pairs(density,material,tolerance):
    left=[];right=[]
    for i in range(3):
        ids=torch.nonzero(material==i,as_tuple=True)[0]
        ids=ids[density[ids].argsort()];other=ids.roll(1)
        keep=(density[ids]-density[other]).abs()/(.5*(density[ids]+density[other]))<=tolerance
        left.append(ids[keep]);right.append(other[keep])
    return torch.cat(left),torch.cat(right)


def topology_terms(z,target,material,density,settings):
    weights=target.new_tensor(settings['reliability_weights'])
    spatial=attraction(target[:,0],target[:,1],material,weights,settings['spatial_distance_medians'],settings['minimum_attraction'])
    temporal=attraction(target[:,0],target[:,2],material,weights,settings['temporal_distance_medians'],settings['minimum_attraction'])
    left,right=density_matched_pairs(density,material,settings['density_relative_tolerance'])
    if len(left)==0:raise ValueError('No same-material density-matched pairs for topology-distance training')
    desired=weighted_distance(target[left,0],target[right,0],weights).detach()
    actual=(z[left,0]-z[right,0]).square().mean(-1)
    distance_loss=F.smooth_l1_loss(actual,desired,beta=settings['distance_huber_beta'])
    return spatial,temporal,distance_loss,len(left),weights


def neighborhood_density(x):
    # Repository producer sorts the center and nearest atoms by distance.
    radius=x[:,0,12].norm(dim=-1)
    return 12/(4*torch.pi/3*radius.pow(3))
