"""Trainable MACE export plus a proper spatial early-warning likelihood."""
import torch
from torch import nn
from torch.nn import functional as F
from src.research.supervised_onset.model import CapacityEncoder
from src.research.spatial_distance.model import parameters, log_likelihood, cdf, capped_mean, capped_median


class DistanceEncoder(nn.Module):
    def __init__(self,encoder_config):
        super().__init__()
        self.encoder=CapacityEncoder(**encoder_config)
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))

    def forward(self,graph,return_embedding=False):
        z=self.encoder(graph)
        # Tensor-core encoder/head, but probability parameters and tail
        # likelihoods retain float32 precision under autocast.
        raw=self.head(z).float()[:,None]
        parts=parameters(raw,raw[...,0]*0)
        return (parts,z.float()) if return_embedding else parts


def loss_terms(parts,distance,config):
    """Fixed positive horizon weights preserve a proper probability objective.

    No label-dependent weighting, AP/ranking term, or alarm reward is used.
    Stable log CDF/survival avoid clipping saturated probabilities in training.
    """
    log_weight,zero,mu,sigma=parts
    radii=distance.new_tensor(config['early_radii'])
    z=(radii.log()-mu[...,None])/sigma[...,None]
    log_cdf=torch.logaddexp(F.logsigmoid(zero)[...,None],
        F.logsigmoid(-zero)[...,None]+torch.special.log_ndtr(z))
    log_survival=F.logsigmoid(-zero)[...,None]+torch.special.log_ndtr(-z)
    log_cdf=torch.logsumexp(log_weight[...,None]+log_cdf,1)
    log_survival=torch.logsumexp(log_weight[...,None]+log_survival,1)
    bce=torch.where(distance[:,None]<=radii,-log_cdf,-log_survival)
    early=(bce*distance.new_tensor(config['early_weights'])).sum(-1)
    nll=-log_likelihood(parts,distance,config['distance_cap'])
    return nll+config['early_weight']*early,nll,early
