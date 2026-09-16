"""Finite-time motion and subspace operations for native encoder training."""
import torch


def time_differences(z, times):
    """Native increments, physical velocities, acceleration and finite-lag bend."""
    dt = times[:,1:]-times[:,:-1]
    if torch.any(dt <= 0): raise ValueError('Physical sequence times must increase')
    dt = dt.to(z.dtype)
    increment = z[:,1:]-z[:,:-1]
    velocity = increment/dt[...,None]
    change = velocity[:,1:]-velocity[:,:-1]
    acceleration = 2*change/(dt[:,1:]+dt[:,:-1])[...,None]
    harmonic_dt = 2*dt[:,1:]*dt[:,:-1]/(dt[:,1:]+dt[:,:-1])
    bend = change*harmonic_dt[...,None]
    return increment, velocity, acceleration, bend


def orthogonal_basis(raw):
    return torch.linalg.qr(raw, mode='reduced').Q


def projection_residual(delta, basis):
    coefficients = torch.einsum('nd,ndr->nr',delta,basis)
    residual = delta-torch.einsum('ndr,nr->nd',basis,coefficients)
    return residual.square().sum(-1)
