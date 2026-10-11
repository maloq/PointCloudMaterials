"""Single-element tabulated EAM with CUDA forces and analytic force tangents."""
from .potential import FastNi
from .dynamics import CapturedNVE

__all__ = ['FastNi', 'CapturedNVE']
