"""Ni EAM full-cell BAOAB continuations and analytic local tangent observations."""
import torch

from src.simulation.eam_gpu.potential import FastNi
from src.research.continuous_motion.teacher import archive, load, collect as collect_teacher
from .data import resolve_path


def _potential(c, atoms, box):
    files = {'Ni': resolve_path(c['potential_file'])}
    potential = FastNi(files['Ni'], atoms,
                       box=torch.tensor(box, device='cuda', dtype=torch.float64),
                       dtype=torch.float32, expected_sha256=c['potential_sha256'])
    return potential, files


def collect(c, plan, job, deadline):
    return collect_teacher(c, plan, job, deadline, mass=58.71, potential_factory=_potential)
