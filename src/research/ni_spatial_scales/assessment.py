"""Shared central-target evaluation with explicitly broader observed inputs."""
from src.research.ni_kinetic_latent import assessment as shared
from .data import load, read, root, cache, FAMILY
from .learning import fitted


def evaluate(c, arm, deadline):
    plan = read(root(c)/'technical/plan.json')
    return shared.evaluate(c, arm, deadline, load_provider=load, fitted_provider=fitted,
        family=FAMILY, input_key='input', labels_path=cache(plan['base_config'])/'labels.pt')
