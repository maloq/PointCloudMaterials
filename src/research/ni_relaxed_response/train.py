"""Matched Ni training with finite responses of quenched observations."""
from src.research.ni_response.train import run as run_matched
from .data import FAMILY
from .targets import short_objective


def run(c, p, arm, deadline, micro):
    return run_matched(
        c, p, arm, deadline, micro,
        objective=short_objective,
        family=FAMILY,
        run_name_prefix='ni-relaxed-esen-',
    )
