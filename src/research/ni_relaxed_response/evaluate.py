"""Matched Ni evaluation, using the relaxed cache and metric definitions."""
from src.research.ni_response.evaluate import cohorts, infer, infer_values
from src.research.ni_response.evaluate import evaluate as evaluate_matched
from .data import FAMILY


def evaluate(c, p, name):
    return evaluate_matched(c, p, name, family=FAMILY)
