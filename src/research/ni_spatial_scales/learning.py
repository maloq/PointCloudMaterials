"""Matched regional-input models with the shared local-measurement trainer."""
from functools import partial

from src.research.ni_kinetic_latent import learning as shared
from src.research.ni_kinetic_latent.model import Model
from .data import load, FAMILY


class SpatialModel(Model):
    """Encode local244 + regional87; reconstruct only central local244."""
    def __init__(self, c, kind):
        super().__init__(c, kind, input_dimension=331)


def fit(c, arm, deadline):
    return shared.fit(c, arm, deadline, load_provider=partial(load, arm=arm),
                      model_factory=SpatialModel, family=FAMILY, input_key='input',
                      input_normalization_key='context_normalization', run_prefix='niscales')


def fitted(c, arm):
    return shared.fitted(c, arm, model_factory=SpatialModel)


def benchmark(c, arm=None):
    arm = c['arms'][0] if arm is None else arm
    return shared.benchmark(c, load_provider=partial(load, arm=arm),
                            model_factory=SpatialModel, input_key='input', kinds=('hybrid',),
                            report_name='benchmark-' + arm['scale'])
