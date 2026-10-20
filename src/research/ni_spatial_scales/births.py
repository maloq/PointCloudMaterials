"""Matched radius readouts and controls for information available in context."""
from src.research.ni_kinetic_latent import births as shared
from .data import load, load_births, FAMILY
from .learning import fitted


SPACES = shared.SPACES + ['measurement_context', 'history_measurement_context']


def controls(c, deadline):
    local = next(arm for arm in c['arms'] if arm['scale'] == 'local')
    def central_history(config, bank):
        cohort, target, _ = load_births(config, bank, local)
        return cohort, target
    return shared.controls(c, deadline, load_provider=lambda config: load(config, local),
                           birth_provider=central_history, family=FAMILY)


def _context_spaces(_observed, input_history):
    values = input_history.numpy()
    return dict(measurement_context=values[:, -1],
                history_measurement_context=values.reshape(len(values), -1))


def evaluate(c, arm, deadline):
    return shared.evaluate(c, arm, deadline, load_provider=load, birth_provider=load_births,
        fitted_provider=fitted, family=FAMILY, extra_spaces_provider=_context_spaces,
        encoder_input_description=f"present331: central244 plus {arm['scale']} regional87; no future context or fixed-membership past")


def compare(c):
    extra = []
    for scale in sorted({arm['scale'] for arm in c['arms']}):
        extra += [(scale, 'measurement_context', 'controls', 'measurements'),
                  (scale, 'history_measurement_context', 'controls', 'history_measurements'),
                  (scale, 'measurements_latent', scale, 'measurement_context'),
                  (scale, 'history_measurements_latent', scale, 'history_measurement_context')]
    return shared.compare(c, family=FAMILY, treatment_key='scale', spaces=SPACES,
                          extra_contrasts=extra)
