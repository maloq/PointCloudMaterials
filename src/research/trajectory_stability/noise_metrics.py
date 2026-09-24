"""Source-balanced input-noise response in a fixed training-reference metric."""
import numpy as np

from .spectrum import source_weights


def noise_response(clean, perturbed, repeated, temporal_delta, reference_trace,
                   source, input_mse_A2, sigma_A):
    clean, perturbed, repeated, temporal_delta = [np.asarray(x, dtype=np.float64)
        for x in (clean, perturbed, repeated, temporal_delta)]
    if clean.ndim != 2 or not len(clean) or any(x.shape != clean.shape or not np.isfinite(x).all()
            for x in (clean, perturbed, repeated, temporal_delta)):
        raise ValueError('Finite clean/noisy/repeated/temporal arrays must have identical row/feature shapes')
    source, input_mse_A2 = np.asarray(source), np.asarray(input_mse_A2, dtype=float)
    if source.shape != (len(clean),) or input_mse_A2.shape != source.shape or not np.isfinite(input_mse_A2).all() or np.any(input_mse_A2 < 0):
        raise ValueError('Source IDs and nonnegative realized input noise must align with outputs')
    if not np.isfinite(reference_trace) or reference_trace <= 0:
        raise ValueError('Positive fixed reference variance is required')
    if not np.isfinite(sigma_A) or sigma_A <= 0:
        raise ValueError('Noise sigma must be positive, in Angstrom per coordinate')
    weights = source_weights(source)
    response2 = np.square(perturbed-clean).sum(1)
    floor2 = np.square(repeated-clean).sum(1)
    motion2 = np.square(temporal_delta).sum(1)
    response, floor, motion = [float(np.sqrt(weights@x/(2*reference_trace)))
                               for x in (response2, floor2, motion2)]
    input_rms = float(np.sqrt(weights@input_mse_A2))
    if input_rms <= 0:
        raise ValueError('Nonzero sigma produced zero realized input displacement')
    jump = np.sqrt(response2/(2*reference_trace))
    order = np.argsort(jump, kind='stable')
    cumulative = np.cumsum(weights[order]); cumulative /= cumulative[-1]
    result = dict(rows=len(clean), sources=len(np.unique(source)), sigma_A=float(sigma_A),
        input_rms_displacement_A=input_rms,
        response_rms=response, response_rms_raw=float(np.sqrt(weights@response2)),
        response_p95=float(jump[order[np.searchsorted(cumulative, .95)]]),
        sensitivity_per_A=response/input_rms, repeat_rms=floor,
        temporal_rms_matched_075=motion,
        noise_to_temporal_ratio=response/motion if motion > 0 else None,
        response_to_repeat_ratio=response/floor if floor > 0 else None)
    return result
