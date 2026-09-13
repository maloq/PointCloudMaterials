"""Physical score projection and matched source/seed transition comparison."""

import numpy as np
import torch

from src.research.forecast_spatial_mixture.evaluate import score_source, assess
from src.research.forecast_spatial_mixture.compare import paired_f1
from src.training_methods.embedding_forecast.context_mixture import ContextMixtureForecaster, crystal_probability
from test_forecast_context_mixture import variant


def test_batched_physical_projection_preserves_center_origin_and_spatial_history():
    torch.manual_seed(10)
    model = ContextMixtureForecaster(3, 4, .75, [.75, 1.5, 2.25], variant(spatial=2)).eval()
    z = torch.randn(2, 9, 3); spatial = torch.randn_like(z); radii = torch.ones(2, 9, 2)*7
    anchors = np.array([3, 4, 5]); mean = torch.randn(3); scale = torch.rand(3)+.5
    weight = torch.randn(3, dtype=torch.float64); bias = torch.tensor(.2, dtype=torch.float64)
    result = score_source(z, spatial, radii, anchors, model, mean, scale, weight, bias, 2)
    for center in range(2):
        for origin, anchor in enumerate(anchors):
            history = (z[center: center+1, anchor-3:anchor+1]-mean)/scale
            neighbor = (spatial[center: center+1, anchor-3:anchor+1]-mean)/scale
            with torch.inference_mode():
                output = model(history, neighbor, radii[center:center+1, anchor-3:anchor+1])
                margin = output['mean'].double()@(scale.double()*weight)+mean.double()@weight+bias
                probability = crystal_probability(output, scale.double()*weight, mean.double()@weight+bias)
            np.testing.assert_allclose(result['mean_margin'][center, origin], margin[0].numpy(), atol=1e-6)
            np.testing.assert_allclose(result['frame_crystal_probability'][center, origin], probability[0].numpy(), atol=1e-6)


def test_local_assay_keeps_censored_negatives_and_reports_missed_event_timing():
    crystal = np.zeros((2, 2, 16), dtype=bool)
    crystal[:, 0, 8:] = True
    anchors = np.arange(3, 11)
    scores = np.full((2, 2, len(anchors), 4), -1., dtype=np.float32)
    for s in range(2):
        scores[s, 0] = (anchors[:, None]+np.arange(1, 5) >= 8)*2.-1.
    data = dict(anchors=anchors, crystal=crystal, sources=[dict(split='val', source_index=1), dict(split='test', source_index=2)])
    plan = dict(cadence_ps=.75, horizons_ps=[3], persistence_frames=[3], negative_history_frames=3,
                bootstrap_repetitions=20, bootstrap_seed=5)
    perfect, clustered = assess(scores, data, plan, 'mean_margin')
    metric = perfect['onset'][0]
    assert metric['f1'] == 1 and metric['timing_mae_ps'] == 0
    assert metric['tn'] > 0 and metric['timed_within_1_5_ps_recall'] == 1
    scores[1, 0] = -1
    missed, _ = assess(scores, data, plan, 'mean_margin')
    assert missed['onset'][0]['recall'] == 0
    assert missed['onset'][0]['timing_mae_ps'] is None
    assert clustered['mean_margin_3ps_p3'].shape == (1, 7)


def test_paired_source_bootstrap_averages_seeds_before_f1():
    candidate = np.array([[[4, 0, 1, 5], [8, 0, 2, 10]], [[2, 0, 3, 5], [4, 0, 6, 10]]])
    baseline = np.array([[[1, 0, 4, 5], [2, 0, 8, 10]], [[1, 0, 4, 5], [2, 0, 8, 10]]])
    result = paired_f1(candidate, baseline, 100, 4)
    expected = 2*3/(2*3+2)-2*1/(2*1+4)
    np.testing.assert_allclose(result['event_f1_difference'], expected)
    np.testing.assert_allclose(result['ci95'], [expected, expected])


def test_anchor_only_spatial_mixture_uses_only_current_observations():
    from src.training_methods.embedding_forecast.context_mixture import mixture_nll
    from src.training_methods.embedding_forecast.augmentation import augment_history
    model = ContextMixtureForecaster(3, 1, .75, [.75, 1.5, 2.25], variant(spatial=2))
    history = torch.randn(2, 1, 3)
    augmented = augment_history(history, dict(noise_std=.01, frame_dropout=.15))
    torch.testing.assert_close(augmented, history, rtol=0, atol=0)
    torch.testing.assert_close(model.past_time, torch.zeros(1), rtol=0, atol=0)
    spatial = torch.randn_like(history).requires_grad_()
    output = model(augmented, spatial, torch.ones(2, 1, 2)*7)
    loss = mixture_nll(output, torch.randn(2, 3, 3)).mean()
    loss.backward()
    assert torch.isfinite(loss) and spatial.grad.abs().sum() > 0
