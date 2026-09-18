"""Raw latent diagnostics must resolve variation below the feature mean."""

import numpy as np

from src.vis_tools.latent_analysis_vis import save_pca_visualization


def test_raw_pca_variance_matches_centered_energy_for_small_float32_changes(tmp_path):
    rng = np.random.default_rng(123)
    variation = rng.normal(size=(4000, 8)) * np.arange(1, 9) * 1e-4
    latents = (np.linspace(-2, 2, 8) + variation).astype(np.float32)
    # Independent reference: form centered products explicitly, rather than
    # subtracting a large mean product from an uncentered second moment.
    centered = latents.astype(np.float64)
    centered -= centered.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)[::-1]
    expected = eigenvalues / np.sum(eigenvalues)
    result = save_pca_visualization(latents, np.empty(0), tmp_path)
    np.testing.assert_allclose(result['explained_variance_ratio'], expected, rtol=1e-8, atol=1e-12)
    assert result['n_components_95_var'] == np.searchsorted(expected.cumsum(), .95) + 1
