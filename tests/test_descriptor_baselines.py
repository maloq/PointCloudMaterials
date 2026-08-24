from __future__ import annotations

import numpy as np

from src.baselines import (
    CNADescriptorBaseline,
    SOAPDescriptorBaseline,
    SteinhardtDescriptorBaseline,
)


def _point_clouds() -> np.ndarray:
    center = np.zeros((1, 3), dtype=np.float64)
    shell = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    outer = 2.5 * np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )
    points = np.concatenate((center, shell, outer), axis=0)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    deformed = points.copy()
    deformed[1] *= 1.15
    return np.stack((points, points @ rotation.T, deformed))


def test_steinhardt_and_cna_descriptors_are_rotation_invariant() -> None:
    point_clouds = _point_clouds()
    steinhardt = SteinhardtDescriptorBaseline(
        l_values=(4, 6),
        center_atom_tolerance=1e-8,
        shell_min_neighbors=4,
        shell_max_neighbors=10,
        append_shell_size=True,
    )
    cna = CNADescriptorBaseline(
        center_atom_tolerance=1e-8,
        shell_min_neighbors=4,
        shell_max_neighbors=10,
        max_signatures=6,
        append_shell_size=True,
        fit_max_pointclouds=3,
    )

    steinhardt_features = steinhardt.transform(point_clouds)
    cna_features = cna.fit_transform(point_clouds)

    np.testing.assert_allclose(
        steinhardt_features[0], steinhardt_features[1], atol=1e-6
    )
    np.testing.assert_allclose(cna_features[0], cna_features[1], atol=1e-6)


def test_soap_descriptor_runs_directly_through_dscribe() -> None:
    point_clouds = _point_clouds()
    soap = SOAPDescriptorBaseline(
        species="Al",
        point_scale=1.0,
        center_atom_tolerance=1e-8,
        shell_min_neighbors=4,
        shell_max_neighbors=10,
        r_cut=5.0,
        r_cut_multiplier=1.25,
        r_cut_min=1.1,
        n_max=2,
        l_max=2,
        sigma=0.3,
        pca_components=2,
        fit_max_pointclouds=3,
        n_jobs=1,
    )

    features = soap.fit_transform(point_clouds)

    assert features.shape == (3, 2)
    np.testing.assert_allclose(features[0], features[1], atol=1e-6)
