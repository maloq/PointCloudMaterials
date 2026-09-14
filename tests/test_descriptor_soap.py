"""Characterize the maintained SOAP baseline's retired-helper replacement."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.analysis.pipeline_runtime import _resolve_analysis_module_class
from src.baselines.descriptor_baselines import SOAPDescriptorBaseline


def test_soap_matches_historical_constructor():
    pytest.importorskip("dscribe")
    from ase import Atoms

    soap = SOAPDescriptorBaseline._build_soap_descriptor(
        species="Al", r_cut=4.0, n_max=2, l_max=1, sigma=0.3
    )
    atoms = Atoms("Al3", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    actual = soap.create(atoms, centers=[0])
    # Generated with build_soap from 256f006^, before its package was retired.
    expected = [[1.052142649137587, 0.581292683101143,
                 0.3211552955332571, 0.001386484580877551,
                 -0.007647114182216291, 0.04217742924976539]]
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_retired_motif_checkpoint_requires_recorded_source():
    with pytest.raises(ValueError, match="recorded source snapshot"):
        _resolve_analysis_module_class(
            OmegaConf.create({"model_type": "temporal_motif_field"})
        )
