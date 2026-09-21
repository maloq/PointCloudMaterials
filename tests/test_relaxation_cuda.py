import numpy as np
import pytest
from src.hardware_benchmark.relaxation import read_force_dump, force_metrics


def test_force_dump_preserves_atom_identity_and_precision(tmp_path):
    p=tmp_path/'forces.dump'
    p.write_text('ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type fx fy fz\n1 1 0.1234567891234567 0 -2\n2 1 1 2 3\n')
    f=read_force_dump(p,np.array([1,2]))
    assert f[0,0]==float('0.1234567891234567')
    with pytest.raises(AssertionError):read_force_dump(p,np.array([2,1]))
    p.write_text(p.read_text().replace('fx fy fz','x y z'))
    with pytest.raises(ValueError,match='schema'):read_force_dump(p,np.array([1,2]))


def test_force_error_units_and_relative_normalization():
    a=np.array([[1.,2.,2.]])
    metrics=force_metrics(a,a+.1)
    assert metrics['force_component_rms_error_eV_per_A']==pytest.approx(.1)
    assert metrics['force_relative_rms_error']==pytest.approx(.1/np.sqrt(3))
    with pytest.raises(ValueError,match='Unpaired'):force_metrics(a,a[0])
