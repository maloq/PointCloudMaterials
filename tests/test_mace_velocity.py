"""Physical symmetries and atom identity at the velocity input boundary."""
import numpy as np
import torch

from src.models.encoders.mace_context import make_context_graph
from src.models.encoders.mace_velocity import velocity_moments, weighted_pool
from src.research.mace_velocity.data import motion_observables


def sample():
    rng=np.random.default_rng(73)
    points=rng.normal(size=(240,3))*5
    points[0]=0
    velocities=rng.normal(size=points.shape)*3
    return points.astype(np.float32),velocities.astype(np.float32)


def pooled(x,v):
    g=make_context_graph([x],'halo_inner',device='cpu')
    e,o=velocity_moments(g,torch.from_numpy(v))
    return torch.cat((weighted_pool(e,g),weighted_pool(o,g)),1).numpy()


def test_velocity_messages_symmetries_and_nonzero_signal():
    x,v=sample();rng=np.random.default_rng(31)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    p=np.r_[0,rng.permutation(np.arange(1,len(x)))]
    original=pooled(x,v)
    for xx,vv in [(x@q,v@q),(x[p],v[p]),(x+3,v),(x,v+np.array([9,-4,11]))]:
        np.testing.assert_allclose(pooled(xx.astype(np.float32),vv.astype(np.float32)),original,rtol=2e-4,atol=2e-6)
    reverse=pooled(x,-v)
    np.testing.assert_allclose(reverse[:,:33],original[:,:33])
    np.testing.assert_allclose(reverse[:,33:],-original[:,33:])
    np.testing.assert_allclose(pooled(x,np.zeros_like(v)),0)
    assert np.linalg.norm(original)>0.1


def test_pruned_graph_retains_original_velocity_alignment_across_clouds():
    x,v=sample();clouds=[x,x[:180]+np.array([20,0,0],np.float32)]
    g=make_context_graph(clouds,'halo_inner',device='cpu')
    np.testing.assert_array_equal(g.positions.numpy(),np.concatenate(clouds)[g.input_index.numpy()])
    np.testing.assert_array_equal(g.input_batch.numpy(),(g.input_index.numpy()>=len(x)).astype(int))


def test_local_velocity_fit_distinguishes_expansion_rotation_and_translation():
    x,_=sample()
    translation=motion_observables(x,np.ones_like(x)*7)
    np.testing.assert_allclose(translation,0,atol=1e-10)
    expansion=motion_observables(x,.2*x)
    # The physical fit uses a small isotropic 0.001 A^2 ridge, so affine
    # reconstruction has a bounded, nonzero shrinkage error.
    assert abs(expansion[6]-.6)<2e-4
    assert expansion[3]<1e-8 and expansion[4]<1e-8 and expansion[5]<3e-8
    rotation=motion_observables(x,np.cross(x,np.array([0,0,.2])))
    assert abs(rotation[6])<1e-4 and rotation[3]<1e-8 and rotation[4]>.07
    reversed_values=motion_observables(x,-.2*x)
    np.testing.assert_allclose(reversed_values[:6],expansion[:6])
    np.testing.assert_allclose(reversed_values[6:],-expansion[6:])


def test_paired_dump_conversion_matches_ids_and_times(tmp_path):
    from src.data.conversion.paired_velocity import convert
    paths=[tmp_path/'positions.lammpstrj',tmp_path/'velocities.lammpstrj']
    for path,fields in zip(paths,['x y z','vx vy vz'],strict=True):
        with path.open('w') as stream:
            for step in range(4):
                stream.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 40\n0 40\n0 40\nITEM: ATOMS id type {fields}\n')
                if fields=='x y z':stream.write(f'2 1 2 3 {step}\n1 1 1 2 {step}\n')
                else:stream.write(f'1 1 {step} 0 1\n2 1 {step+1} 0 2\n')
    result=convert(*paths,tmp_path/'binary',frame_count=1,atom_count=2)
    np.testing.assert_array_equal(result.timesteps,[1,2])
    np.testing.assert_array_equal(result.positions[:,0,0],[1,1])
    np.testing.assert_array_equal(result.velocities[:,0,0],[1,2])
    assert paths[0].exists() and paths[1].exists()
