"""Scientific controls for the liquid benchmark's geometry and full MACE path."""
import numpy as np
import torch
import pytest
from ase.build import bulk
from scipy.spatial import cKDTree
from src.analysis.liquid_structure import bond_order,persistence_image,nonaffine_displacement
from src.models.encoders.atomic_graph import ReferenceMACEEncoder


def test_fcc_order_and_affine_motion():
    points=bulk('Al','fcc',a=4.05,cubic=True).repeat((8,8,8)).positions
    tree=cKDTree(points)
    center=np.argmin(np.linalg.norm(points-points.mean(0),axis=1))
    _,ids=tree.query(points[center],k=65)
    _,nn=tree.query(points[ids[:13]],k=13)
    vectors=points[nn[:,1:]]-points[ids[:13],None]
    order,connections=bond_order(vectors[None],3.7)
    np.testing.assert_allclose(order[0,:4],[.19094065,.5745243,-.15931737,-.0131606],atol=1e-6)
    np.testing.assert_array_equal(connections,[[12,12,12]])
    x=(points[ids[1:25]]-points[center])[None]
    matrix=np.array([[1.01,.02,0],[0,.99,0],[0,0,1.03]])
    assert nonaffine_displacement(x,x@matrix)[0]<1e-20


def test_alpha_images_ignore_rotation_and_permutation():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(65,3))*2
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    np.testing.assert_allclose(persistence_image(x),persistence_image((x@rotation)[rng.permutation(len(x))]),rtol=1e-5,atol=1e-6)


def edge_list(x,cutoff):
    dist=torch.cdist(x,x)
    edge=((dist<cutoff)&~torch.eye(len(x),device=x.device,dtype=torch.bool)).nonzero()
    return edge[None],torch.tensor([len(edge)])


def test_reference_mace_rotation_and_zero_cutoff_buffer():
    torch.manual_seed(1)
    model=ReferenceMACEEncoder(channels=8,max_ell=2,accelerated=False).double()
    x=torch.randn(1,40,3,dtype=torch.float64)*2
    edges,count=edge_list(x[0],4.25)
    material=torch.tensor([0])
    rotation,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    a=model(x,material,edges,count)
    b=model(x@rotation,material,edges,count)
    torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
    strict,n=edge_list(x[0],4.)
    torch.testing.assert_close(a,model(x,material,strict,n),rtol=1e-8,atol=1e-9)
    a.square().mean().backward()
    grads=[p.grad for name,p in model.named_parameters() if 'conv_tp_weights' in name]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0


def test_reference_mace_complete_two_hop_halo_matches_larger_graph():
    torch.manual_seed(31)
    model=ReferenceMACEEncoder(channels=8,max_ell=2,accelerated=False).double().eval()
    x=torch.randn(1,180,3,dtype=torch.float64)*4
    x[:,0]=0
    inside=x[0].norm(dim=-1)<8.11
    assert inside.sum()<len(inside)
    local=x[:,inside]
    full_edges,full_count=edge_list(x[0],4.25)
    local_edges,local_count=edge_list(local[0],4.25)
    material=torch.tensor([0])
    with torch.no_grad():
        full=model(x,material,full_edges,full_count)
        cropped=model(local,material,local_edges,local_count)
    torch.testing.assert_close(full,cropped,rtol=1e-8,atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Tests the actual fused CUDA backend')
def test_fused_mace_gpu_scalar_rotation_invariance_and_radial_gradients():
    torch.manual_seed(41)
    model=ReferenceMACEEncoder(accelerated=True).cuda()
    x=torch.randn(1,80,3,device='cuda')*2
    x[:,0]=0
    edge,count=edge_list(x[0],4.25);count=count.cuda()
    rotation,_=torch.linalg.qr(torch.randn(3,3,device='cuda'))
    material=torch.tensor([0],device='cuda')
    a=model(x.clone(),material,edge,count)
    b=model((x@rotation).clone(),material,edge,count)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
    strict,strict_count=edge_list(x[0],4.)
    torch.testing.assert_close(a,model(x.clone(),material,strict,strict_count.cuda()),rtol=2e-5,atol=1e-7)
    a.square().mean().backward()
    gradients=[p.grad for name,p in model.named_parameters() if 'conv_tp_weights' in name]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients)>0
