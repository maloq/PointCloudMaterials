import numpy as np
import torch

from src.research.gatr_conditional_information.data import radial_control,future_labels
from src.research.gatr_conditional_information.metrics import matched_pairs,pair_mask,improvement
from src.research.gatr_conditional_information.probe import predict_path


def test_radial_control_forgets_angles_and_neighbor_order():
    rng=np.random.default_rng(21)
    radius=np.r_[np.linspace(2.,7.8,90),np.linspace(8.,16.,10)]
    x=rng.normal(size=(100,3));x=x/np.linalg.norm(x,axis=1,keepdims=True)*radius[:,None]
    y=rng.normal(size=(100,3));y=y/np.linalg.norm(y,axis=1,keepdims=True)*radius[:,None]
    x=np.vstack((np.zeros((1,3)),x));y=np.vstack((np.zeros((1,3)),y))[rng.permutation(101)]
    a=radial_control(x,9.121389139452193,33);b=radial_control(y,9.121389139452193,33)
    np.testing.assert_allclose(a[0],b[0],atol=1e-12)
    np.testing.assert_allclose(np.sort(np.linalg.norm(a[0],axis=1))[1:],radius[:90],atol=2e-6)


def test_future_excludes_existing_crystal_and_unobserved_confirmation():
    labels=np.zeros((40,2),int);labels[15:25,0]=1;labels[37:,1]=1
    config=dict(future_horizons_ps=[3,6,9],confirmation_frames=4,negative_history_frames=3)
    eligible,y,onset=future_labels(labels,1.,config)
    np.testing.assert_array_equal(onset,[15,40])
    assert eligible[12,0] and y[12,0].all()
    assert not y[5,0].any()
    assert not eligible[15:,0].any()
    assert not eligible[:2].any() and not eligible[28:].any()
    assert not y[eligible[:,1],1].any()


def test_matching_uses_same_source_time_and_radial_density_only():
    a=dict(source=np.array([1,1,1,2]),frame=np.array([0,0,1,0]),atom=np.array([3,4,3,3]),
        radii80=np.ones((4,80)),radial_quantiles=np.ones((4,33)),order=np.ones((4,8)))
    config=dict(match_density_relative=.02)
    p=matched_pairs(a,config)
    np.testing.assert_array_equal(p['left'],[0]);np.testing.assert_array_equal(p['right'],[1])
    assert pair_mask(p,.025,config).all()
    a['order'][1,6]=1.1
    assert not pair_mask(matched_pairs(a,config),.025,config).any()


def test_source_improvement_uses_paired_losses():
    result=improvement([1.,4.],[.5,2.],np.array([[0,0],[0,1],[1,1]]))
    assert result['improvement_percent']==50 and result['low']==50 and result['high']==50


def test_gpu_probe_recovers_linear_signal_without_test_moment_leakage():
    assert torch.cuda.is_available(),'Run this audit test on its requested A100'
    rng=np.random.default_rng(42);x=rng.normal(size=(300,4));coef=np.array([1.,2.,-1.,.5]);y=(x@coef)[:,None]
    test=rng.normal(size=(30,4));sources=np.repeat(np.arange(3),100)
    a,mean,scale=predict_path(x,y,test,sources,'linear',[1e-9],123,64,False)
    b,mean2,scale2=predict_path(x,y,np.vstack((test,1000*np.ones((1,4)))),sources,'linear',[1e-9],123,64,False)
    np.testing.assert_allclose(a[0,:,0],test@coef,atol=1e-6)
    np.testing.assert_allclose(a,b[:,:30],atol=1e-10)
    np.testing.assert_array_equal(mean,mean2);np.testing.assert_array_equal(scale,scale2)


def test_block_scaling_does_not_amplify_unseen_near_empty_tail():
    rng=np.random.default_rng(71);x=rng.normal(size=(300,6));x[:,-1]*=1e-10
    y=(x[:,0]-x[:,1])[:,None];test=x[:2].copy();test[1]=test[0];test[1,-1]=1
    p,_,_=predict_path(x,y,test,np.repeat(np.arange(3),100),'linear',[1e-4],1,64,False,pooled_tail=3)
    assert abs(p[0,0,0]-p[0,1,0])<1e-6
