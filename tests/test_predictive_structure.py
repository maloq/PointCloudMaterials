"""Scientific weighting and uncertainty controls for supervised encoder training."""
import numpy as np
import torch
from src.training_methods.predictive_structure import current_loss,future_errors
from src.analysis.predictive_structure import cluster_interval


def test_repeating_al_does_not_change_material_balanced_geometry_loss():
    target=torch.tensor([1.,2.,3.])[:,None].expand(-1,88)
    prediction=torch.zeros_like(target);material=torch.tensor([0,1,2])
    reference=current_loss(prediction,target,material)
    indices=torch.tensor([0]*100+[1,2])
    torch.testing.assert_close(current_loss(prediction[indices],target[indices],material[indices]),reference)


def test_mobility_has_equal_family_weight_despite_fewer_components():
    target=torch.zeros(4,72);target[:,66:]=1
    errors=future_errors(torch.zeros_like(target),target)
    torch.testing.assert_close(errors,torch.tensor([[0.,0.,1.]]).expand(4,-1))
    torch.testing.assert_close(errors.mean(),torch.tensor(1/3))


def test_source_bootstrap_does_not_gain_precision_by_duplicating_atoms():
    sources=np.repeat(np.arange(3),2)
    error=np.array([.8,.9,.6,.8,.9,1.1]);baseline=np.ones(6)
    draws=np.array([[0,1,2],[0,0,1],[1,1,2],[2,2,2],[0,0,0]])
    score,ci=cluster_interval(error,baseline,sources,draws)
    repeated_score,repeated_ci=cluster_interval(np.repeat(error,20),np.repeat(baseline,20),np.repeat(sources,20),draws)
    np.testing.assert_allclose([score,*ci],[repeated_score,*repeated_ci])
