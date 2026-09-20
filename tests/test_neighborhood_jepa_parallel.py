"""Global-statistics dual-GPU replay and wider MACE snapshot checks."""
import pytest
import torch
from e3nn import o3
from test_neighborhood_jepa import sample
from test_neighborhood_jepa_v2 import fcc,fixture
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.neighborhood_jepa.model import IRREPS
from src.training_methods.neighborhood_jepa.v2.model import Model,Encoder
from src.training_methods.neighborhood_jepa.v2.objective import Objective
from src.training_methods.neighborhood_jepa.v2.parallel import ParallelEncoder
from src.training_methods.shared_pretraining.compilation import compile_encoder


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual wider cuEquivariance CUDA encoder')
def test_wide_mace_rotation_and_parameter_increase():
    torch.manual_seed(9)
    small=Encoder(32).cuda().eval()
    large=Encoder(64).cuda().eval()
    assert sum(p.numel() for p in large.parameters())>2*sum(p.numel() for p in small.parameters())
    x=fcc();r=o3.rand_matrix()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=large(move(collate([sample(x)],'mace'),'cuda'))
        b=large(move(collate([sample(x@r.numpy().T)],'mace'),'cuda'))
    torch.testing.assert_close(a[:,:128],b[:,:128],atol=.002,rtol=.002)
    torch.testing.assert_close(a[:,128:]@IRREPS.D_from_matrix(r).cuda().T,b[:,128:],atol=.0002,rtol=.002)


@pytest.mark.skipif(torch.cuda.device_count()!=2,reason='Exactly two allocated GPUs required')
@pytest.mark.parametrize('compiled',[False,True])
def test_two_gpu_global_loss_and_gradient_parity(compiled):
    torch._dynamo.reset()
    torch.manual_seed(91)
    spec,plan,manifest,target,_=fixture('E',n=4)
    a,b=Model(64).cuda(),Model(64).cuda()
    b.load_state_dict(a.state_dict())
    oa,ob=Objective(manifest,spec).cuda(),Objective(manifest,spec).cuda()
    samples=[sample(fcc()*(1+.003*i)) for i in range(4*len(plan.views))]
    batches=[collate(samples[i:i+7],'mace') for i in range(0,len(samples),7)]
    if compiled:compile_encoder(b.encoder,move(batches[0],'cuda'),'bf16')
    runner=ParallelEncoder(b,batches[0],'bf16',compiled)
    try:
        target=move(target,'cuda')
        torch.manual_seed(19)
        with torch.autocast('cuda',dtype=torch.bfloat16):encoded=a.encoder(move(collate(samples,'mace'),'cuda')).float()
        loss,_=oa(a,encoded,target)
        loss.backward()
        torch.manual_seed(19)
        value,terms,diagnostics=runner.step(b,ob,batches,target)
        assert diagnostics['independent_anchors']==4
        assert abs(float(loss.detach())-value)<.002
        ga=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
        gb=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
        relative=float((ga-gb).norm()/ga.norm())
        assert relative<.01,relative
        torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
        optimizer=torch.optim.AdamW(b.parameters(),lr=.0002)
        optimizer.step()
        runner.synchronize()
        for primary,secondary in zip(b.encoder.parameters(),runner.encoders[1].parameters()):
            torch.testing.assert_close(primary,secondary.to('cuda:0'),rtol=0,atol=0)
    finally:runner.close()
