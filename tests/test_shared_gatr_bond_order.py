import copy
import itertools
import numpy as np
import pytest
import torch
from torch import nn

from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.bond_order import BOND_IRREPS,bond_order_errors
from src.models.encoders.equivariant_bond import gatr_bond_features
from src.models.encoders.mixed_gatr import MixedBondGATr
from src.models.encoders.mixed_mace import training_encode
from src.training_methods.shared_pretraining.mixed import BondObjective
from src.training_methods.shared_pretraining.runtime import cached_update
from test_shared_mace_bond_order import sample
from e3nn import o3
from gatr.interface import embed_point


def test_atom_tensor_pooling_retains_fcc_order_when_mean_vector_vanishes():
    x=torch.tensor([p for p in itertools.product((-1.,0.,1.),repeat=3) if sum(v*v for v in p)==2.])[None]
    assert torch.count_nonzero(x.mean(1))==0
    mv=embed_point(x).unsqueeze(-2)
    features=gatr_bond_features(mv,torch.ones(1,12),o3.SphericalHarmonics([4,6],False,'component'))
    assert features.shape==(1,88)
    assert features[:,:36].norm()>.1 and features[:,36:].norm()>.1


@pytest.mark.parametrize('reflection',[False,True])
def test_actual_gatr_bond_covariance_export_and_gradients(reflection):
    torch.manual_seed(29);model=MixedBondGATr([('Al','a',False)]).eval()
    a=sample(2);rotation=o3.rand_matrix(dtype=torch.float64)*(-1 if reflection else 1)
    b=copy.deepcopy(a);b['positions']=a['positions']@rotation.numpy().astype(np.float32).T
    first=collate([a],'gatr',bond_order=True);second=collate([b],'gatr',bond_order=True)
    encoded=training_encode(model,first);rotated=training_encode(model,second)
    assert encoded.shape==(1,832)
    torch.testing.assert_close(model.encoder(first),encoded[:,:128],rtol=0,atol=0)
    q=model.bond_order(encoded[:,128:]);qr=model.bond_order(rotated[:,128:])
    d=BOND_IRREPS.D_from_matrix(rotation).float()
    torch.testing.assert_close(qr,q@d.T,atol=2e-5,rtol=2e-4)
    torch.testing.assert_close(bond_order_errors(q,first['bond_order']),
        bond_order_errors(qr,second['bond_order']),atol=2e-5,rtol=2e-4)
    bond_order_errors(q,first['bond_order']).mean().backward()
    assert sum(p.grad.norm() for p in model.encoder.parameters() if p.grad is not None)>0
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.encoder.readout.parameters())
    torch.testing.assert_close(first['bond_order'],collate([a],'mace',bond_order=True)['bond_order'],rtol=0,atol=0)


class CacheEncoder(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(5,832)
    def forward(self,batch,return_equivariant=False):
        value=self.linear(batch['features']);return value if return_equivariant else value[:,:128]


@pytest.mark.parametrize('temporal',[False,True])
def test_gatr_bond_gradient_replay_and_no_context_labels(temporal):
    torch.manual_seed(17);keys=[('Al','a',False),('Mg','b',False)]
    model=MixedBondGATr(keys);model.encoder=CacheEncoder();other=copy.deepcopy(model)
    norm={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('physical',85),('tda',144)]}
    objective=BondObjective(norm,keys,.1,.001,.1)
    n=4;count=(3 if temporal else 2)*n
    batch=dict(features=torch.randn(count,5),physical=torch.randn(count,85),tda=torch.randn(count,144),
        tda_valid=torch.ones(count,dtype=torch.bool),bond_order=torch.randn(count,22))
    for key in ('physical','tda','bond_order'):batch[key][2*n:]=float('nan')
    batch['tda_valid'][2*n:]=False
    extra=dict(domain=torch.tensor([0,1,0,1]))
    if temporal:extra['triplet_dt']=torch.tensor([[.1,.2]]*n)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    loss,_=objective(model,training_encode(model,batch),batch|extra,temporal,torch.ones(n))
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    chunks=[{k:v[a:a+3] for k,v in batch.items()} for a in range(0,count,3)]
    cached_update(other,objective,chunks,torch.optim.SGD(other.parameters(),lr=.001),temporal,[.1]*n,extra)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5,msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Compiled BF16 GATr path requires CUDA')
def test_compiled_bf16_gatr_bond_covariance_and_backward():
    from src.training_methods.shared_pretraining.compilation import compile_encoder
    torch.manual_seed(33);model=MixedBondGATr([('Al','a',False)]).cuda().eval()
    a=sample(2,24);b=copy.deepcopy(a);rotation=o3.rand_matrix(dtype=torch.float64)
    b['positions']=a['positions']@rotation.numpy().astype(np.float32).T
    first=move(collate([a],'gatr',bond_order=True),'cuda')
    second=move(collate([b],'gatr',bond_order=True),'cuda')
    compile_encoder(model.encoder,first,'bf16')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=training_encode(model,first);rotated=training_encode(model,second)
        q=model.bond_order(encoded[:,128:]);qr=model.bond_order(rotated[:,128:])
        loss=bond_order_errors(q,first['bond_order']).mean()
    assert q.dtype==torch.float32 and encoded.shape==(1,832)
    d=BOND_IRREPS.D_from_matrix(rotation).float().cuda()
    torch.testing.assert_close(qr,q@d.T,atol=2e-5,rtol=3e-4)
    loss.backward()
    gradients=[p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients) and sum(g.norm() for g in gradients)>0
