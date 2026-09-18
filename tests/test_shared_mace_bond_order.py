import copy
import itertools

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from torch import nn

from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.bond_order import (
    BOND_IRREPS,bond_order_targets,bond_order_magnitudes,bond_order_errors)
from src.models.encoders.mixed_mace import MixedSnapshotMACE,EquivariantBondOrder,training_encode
from e3nn import o3
from src.training_methods.shared_pretraining.mixed import MACEBondObjective
from src.training_methods.shared_pretraining.runtime import cached_update


def sample(seed,n=20):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,3)).astype(np.float32)*2;x[0]=0
    pairs=cKDTree(x).query_pairs(5.,output_type='ndarray')
    return dict(positions=x[None],weights=np.ones((1,n),np.float32),center=0,times=np.array([0.],np.float32),
        species=seed%5,log_scale=.1,physical=rng.normal(size=85).astype(np.float32),
        tda=rng.normal(size=144).astype(np.float32),tda_valid=True,
        edges=np.concatenate((pairs,pairs[:,::-1])).T)


def test_q_targets_match_existing_complex_bond_order_and_known_fcc():
    from src.analysis.liquid_structure import bond_order
    vectors=torch.randn(6,12,3)
    scalar,_=bond_order(np.tile(vectors.numpy()[:,None],(1,13,1,1)),3.)
    np.testing.assert_allclose(bond_order_magnitudes(bond_order_targets(vectors)).numpy(),scalar[:,:2],rtol=3e-6,atol=1e-7)
    fcc=torch.tensor([p for p in itertools.product((-1.,0.,1.),repeat=3) if sum(v*v for v in p)==2.])[None]
    torch.testing.assert_close(bond_order_magnitudes(bond_order_targets(fcc)),torch.tensor([[.19094065,.57452426]]),atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError,match='nonzero'):bond_order_targets(torch.zeros(1,12,3))


@pytest.mark.parametrize('reflection',[False,True])
def test_head_and_targets_are_covariant_and_loss_invariant_in_bf16_context(reflection):
    torch.manual_seed(7);rotation=o3.rand_matrix(dtype=torch.float64)
    if reflection:rotation=-rotation
    inputs=o3.Irreps('4x0e + 4x1o + 4x2e');d=inputs.D_from_matrix(rotation).float()
    out_d=BOND_IRREPS.D_from_matrix(rotation).float()
    head=EquivariantBondOrder(channels=4);h=torch.randn(5,36,requires_grad=True)
    bonds=torch.randn(5,12,3);q=bond_order_targets(bonds)
    qr=bond_order_targets(bonds@rotation.float().T)
    rotated_h=h@d.T
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y=head(h);yr=head(rotated_h)
    assert y.dtype==torch.float32
    torch.testing.assert_close(yr,y@out_d.T,atol=2e-5,rtol=2e-4)
    torch.testing.assert_close(qr,q@out_d.T,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(bond_order_errors(y,q),bond_order_errors(yr,qr),atol=2e-5,rtol=2e-5)
    bond_order_errors(y,q).mean().backward()
    assert h.grad[:,16:].norm()>0 and h.grad[:,:16].norm()==0


def test_collation_targets_pack_centers_and_ignore_atom_permutation():
    a=sample(2);b=sample(3,24);batch=collate([a,b],'mace',bond_order=True)
    assert batch['packed_centers'].tolist()==[0,20]
    perm=np.random.default_rng(6).permutation(24);c=copy.deepcopy(b)
    c['positions']=b['positions'][:,perm];c['center']=int(np.flatnonzero(perm==0)[0])
    # Edge indices matter to the encoder, but not to target production.
    inverse=np.argsort(perm);c['edges']=inverse[b['edges']]
    other=collate([c],'mace',bond_order=True)
    torch.testing.assert_close(batch['bond_order'][1:],other['bond_order'],atol=1e-7,rtol=1e-6)


class CacheEncoder(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(5,416)
    def forward(self,batch,return_equivariant=False):
        value=self.linear(batch['features']);return value if return_equivariant else value[:,:128]


@pytest.mark.parametrize('temporal',[False,True])
def test_bond_gradient_cache_matches_full_batch_and_past_is_not_supervised(temporal):
    torch.manual_seed(17);keys=[('Al','a',False),('Mg','b',False)]
    model=MixedSnapshotMACE(keys,backend='e3nn');model.encoder=CacheEncoder();other=copy.deepcopy(model)
    normalization={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('physical',85),('tda',144)]}
    objective=MACEBondObjective(normalization,keys,.1,.001,.1)
    n=4;count=(3 if temporal else 2)*n
    batch=dict(features=torch.randn(count,5),physical=torch.randn(count,85),tda=torch.randn(count,144),
        tda_valid=torch.ones(count,dtype=torch.bool),bond_order=torch.randn(count,22))
    for key in ('physical','tda','bond_order'):batch[key][2*n:]=float('nan')
    batch['tda_valid'][2*n:]=False
    extra=dict(domain=torch.tensor([0,1,0,1]))
    if temporal:extra['triplet_dt']=torch.tensor([[.1,.2]]*n)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    loss,terms=objective(model,training_encode(model,batch),batch|extra,temporal,torch.ones(n))
    assert terms['bond_order_weighted']>0
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    chunks=[{k:v[a:a+3] for k,v in batch.items()} for a in range(0,count,3)]
    cached_update(other,objective,chunks,torch.optim.SGD(other.parameters(),lr=.001),temporal,[.1]*n,extra)
    for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,atol=2e-6,rtol=2e-5,msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual cuEquivariance BF16 path requires GPU')
def test_cuda_mace_bf16_rotation_and_default_export():
    torch.manual_seed(29);model=MixedSnapshotMACE([('Al','a',False)]).cuda().eval()
    a=sample(2);b=copy.deepcopy(a);rotation=o3.rand_matrix(dtype=torch.float64)
    b['positions']=a['positions']@rotation.numpy().astype(np.float32).T
    first=move(collate([a],'mace',bond_order=True),'cuda')
    second=move(collate([b],'mace',bond_order=True),'cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=training_encode(model,first);rotated=training_encode(model,second)
        q=model.bond_order(encoded[:,128:]);qr=model.bond_order(rotated[:,128:])
        exported=model.encoder(first)
    assert encoded.shape==(1,416) and exported.shape==(1,128)
    # cuEquivariance's floating-point scatter order is not bit deterministic.
    torch.testing.assert_close(exported,encoded[:,:128],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(encoded[:,:128],rotated[:,:128],atol=2e-4,rtol=2e-4)
    d=BOND_IRREPS.D_from_matrix(rotation).float().cuda()
    torch.testing.assert_close(qr,q@d.T,atol=1e-5,rtol=3e-4)
