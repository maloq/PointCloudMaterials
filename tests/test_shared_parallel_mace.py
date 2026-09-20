"""Actual dual-GPU parity, including global grouped VICReg and tensor supervision."""
import copy
import numpy as np
import pytest
import torch

from src.data.structural_pretraining.batches import collate, move
from src.models.encoders.mixed_mace import MixedSnapshotMACE
from src.training_methods.shared_pretraining.mixed import BondObjective
from src.training_methods.shared_pretraining.parallel_mace import ParallelMACE
from src.training_methods.shared_pretraining.runtime import cached_update
from test_shared_mace_bond_order import sample


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='Requires two real CUDA devices')
@pytest.mark.parametrize('temporal,precision',[(False,'float32'),(True,'float32'),(True,'bf16')])
def test_parallel_matches_single_global_update_and_syncs_replicas(temporal,precision):
    torch.set_num_threads(2);torch.manual_seed(17)
    keys=[('Al','a',False),('Mg','b',False)]
    model=MixedSnapshotMACE(keys).cuda()
    other=MixedSnapshotMACE(keys).cuda();other.load_state_dict(model.state_dict())
    normalization={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('physical',85),('tda',144)]}
    objective=BondObjective(normalization,keys,.1,.001,.1).cuda()
    n=6;count=n*(3 if temporal else 2)
    # Uneven graph sizes and chunks deliberately put each domain on both devices.
    batches=[collate([sample(i,20+i%4) for i in range(a,min(a+5,count))],'mace',bond_order=True)
             for a in range(0,count,5)]
    extra=dict(domain=np.array([0,1,0,1,0,1]))
    if temporal:extra['triplet_dt']=np.array([[.1,.2]]*n)
    with torch.no_grad():other.encoder(move(batches[0],'cuda'))
    parallel=ParallelMACE(other,[0,1],batches[0],precision)
    try:
        first=torch.optim.SGD(model.parameters(),lr=.003)
        second=torch.optim.SGD(other.parameters(),lr=.003)
        for _ in range(2):
            reference=cached_update(model,objective,batches,first,temporal,[.2]*n,extra,precision)
            result=parallel.update(other,objective,batches,second,temporal,[.2]*n,extra)
            assert result['loss']==pytest.approx(reference['loss'],rel=3e-5,abs=3e-6)
            for (name,p),(_,q) in zip(model.named_parameters(),other.named_parameters(),strict=True):
                torch.testing.assert_close(p,q,atol=3e-6,rtol=3e-5,msg=name)
            for p,q in zip(other.encoder.parameters(),parallel.encoders[1].parameters(),strict=True):
                torch.testing.assert_close(p,q.to(p.device),atol=0,rtol=0)
        assert other.state_dict().keys()==model.state_dict().keys()
    finally:parallel.close()
