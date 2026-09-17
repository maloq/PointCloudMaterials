import torch
from src.research.local_predictability.onset_model import OnsetModel
from src.research.local_predictability.metrics import hazard_loss
from test_local_predictability_native import observation


def test_onset_training_reaches_encoder_and_retains_nested_initialization():
    torch.manual_seed(20260919)
    parent=OnsetModel('snapshot',activation_checkpoint=False)
    child=OnsetModel('history12',activation_checkpoint=False)
    child.load_state_dict(parent.state_dict())
    obs=observation();condition=torch.zeros(1,7)
    a=parent([obs],condition);b=child([obs],condition)
    torch.testing.assert_close(a['logits'],b['logits'],atol=2e-6,rtol=2e-5)
    hazard_loss(b['logits'],torch.tensor([2])).mean().backward()
    gate=child.encoder.history_alpha.grad
    assert torch.isfinite(gate).all() and torch.all(gate.abs()>1e-10)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in child.encoder.parameters())


def test_native_training_checkpoint_and_equal_continuations(tmp_path):
    import numpy as np
    import pytest
    from src.research.local_predictability.supervised import fit_stage
    if not torch.cuda.is_available():pytest.skip('Native training integration requires allocated CUDA device')
    obs=observation().to('cuda')
    class Windows:
        rows=[dict(source_id=i,center_id=1,anchor=64,split='train') for i in range(8)]
        def observation(self,index,variant):return obs
        def statistics(self):return {}
    windows=Windows();labels=dict(event_bin=np.array([0,1,2,3,4,5,6,6]),source_id=np.arange(8))
    config=dict(output=str(tmp_path),activation_checkpoint=False,max_spatial_edges=1000000,
        updates_per_stage=2,training_deadline_utc='2099-01-01T00:00:00+00:00',microbatch=dict(snapshot=8,history12=8,repeat12=8))
    cond=torch.zeros(8,7,device='cuda');event=torch.tensor(labels['event_bin'],device='cuda')
    splits=dict(train=np.arange(8));identity={'integration_test':True}
    parent=fit_stage(config,windows,labels,cond,event,splits,list(range(8)),identity,'parent')
    for variant in ['snapshot','history12','repeat12']:
        checkpoint=fit_stage(config,windows,labels,cond,event,splits,list(range(8)),identity,variant,parent)
        state=torch.load(checkpoint,weights_only=False,map_location='cpu')
        assert state['step']==2 and state['stage']==variant
        assert 'optimizer' in state and 'sampler' in state and 'cuda_rng' in state
