"""Queue dependency/deadline behavior, without consuming an allocated GPU."""
import json
import pytest
import torch
from src.training_methods.shared_pretraining import queue


def test_partial_stage_prevents_dependent_training(monkeypatch):
    calls=[]
    monkeypatch.setattr(queue,'execute_stage',lambda path,phase,deadline: calls.append(phase) or False)
    assert queue.run_pipeline({'structural':'parent','causal':'child','analysis':'probes'},float('inf')) is False
    assert calls==['structural']
    with pytest.raises(RuntimeError,match='Final allocated continuation'):
        queue.run_pipeline({'structural':'parent'},float('inf'),final_slot=True)


def test_serial_resume_stops_at_partial_variant_without_slurm(tmp_path,monkeypatch):
    plan_path=tmp_path/'plan.json'
    plan=dict(output=str(tmp_path/'output'),runs=[dict(name=name,configs=dict(structural=name)) for name in ['mace','gatr']])
    plan_path.write_text(json.dumps(plan));calls=[];completed=set()
    def stage(path,phase,deadline):
        calls.append(path)
        return path in completed
    monkeypatch.setattr(queue,'execute_stage',stage)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    deadline='2099-01-01T00:00:00+00:00'
    assert queue.serial(plan_path,deadline) is False
    assert calls==['mace']
    state=tmp_path/'output/technical/queue-state.json'
    assert json.loads(state.read_text())['state']=='checkpointed'
    completed.add('mace');calls.clear()
    assert queue.serial(plan_path,deadline) is False
    assert calls==['mace','gatr']
    completed.add('gatr');calls.clear()
    assert queue.serial(plan_path,deadline) is True
    assert calls==['mace','gatr']
    assert json.loads(state.read_text())['state']=='complete'


def test_expired_deadline_cannot_launch_and_failures_block_queue(tmp_path,monkeypatch):
    plan_path=tmp_path/'plan.json'
    plan_path.write_text(json.dumps(dict(output=str(tmp_path/'output'),runs=[dict(name='mace',configs={'structural':'missing'})])))
    with pytest.raises(ValueError,match='future deadline'):
        queue.serial(plan_path,'2000-01-01T00:00:00+00:00')
    assert not (tmp_path/'output').exists()
    monkeypatch.setattr(queue,'execute_stage',lambda *args: (_ for _ in ()).throw(FloatingPointError('Nonfinite objective')))
    with pytest.raises(FloatingPointError,match='Nonfinite objective'):
        queue.serial(plan_path,'2099-01-01T00:00:00+00:00')
    state=json.loads((tmp_path/'output/technical/queue-state.json').read_text())
    assert state['state']=='failed' and 'Nonfinite objective' in state['error']
