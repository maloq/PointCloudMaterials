"""Performance paths must preserve sampling, labels and selection inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.training_methods.neighborhood_jepa.v2.data import Data, pack
from src.training_methods.neighborhood_jepa.v2.runtime import selection_phases
from src.training_methods.neighborhood_jepa.regularization.release import check_gate
from test_neighborhood_jepa import sample


def test_loader_groups_io_but_preserves_sampler_order_and_repeats():
    class Rows(Data):
        def __init__(self):
            self.rows=[({'id':v},0) for v in ['b','a','b','c']]
            self.visited=[]
        def __getitem__(self,index):
            self.visited.append(self.rows[index][0]['id'])
            return index
    data=Rows()
    result=next(iter(DataLoader(data,batch_sampler=[[3,0,1,2,0]])))
    assert result.tolist()==[3,0,1,2,0]
    assert data.visited==['a','b','b','b','c']


def test_selection_pack_keeps_current_graphs_and_all_targets():
    samples=[]
    for row in range(3):
        samples.append(dict(views=[sample(np.array([[0,0,0],[1+slot,0,0]],np.float32)) for slot in range(4)],
            moments=np.full((4,120),row,np.float32),position=np.zeros((7,3),np.float32),times=np.array([-.75,0,.75],np.float32),
            physical=np.zeros((2,85),np.float32),tda=np.zeros((2,144),np.float32),query_atom_ids=np.arange(7),
            frame=2,index=row,group=0,temperature_K=450.))
    all_batches,all_targets=pack(samples,8)
    selected,targets=pack(samples,8,view_slots=[2])
    assert sum(len(b['log_scale']) for b in all_batches)==12
    assert sum(len(b['log_scale']) for b in selected)==3
    torch.testing.assert_close(selected[0]['packed_positions'],torch.tensor([[0,0,0],[3,0,0]]*3,dtype=torch.float32))
    for key in targets:torch.testing.assert_close(targets[key],all_targets[key],rtol=0,atol=0)


def test_selection_labels_cached_and_use_tracked_ids(tmp_path,monkeypatch):
    import src.training_methods.neighborhood_jepa.v2.runtime as runtime
    cache=tmp_path/'cache';parent=tmp_path/'parent';folder=cache/'12';shard=parent/'shards'/'s'
    folder.mkdir(parents=True);shard.mkdir(parents=True)
    np.save(folder/'atom_ids.npy',[30,10,20])
    np.save(folder/'labels.npy',[[0,1],[0,2],[0,3]])
    np.save(shard/'query_atom_ids.npy',[[20],[30],[10]])
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(config=dict(cache=str(cache)),sources=[dict(lineage='l',id=12)])))
    record=dict(id='s',lineage='l',frame=1)
    data=SimpleNamespace(parent=parent,selection=[2,0,1],rows=[(record,i) for i in range(3)])
    calls=[]
    monkeypatch.setattr(runtime,'resolve_path',lambda p:(calls.append(p),Path(p))[1])
    np.testing.assert_array_equal(selection_phases(data,dict(crystallization_plan=str(plan))),[2,3,1])
    assert len(calls)==2
    # Repeated evaluations require neither path resolution nor reopening labels.
    monkeypatch.setattr(runtime.np,'load',lambda *a,**kw:pytest.fail('reloaded immutable selection labels'))
    np.testing.assert_array_equal(selection_phases(data,{}),[2,3,1])


def test_release_rejects_dead_gate(tmp_path):
    path=tmp_path/'release.json'
    path.with_suffix('.heartbeat.json').write_text(json.dumps(dict(state='holding',time=0)))
    with pytest.raises(RuntimeError,match='not live'):check_gate(path)


def test_release_gate_protects_only_unstarted_tasks(tmp_path):
    import fcntl
    import multiprocessing
    import time
    from src.training_methods.neighborhood_jepa.regularization.release import gate
    root=tmp_path/'campaign'/'technical';root.mkdir(parents=True)
    tasks=[dict(type='fit',name=name) for name in ('active','unstarted')]
    (root/'tasks.json').write_text(json.dumps(tasks))
    for t in tasks:(root/'runs'/t['name']).mkdir(parents=True)
    active=root/'runs/active';pending=root/'runs/unstarted'
    (active/'manifest.json').write_text('{}')
    (root/'lane-0.json').write_text(json.dumps(dict(state='running',task='active')))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(output=str(root.parent))))
    code=tmp_path/'code';module=code/'src/training_methods/neighborhood_jepa/regularization/release.py'
    module.parent.mkdir(parents=True);module.touch()
    receipt=tmp_path/'release.json'
    process=multiprocessing.get_context('spawn').Process(target=gate,args=(str(config),str(code),str(receipt)))
    process.start()
    try:
        until=time.monotonic()+45
        while not receipt.exists() and time.monotonic()<until:
            assert process.is_alive()
            time.sleep(.1)
        assert json.loads(receipt.read_text())['tasks']==['fit/unstarted']
        with (pending/'worker.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (pending/'worker-performance-20260920.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (pending/'status.json').write_text(json.dumps(dict(state='complete')))
        process.join(15)
        assert process.exitcode==0
        with (pending/'worker.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        if process.is_alive():process.terminate();process.join()
