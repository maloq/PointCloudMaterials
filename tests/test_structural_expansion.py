import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data.structural_pretraining.expand import shooting_tasks, expand_task
from src.data.structural_pretraining.prepare import file_hash
from src.analysis.liquid_structure import persistence_image
from src.training_methods.shared_pretraining.initialization import initialize_structural, require_complete_tda
from src.models.encoders.structural import StructuralModel, ARCHITECTURE_REVISION
from src.training_methods.structural_pretraining.objective import Objective


def test_extra_shooting_covers_sources_preserves_ancestry_and_avoids_old_frames():
    sources=[dict(split=split,stratum='al_shooting',lineage=root,frame_count=12,id=str(i))
             for i,(root,split) in enumerate([('a','train'),('a','train'),('b','train'),('held','selection')])]
    parent=dict(sources=sources,tasks=[dict(source=0,frame=3)])
    tasks=shooting_tasks(parent,24,19,5)
    assert tasks==shooting_tasks(parent,24,19,5)
    assert sum(t['count'] for t in tasks)==24
    assert {t['source'] for t in tasks}=={0,1,2}
    assert len({(t['source'],t['frame']) for t in tasks})==len(tasks)
    assert all((t['source'],t['frame'])!=(0,3) for t in tasks)
    assert sum(t['count'] for t in tasks if sources[t['source']]['lineage']=='a')==12
    assert all(t['split']=='train' and 0<t['count']<=5 for t in tasks)


@pytest.mark.parametrize('static',[False,True])
def test_full_tda_expansion_matches_producer_and_never_modifies_parent(tmp_path,static):
    old=tmp_path/'old';new=tmp_path/'new';folder=old/'shards/000000';folder.mkdir(parents=True)
    rng=np.random.default_rng(6);nviews=2 if static else 5
    positions=rng.normal(size=(nviews*95,3)).astype(np.float32)*2
    ids=np.tile(np.arange(95),nviews)
    mapping=np.array([[-1,-1,0,-1,1]] if static else [[0,1,2,3,4]])
    valid=np.zeros(nviews,dtype=bool);valid[0 if static else 2]=True
    target=np.zeros((nviews,144),dtype=np.float32)
    def reference(view):
        x=positions[view*95:(view+1)*95]
        nearest=np.lexsort((ids[:95],np.square(x.astype(np.float64)).sum(-1)))[:80]
        return persistence_image(x[nearest])
    target[np.flatnonzero(valid)[0]]=reference(np.flatnonzero(valid)[0])
    arrays=dict(positions=positions,atom_ids=ids,offsets=np.arange(nviews+1)*95,
                views=mapping,tda=target,tda_valid=valid)
    for k,v in arrays.items():np.save(folder/f'{k}.npy',v)
    task=dict(id='000000',source=0,frame=0,count=1,seed=8,split='train')
    hashes={k:file_hash(folder/f'{k}.npy') for k in arrays}
    record=dict(identity='old',task=task,static=static,hashes=hashes)
    (folder/'complete.json').write_text(json.dumps(record))
    args=(str(new),str(old),{},task,1.,'new',True)
    result=expand_task(args);dest=new/'shards/000000'
    endpoints=mapping[:,[2,4] if static else [2,3,4]].ravel()
    np.testing.assert_allclose(np.load(dest/'tda.npy')[endpoints],np.stack([reference(i) for i in endpoints]))
    assert np.array_equal(np.flatnonzero(np.load(dest/'tda_valid.npy')),np.sort(endpoints))
    assert all(file_hash(folder/f'{k}.npy')==v for k,v in hashes.items())
    assert (folder/'positions.npy').stat().st_ino==(dest/'positions.npy').stat().st_ino
    assert (folder/'tda.npy').stat().st_ino!=(dest/'tda.npy').stat().st_ino
    assert result['labelled_views']==len(endpoints)
    assert expand_task(args)==result


def test_transfer_keeps_encoder_and_physical_predictions_when_target_units_change():
    torch.manual_seed(4);model=StructuralModel('gatr').eval()
    def norm(mean,std):return {k:dict(mean=[mean]*n,std=[std]*n) for k,n in [('physical',85),('tda',144)]}
    old=Objective(norm(2.,3.),'vicreg');new=Objective(norm(-1.,7.),'vicreg')
    config=dict(architecture='gatr',phase='structural',method='vicreg',history_frames=1)
    saved=dict(model=copy.deepcopy(model.state_dict()),objective=old.state_dict(),step=120,best=.2,
               identity=dict(config=config,architecture_revision=ARCHITECTURE_REVISION))
    states=torch.randn(8,128)
    before={k:v.detach().clone() for k,v in model.heads(states).items()}
    initialize_structural(model,new,saved,config)
    after=model.heads(states)
    for name in ('physical','tda'):
        torch.testing.assert_close(before[name]*3+2,after[name]*7-1,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(before['q'],after['q'],rtol=0,atol=0)
    for k,v in model.encoder.state_dict().items():torch.testing.assert_close(v,saved['model']['encoder.'+k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='history_frames'):
        initialize_structural(model,new,saved,dict(config,history_frames=3))


def test_full_tda_requirement_checks_each_endpoint_but_not_context_frames():
    a=dict(views=np.array([[0,1,2,3,4]]),tda_valid=np.array([False,False,True,True,True]),tda=np.ones((5,144)))
    release=SimpleNamespace(manifest=dict(tda_coverage='all_supervised_views',shards=[dict(task=dict(id='x'),static=False)]),arrays={'x':a})
    require_complete_tda(release)
    a['tda_valid'][3]=False
    with pytest.raises(ValueError,match='Missing supervised TDA'):require_complete_tda(release)
