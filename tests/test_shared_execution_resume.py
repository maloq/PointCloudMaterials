import copy
import json
import pytest

from src.data.structural_pretraining.prepare import digest,file_hash
from src.training_methods.shared_pretraining.resume import check_resume,verify_execution_change


def identities():
    a=dict(data='immutable',architecture_revision='same',protocol='same',
           config=dict(batch_size=2048,seed=17,preparation_workers=8),
           implementation=dict(versions=dict(torch='pinned'),files={
               '/frozen/old/src/training_methods/shared_pretraining/runtime.py':'old',
               'src/models/encoders/structural.py':'protected'}))
    b=copy.deepcopy(a);b['config'].update(preparation_workers=1,preparation_processes=4,prefetch_batches=6)
    b['implementation']['files'].pop('/frozen/old/src/training_methods/shared_pretraining/runtime.py')
    b['implementation']['files']['src/training_methods/shared_pretraining/runtime.py']='new'
    return a,b


def test_transition_requires_exact_checkpoint_and_identities(tmp_path):
    a,b=identities();(tmp_path/'last.pt').write_bytes(b'checkpoint')
    receipt=tmp_path/'transition.json'
    receipt.write_text(json.dumps(dict(previous_identity_sha256=digest(a),replacement_identity_sha256=digest(b),
                                      checkpoint_sha256=file_hash(tmp_path/'last.pt'))))
    check_resume(a,b,str(receipt),tmp_path)
    assert (tmp_path/'implementation_transition.json').exists()
    with pytest.raises(ValueError,match='explicit tested'):check_resume(a,b,None,tmp_path)
    (tmp_path/'last.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):check_resume(a,b,str(receipt),tmp_path)


@pytest.mark.parametrize('change',['model','data','batch','seed','versions'])
def test_execution_transition_cannot_change_science(change):
    a,b=identities()
    if change=='model':b['implementation']['files']['src/models/encoders/structural.py']='different'
    elif change=='data':b['data']='different'
    elif change=='versions':b['implementation']['versions']['torch']='different'
    else:b['config']['batch_size' if change=='batch' else 'seed']=5
    with pytest.raises(ValueError):verify_execution_change(a,b)
