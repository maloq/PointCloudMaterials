import numpy as np
import pytest
import torch

from src.training_methods.shared_pretraining.input_pipeline import ProcessPrefetch
from src.training_methods.shared_pretraining.mixed import prepare
from test_shared_mixed_triplets import make_release


def test_process_prefetch_matches_serial_order_targets_and_resume(tmp_path):
    release=make_release(tmp_path)
    config=dict(architecture='mace',history_frames=1,method='vicreg',seed=17,batch_size=8,
                minimum_group_size=2,microbatch_size=3,bond_order_weight=.1,
                preparation_processes=2,prefetch_batches=3,preparation_workers=1)
    # Start at a nonzero update, as an exact checkpoint continuation does.
    loader=ProcessPrefetch(release,config,5,9,pin_memory=False)
    try:
        with pytest.raises(ValueError,match='Expected prefetched'):loader.take(6)
        seen=set()
        for step in range(5,9):
            expected=prepare(release,step,config,pin_memory=False)
            actual=loader.take(step);seen.add(actual[1])
            assert actual[1:6]==expected[1:6]
            for key,value in expected[-1].items():np.testing.assert_array_equal(actual[-1][key],value)
            assert len(actual[0])==len(expected[0])
            for a,b in zip(actual[0],expected[0],strict=True):
                assert a.keys()==b.keys()
                for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0,equal_nan=True)
        assert seen=={False,True}
    finally:loader.close()


def test_process_errors_reach_training_instead_of_hanging(tmp_path):
    release=make_release(tmp_path)
    config=dict(architecture='invalid',history_frames=1,method='vicreg',preparation_processes=1,prefetch_batches=1)
    loader=ProcessPrefetch(release,config,0,1,pin_memory=False)
    try:
        with pytest.raises(ValueError,match='Mixed triplets'):loader.take(0)
    finally:loader.close()
