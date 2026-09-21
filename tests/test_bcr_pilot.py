import numpy as np
from src.research.bcr_pilot.data import patches_from_snapshot,read_melt
from src.training_methods.bcr.data import extract_patch


def test_fast_patch_matches_general_periodic_images():
    rng=np.random.default_rng(2);cell=np.diag([20.,23.,21.]);x=rng.uniform(size=(500,3))*cell.diagonal();ids=np.arange(500)+1;centers=[0,100,499]
    patches,identities,images=patches_from_snapshot(x,cell,ids,centers,5.,256)
    for j,center in enumerate(centers):
        general=extract_patch(x,cell,center,ids,5.)
        a={int(i):v for i,v in zip(identities[j],patches[j])};b={int(i):v for i,v in zip(general['atom_ids'],general['positions'])}
        assert a.keys()==b.keys()
        for i in a:np.testing.assert_allclose(a[i],b[i],atol=1e-6)


def test_melt_box_precision_is_not_silently_float32(tmp_path):
    path=tmp_path/'melt.lammpstrj';length=111.86084474178199
    path.write_text('ITEM: TIMESTEP\n100000\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n'+f'0 {length:.17g}\n'*3+'ITEM: ATOMS id type x y z\n1 1 0 0 0\n2 1 2 3 4\n')
    header,table=read_melt(path)
    assert header['box_high'].dtype==np.float64 and header['box_high'][0]==length
    assert table.shape==(2,5)
