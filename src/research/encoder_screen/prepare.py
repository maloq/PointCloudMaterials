"""Cache full source neighborhoods once, without recomputing physical labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .common import load_config,sha, checked, write


def prepare(config):
    root=Path(config['output']); ref=Path(config['reference'])
    manifest=json.loads((ref/'manifest.json').read_text())
    folder=root/'technical/inputs'; folder.mkdir(parents=True,exist_ok=True)
    receipt=dict(reference=str(ref.resolve()), reference_sha256=sha(ref/'manifest.json'), frames=[], files={})
    for record in manifest['frames']:
        i=record['frame_index']; reference=ref/f'frame-{i:02d}.npz'
        checked(reference,manifest['files'][reference.name])
        path=folder/f'frame-{i:02d}.npz'
        checked(record['file'],record['input_sha256'])
        points=np.load(record['file']); a=np.load(reference)
        np.testing.assert_array_equal(points[a['rows']],a['coords'])
        tree=cKDTree(points)
        # Ten Angstroms covers every declared <=8*scale/9.192189 support.
        if np.minimum(a['coords']-tree.mins,tree.maxes-a['coords']).min() <= 10:
            raise ValueError('Reference centers do not support the 10 A observation cache')
        rows=tree.query_ball_point(a['coords'],10.,return_sorted=True,workers=1)
        centers=[]; patches=[]; near80=[]
        for atom,ids in zip(a['rows'],rows,strict=True):
            ids=np.asarray(ids); centers.append(int(np.flatnonzero(ids==atom).item()))
            patches.append((points[ids].astype(np.float64)-points[atom].astype(np.float64)).astype(np.float32))
        _, nearest=tree.query(a['coords'],k=80,workers=1)
        for atom, ids in zip(a['rows'],nearest,strict=True):
            near80.append((points[ids].astype(np.float64)-points[atom].astype(np.float64)).astype(np.float32))
        np.savez(path,positions=np.concatenate(patches),offsets=np.r_[0,np.cumsum([len(p) for p in patches])],
                 centers=np.asarray(centers),nearest80=np.stack(near80))
        receipt['frames'].append(record); receipt['files'][path.name]=sha(path)
        print('prepared',record['material'],i,flush=True)
    # Native future patches remain physical/relaxed, with original-MD targets.
    from src.research.geoframe_evolution.prediction import prepare as future_prepare
    corpus,cfg,_=future_prepare()
    from src.project_runtime.paths import resolve_path
    future=resolve_path(cfg['cache'])/'relaxed-graphs.npz'
    receipt.update(future_graphs=str(future),future_graphs_sha256=sha(future),
                   future_identity=corpus.manifest['identity'],future_config=cfg)
    write(folder/'manifest.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    prepare(load_config(p.parse_args().config))
