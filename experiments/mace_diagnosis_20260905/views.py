"""Measure independent PTM disagreements across actual spatial positive pairs."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OVITO_THREAD_COUNT']='2'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import confusion_matrix
from experiments.smooth_temporal_encoder_20260905.prepare import ptm_labels,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    cfg=json.loads(parser.parse_args().config.read_text())
    pilot,out=ROOT/cfg['pilot'],ROOT/cfg['output']
    manifest=json.loads((pilot/'data/manifest.json').read_text())
    rng=np.random.default_rng(cfg['seed'])
    grouped={m:[[],[]] for m in ('Al','Mg','Ta')}
    for shard in manifest['shards']:
        if shard['split']!='train':
            continue
        prefix=pilot/'data'/shard['stem']
        x=np.load(str(prefix)+'.clouds.npy',mmap_mode='r')
        labels=np.load(str(prefix)+'.metadata.npz')['labels'].ravel()
        count=1536 if shard['material']=='Ta' else 256
        ids=rng.choice(len(labels),count,replace=False)
        spatial=x.reshape(-1,2,192,3)[ids,1]
        other=np.concatenate([ptm_labels(spatial[start:start+128],.15) for start in range(0,len(ids),128)])
        a,b=grouped[shard['material']]
        a.extend(labels[ids].tolist());b.extend(other.tolist())
        print('Spatial pair assay',shard['stem'],flush=True)
    result={}
    for m,(a,b) in grouped.items():
        a,b=np.array(a),np.array(b)
        result[m]=dict(count=len(a),different_ptm_fraction=float((a!=b).mean()),
            anchor_to_spatial_confusion=confusion_matrix(a,b,labels=[0,1,2,3]).tolist())
    write_json(out/'spatial_view_assay.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
