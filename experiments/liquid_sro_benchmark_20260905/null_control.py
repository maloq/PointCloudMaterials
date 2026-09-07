"""Destroy local structural information while retaining split/temperature marginals."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((Path(__file__).parent/'config.json').read_text());out=root/cfg['output']
    meta=dict(np.load(out/'metadata.npz'));rng=np.random.default_rng(20260910)
    permutation=np.arange(len(meta['split']))
    for split in ('train','val','test'):
        for material in range(3):
            for temperature in np.unique(meta['temperature']):
                ids=np.flatnonzero((meta['split']==split)&(meta['material']==material)&(meta['temperature']==temperature))
                permutation[ids]=rng.permutation(ids)
    np.save(out/'null_permutation.npy',permutation)
    np.save(out/'embeddings/ShuffledSOAP.npy',np.load(out/'embeddings/SOAP.npy')[permutation])


if __name__=='__main__':main()
