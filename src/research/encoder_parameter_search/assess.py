"""Supplement the fixed snapshot assay, preserving all historical measurements."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np
from src.research.encoder_screen.common import write,sha
from src.experiment_runner.metric_docs import write_metric_table
from .metrics import frame


def supplement(folder,reference,output):
    folder=Path(folder);reference=Path(reference);output=Path(output)
    complete=json.loads((folder/'complete.json').read_text())
    manifest=json.loads((reference/'manifest.json').read_text())
    if sha(reference/'manifest.json') != complete['reference_sha256']:
        raise ValueError('Classical reference differs from the completed evaluation')
    result={}
    with threadpool_limits(limits=1):
        for record in manifest['frames']:
            i=record['frame_index'];path=folder/'embeddings'/f'frame-{i:02d}.npz'
            if record['material'] not in complete['task']['materials']: continue
            if sha(reference/f'frame-{i:02d}.npz')!=manifest['files'][f'frame-{i:02d}.npz']:
                raise ValueError(f'Changed classical frame {i}')
            if sha(path)!=complete['feature_files'][path.name]: raise ValueError(f'Changed embeddings: {path}')
            with np.load(path) as features,np.load(reference/f'frame-{i:02d}.npz') as arrays:
                for rep in complete['extraction']['representations']:
                    result[f'frame_{i:02d}_{record["material"]}_{rep}']=frame(features[rep],arrays,record)
    write(output/'technical/metrics.json',result)
    write(output/'technical/evidence.json',dict(source=str(folder.resolve()),
        complete_sha256=sha(folder/'complete.json'),reference_sha256=sha(reference/'manifest.json')))
    write_metric_table(result,output,family='encoder_parameter_search')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--folder',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();supplement(a.folder,a.reference,a.output)
