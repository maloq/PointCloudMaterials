"""Thin BCR workflow entry point; benchmark/verification never run inside training."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from .runtime import train,load_data
from .prepare import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','verify','evaluate','probes','select'])
    p.add_argument('--config',required=True);p.add_argument('--device',default='cpu');p.add_argument('--checkpoint');p.add_argument('--split',choices=['development','test'],default='development')
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    from src.project_runtime.paths import resolve_path
    data=resolve_path(c['data']);output=resolve_path(c['output'])
    if a.stage=='prepare':prepare(c['preparation'],data)
    elif a.stage=='train':train(c['training'],data,output,a.device)
    elif a.stage=='verify':
        from .verify import verify
        verify(c,data,output,a.device)
    elif a.stage=='select':
        from .probes import select_checkpoint
        rows=json.loads((output/'technical/checkpoint-assessments.json').read_text());chosen=select_checkpoint(rows)
        (output/'technical/selection.json').write_text(json.dumps(dict(checkpoint=chosen,criterion='G2/G3 then development reconstruction')))
    else:
        from .analysis import analyze
        analyze(c,data,output,a.checkpoint,a.split,a.device,probes=a.stage=='probes')

if __name__=='__main__':main()
