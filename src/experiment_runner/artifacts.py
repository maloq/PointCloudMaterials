"""Readable run outputs and explicit access to the two repository analysis layouts."""

import json
from pathlib import Path


def result_folders(root):
    root = Path(root)
    for name in ('plots', 'tables', 'technical'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def analysis_artifacts(root):
    """Existing analyses keep their paths; new analyses put intermediate files in technical/."""
    root = Path(root)
    if ((root / 'analysis_metrics.json').is_file() or
            (root / 'analysis_inference_cache.npz.meta.json').is_file()):
        return root  # Repository layout before September 12, 2026.
    return root / 'technical'



def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
