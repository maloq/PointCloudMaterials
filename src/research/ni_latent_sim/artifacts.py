"""Keep recovery exports separate from results made by an older frozen worker."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from src.experiment_runner import metric_docs
from src.experiment_runner.artifacts import write_json


@lru_cache(maxsize=1)
def contract():
    return json.loads((metric_docs.DOCUMENTS/'contracts.json').read_text())['ni_latent_sim']['files']


def export_root(folder):
    previous = folder/'technical/metric-contracts/ni_latent_sim.json'
    if not previous.exists():
        previous = folder/'technical/metric-contract.json'
    if previous.exists() and json.loads(previous.read_text())['files'] != contract():
        revision = hashlib.sha256(json.dumps(contract(), sort_keys=True).encode()).hexdigest()[:12]
        return folder/'analyses'/('recovery-metrics-'+revision)
    return folder


def write_rows(rows, folder, **kwargs):
    destination = export_root(folder)
    path = metric_docs.write_metric_rows(rows, destination, **kwargs)
    write_json(folder/'technical/metric-exports'/(kwargs['name']+'.json'), dict(path=str(path)))
    return path


def table_path(folder, name):
    pointer = folder/'technical/metric-exports'/(name+'.json')
    if pointer.exists():
        return Path(json.loads(pointer.read_text())['path'])
    return folder/'tables'/(name+'.csv')
