"""Measure whether the hard-80 TDA target itself jumps in the same crossings."""

from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np

from src.analysis.liquid_structure import persistence_image
from src.data_utils.topology_targets import fit_targets
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from src.research.mace_tda_ridge_audit.math import balanced_errors
from .data import load_clouds, read_json
from .evaluate import crossing_clouds


def label_crossings(config):
    root=Path(config['output'])
    destination=root/'technical/crossing-labels.npz'
    if destination.exists():
        raise FileExistsError(destination)
    manifest=read_json(Path(config['cache'])/'manifest.json')
    boundaries=[]
    for record in manifest['temporal']:
        clouds=load_clouds(Path(config['cache'])/record['file'])
        selected=np.flatnonzero(np.array(record['rows']) % 17 == 8)[:4]
        np.testing.assert_array_equal(selected,np.arange(64,68))
        boundaries.extend(clouds[i] for i in selected)
    reference=np.load(root/'technical/frozen-mean80/features.npz')
    epsilons=reference['epsilons']
    if reference['crossing_z'].shape[:3] != (len(epsilons),len(boundaries),2):
        raise ValueError('TDA label crossings do not match the embedding intervention cohort')
    inputs=[x[:80] for epsilon in epsilons for cloud in boundaries for x in crossing_clouds(cloud,epsilon)]
    with ProcessPoolExecutor(max_workers=config['cpu_threads']) as pool:
        labels=np.stack(list(pool.map(persistence_image,inputs,chunksize=8))).reshape(len(epsilons),len(boundaries),2,144)
    probes=np.load(Path(config['diagnostics'])/'technical/probes.npz')
    temporal=np.load(Path(config['diagnostics'])/'technical/temporal.npz')
    scales=fit_targets(probes['hot'][probes['split']=='train'],32,.05)['block_scale']
    natural=np.diff(temporal['hot'],axis=1).reshape(-1,144)
    denominator=balanced_errors(np.zeros_like(natural),natural,scales)[0].mean()
    rows=[]
    for epsilon,y in zip(epsilons,labels,strict=True):
        error=balanced_errors(y[:,1],y[:,0],scales)[0].mean()
        rows.append(dict(epsilon_A=float(epsilon),balanced_tda_mse=float(error),
                         fraction_of_075ps_tda_energy=float(error/denominator)))
    np.savez(destination,labels=labels,epsilons=epsilons,block_scales=scales)
    write_json(root/'technical/label-crossing.json',dict(rows=rows,natural_075ps_tda_energy=float(denominator),patches=len(boundaries)))
    snapshot_metric_docs(root,'mace_context')
    with (root/'tables/label-crossing.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(rows,flush=True)
