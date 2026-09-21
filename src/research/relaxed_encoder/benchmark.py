"""Paired tolerance benchmark on the same CPU host, potential and full cells."""
import json
from types import SimpleNamespace
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,geometry_packet
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions,publish
from src.data.conversion.relaxation import read_relaxed,convert
from src.simulation.relaxation import relax_frame
from src.analysis.liquid_structure import persistence_image
from src.training_methods.neighborhood_jepa.regularization.data import order_targets
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.experiment_runner.metric_docs import write_metric_table
from .prepare import settings,paired_clouds,target_cloud


def fidelity(reference,approximate,physical_std,tda_std,order_std,scale):
    result={'coordinate_rms_A':float(np.sqrt(np.mean(np.sum((reference-approximate)**2,axis=-1))))}
    for name,fn,std in [('physical',geometry_packet,physical_std),('tda',persistence_image,tda_std),('order',lambda p:order_targets(p*REFERENCE_RADIUS/scale,scale),order_std)]:
        a=np.stack([fn(target_cloud(x,scale)) for x in reference]);b=np.stack([fn(target_cloud(x,scale)) for x in approximate])
        result[name+'_standardized_mse']=float(np.mean(((a-b)/std)**2))
    return result


def run(plan,ranks):
    c=plan['config'];root=resolve_path(c['output'])/'technical/benchmark';root.mkdir(parents=True,exist_ok=True)
    norm=json.loads(resolve_path(c['normalization_manifest']).read_text())['normalization'];order=json.loads(resolve_path(c['order_manifest']).read_text())
    results={}
    for temperature in (400,450,500):
        source=next(s for s in plan['sources'] if s['pilot_fit'] and s.get('validation_role',s['split'])=='train' and s['temperature_K']==temperature)
        raw=ShootingBinaryTrajectory.load(resolve_path(source['path']));frame=c['frames'][0]
        if file_hash(raw.root/'manifest.json')!=source['manifest_sha256']:raise ValueError('Benchmark source changed')
        low=raw.box_low[frame].astype(np.float64);box=raw.box_high[frame].astype(np.float64)-low;hot=raw.positions[frame].astype(np.float64)
        queries=np.searchsorted(raw.atom_ids,source['pool_atom_ids']);reference=None;seconds=None
        for tol in (.01,.03,.1):
            name=f'T{temperature}-force{tol}';path=root/f'{name}.json'
            if path.exists():
                row=json.loads(path.read_text());cold=np.load(root/f'{name}.npy')
            else:
                work=resolve_path(c['scratch'])/'benchmark'/name;archive=resolve_path(c['archive'])/'benchmark'/name
                absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count);absolute.positions=AbsolutePositions(raw)
                try:
                    if not (work/'metadata.json').exists():relax_frame(absolute,frame,work,settings(plan,ranks,force_tolerance=tol))
                    pos,meta=read_relaxed(work);_,cold,_=paired_clouds(hot,pos-low,box,queries)
                    np.save(root/f'{name}.npy',cold)
                    row=dict(source=source['id'],temperature_K=temperature,tolerance_eV_per_A=tol,seconds=meta['seconds'],fmax_eV_per_A=meta['fmax_eV_per_A'],
                        energy_eV_per_atom=meta['energy_eV']/raw.atom_count,ranks=ranks,atoms=raw.atom_count,metadata=meta)
                    # Keep the double precision dump until archive publication succeeds.
                    convert(work,delete_source=False,local_cloud_dtype='float32')
                    if not archive.exists():publish(work,archive)
                    save_json(path,row)
                except Exception:
                    failure=resolve_path(c['archive'])/'benchmark-failures'/name
                    if work.exists() and not failure.exists():publish(work,failure)
                    raise
            if tol==.01:reference=cold;seconds=row['seconds']
            row.update(fidelity(reference,cold,np.array(norm['physical']['std']),np.array(norm['tda']['std']),np.array(order['std']),c['scale']))
            row.update(speedup=seconds/row['seconds'],seconds_per_64_centers=row['seconds']/64,seconds_per_256_centers=row['seconds']/256)
            results[name]=row;save_json(root/'results.json',results)
            write_metric_table(results,resolve_path(c['output'])/'benchmark',family='relaxed_encoder',name='relaxation-fidelity')
            print(json.dumps({k:v for k,v in row.items() if k!='metadata'}),flush=True)
