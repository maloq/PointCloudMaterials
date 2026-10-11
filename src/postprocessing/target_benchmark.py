"""Local-only target preparation timing and numerical audit on existing patches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.research.liquid_predictability.descriptors import patch_descriptors,patch_descriptor_batch

VARIANTS=('reference','full_safe','spatial_safe','spatial_fast','native_full','native_spatial','native_batch','native_cna_batch','native_selected')


def full(variant):
    return variant in ('reference','full_safe','native_full')


def families_for(variant,selected=None):
    if full(variant):return ('geometry','bond_order','cna','tda')
    if variant=='native_cna_batch' or (variant=='native_selected' and any(n.startswith('cna/') for n in selected)):
        return ('bond_order','cna','tda')
    return ('bond_order','tda')


@lru_cache(None)
def reference(path):
    # The pre-edit source is an explicit audit input, never a production fallback.
    sys.path.insert(0,str(Path(path).parent))
    spec=importlib.util.spec_from_file_location('reference_descriptors',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def calculate(points, variant, old, selected=None):
    if variant=='reference':return reference(old).patch_descriptors(points)
    options=dict(families=families_for(variant,selected))
    if variant=='spatial_fast':options['alpha_precision']='fast'
    if variant.startswith('native'):options['alpha_backend']='native'
    if variant=='native_selected':options['selected_columns']=selected
    return patch_descriptors(points,**options)


def initialize(sample,old,variants,selected):
    global _sample, _old, _selected
    _sample=np.load(sample,mmap_mode='r');_old=old;_selected=selected
    for variant in variants:
        calculate(_sample[0],variant,old,selected)


def block(task):
    variant,begin,stop=task
    if variant in ('native_batch','native_cna_batch','native_selected'):
        return patch_descriptor_batch(_sample[begin:stop],families=families_for(variant,_selected),
            selected_columns=_selected if variant=='native_selected' else None)[0]
    return np.stack([calculate(p,variant,_old)[0] for p in _sample[begin:stop]])


def run(args):
    output=Path(args.output);tech=output/'technical';tech.mkdir(parents=True,exist_ok=True)
    cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    packed=Path(args.packed);manifest=json.loads((packed/'manifest.json').read_text())
    columns=manifest['columns'];names=[c['name'] for c in columns]
    indices=[];meta=[];seen=set();offset=0
    for shard in manifest['shards']:
        task=shard['task'];key=task['source']
        if key not in seen:
            rows=np.linspace(offset,offset+shard['rows']-1,min(args.rows_per_source,shard['rows']),dtype=int)
            indices.extend(rows.tolist());seen.add(key)
            meta.extend([dict(source=key,material=task['material'],lineage=task['lineage'],row=int(row)) for row in rows])
        offset+=shard['rows']
    x=np.load(packed/'positions.npy',mmap_mode='r')[indices]
    expected=np.load(packed/'features.npy',mmap_mode='r')[indices]
    scale=np.load(packed/'target-standardization.npz')['scale'].astype(np.float64)
    sample=cache/'positions.npy';np.save(sample,x)
    np.save(cache/'cached-targets.npy',expected)
    write_json(tech/'input.json',dict(packed=str(packed),manifest_sha256=sha(packed/'manifest.json'),
        packed_identity=manifest['identity'],rows=meta,positions_sha256=sha(sample),
        cached_targets_sha256=sha(cache/'cached-targets.npy'),reference_sha256=sha(Path(args.reference))))
    from src.research.liquid_predictability.rich_multimaterial_data import descriptor_implementation
    implementation=descriptor_implementation()
    variants=args.variants
    selection=None
    if 'native_selected' in variants:
        from src.postprocessing.descriptors.selection import load
        selection=load(dict(path=args.selection,sha256=sha(Path(args.selection))))
    def mask_for(variant):
        return np.array([n in selection if variant=='native_selected' else
            n.split('/')[0] in families_for(variant,selection) for n in names])
    warm={}
    for variant in variants:
        print(json.dumps(dict(stage='warmup',variant=variant,host=platform.node())),flush=True)
        start=time.perf_counter();_,current=calculate(x[0],variant,args.reference,selection)
        warm[variant]=time.perf_counter()-start
        selected=np.array(names)[mask_for(variant)].tolist()
        if current!=selected:raise ValueError('Descriptor names/order changed: '+variant)
    values={};times={v:[] for v in variants}
    for repeat in range(args.repeats):
        for variant in np.random.default_rng(repeat).permutation(variants):
            start=time.perf_counter()
            values[variant]=(patch_descriptor_batch(x,families=families_for(variant,selection),
                selected_columns=selection if variant=='native_selected' else None)[0]
                if variant in ('native_batch','native_cna_batch','native_selected') else np.stack([calculate(p,variant,args.reference)[0] for p in x]))
            times[variant].append(time.perf_counter()-start)
            print(json.dumps(dict(stage='single',repeat=repeat,variant=str(variant),seconds=times[variant][-1])),flush=True)
            np.save(cache/(str(variant)+'.npy'),values[variant])
        write_json(tech/'timing-progress.json',dict(single_worker_seconds=times))
    comparisons=[]
    for variant in variants:
        mask=mask_for(variant)
        baseline=values['reference'][:,mask].astype(np.float64);error=values[variant].astype(np.float64)-baseline
        selected=np.array(names)[mask];families=np.array([c['family'] for c in columns])[mask]
        for family in np.unique(families):
            take=families==family;e=error[:,take];z=e/scale[mask][take]
            flat=int(np.abs(e).argmax());row,col=np.unravel_index(flat,e.shape)
            comparisons.append(dict(variant=variant,family=family,rows=len(x),columns=int(take.sum()),
                max_absolute=float(np.abs(e).max()),rms_standardized=float(np.sqrt(np.mean(z*z))),
                max_standardized=float(np.abs(z).max()),unequal_values=int(np.count_nonzero(e)),
                max_error_feature=str(selected[take][col]),max_error_row=meta[row]))
        if variant!='spatial_fast' and np.max(np.abs(error)/scale[mask])>1e-5:
            write_json(tech/'failed-numerical.json',dict(comparisons=comparisons,implementation=implementation))
            raise ValueError(f'{variant}: target drift exceeds 1e-5 of training SD; see numerical audit')
    parallel={v:[] for v in variants}
    startup=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(str(sample),args.reference,variants,selection)) as pool:
        # Forces initialization/compilation before the steady-state timing.
        list(pool.map(block,[(variants[0],0,1)]*args.workers))
        startup=time.perf_counter()-startup
        for repeat in range(args.repeats):
            for variant in np.random.default_rng(repeat).permutation(variants):
                jobs=[(variant,i,min(i+32,len(x))) for i in range(0,len(x),32)]
                start=time.perf_counter();result=np.concatenate(list(pool.map(block,jobs)))
                parallel[variant].append(time.perf_counter()-start)
                print(json.dumps(dict(stage='parallel',repeat=repeat,variant=str(variant),seconds=parallel[variant][-1])),flush=True)
                if not np.array_equal(result,values[variant]):raise ValueError('Parallel descriptor output changed')
    if descriptor_implementation()!=implementation:raise ValueError('Producer changed while benchmarking')
    cache_difference=(values['reference'].astype(np.float64)-expected)/scale
    at=np.unravel_index(np.abs(cache_difference).argmax(),cache_difference.shape)
    import gudhi,scipy,numba
    summary=dict(rows=len(x),sources=len(seen),materials=sorted({r['material'] for r in meta}),
        hostname=platform.node(),affinity=sorted(os.sched_getaffinity(0)),workers=args.workers,
        warmup_seconds=warm,worker_startup_seconds=startup,single_worker_seconds=times,
        parallel_seconds=parallel,numerical=comparisons,implementation=implementation,
        reference_sha256=sha(Path(args.reference)),benchmark_sha256=sha(Path(__file__)),
        reference_vs_cached=dict(max_standardized=float(np.abs(cache_difference).max()),
            rms_standardized=float(np.sqrt(np.mean(cache_difference**2))),
            max_error_row=meta[at[0]],max_error_feature=names[at[1]],
            note='Original producer on this CPU versus sealed targets; measured separately from optimization drift.'),
        cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        versions=dict(numpy=np.__version__,scipy=scipy.__version__,gudhi=gudhi.__version__,numba=numba.__version__),
        note='Steady-state compute timings exclude input sampling, warmup and publication; no online run.')
    if any(v.startswith('native') for v in variants):
        from src.postprocessing.descriptors.alpha3d import extension
        summary['native_build']=json.loads(Path(extension().build_receipt).read_text())
    if selection is not None:
        summary['selection']=dict(path=args.selection,sha256=sha(Path(args.selection)),columns=selection)
    write_json(tech/'results.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packed',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--rows-per-source',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS[:-1]))
    p.add_argument('--selection')
    args=p.parse_args()
    if min(args.workers,args.rows_per_source,args.repeats)<1:p.error('Counts must be positive')
    if 'reference' not in args.variants or len(set(args.variants))!=len(args.variants):p.error('Require one reference and unique variants')
    if 'native_selected' in args.variants and not args.selection:p.error('native_selected requires --selection')
    run(args)


if __name__=='__main__':
    # Keep worker functions importable when routed through the unified CLI.
    from src.postprocessing.target_benchmark import main as entry
    entry()
