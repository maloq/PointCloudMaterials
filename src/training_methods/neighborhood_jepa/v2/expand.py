"""Source-balanced expansion from the registered native-Al full-TDA release."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash,digest,save_json
from src.training_methods.neighborhood_jepa.prepare import prepare_task
from .data import prepare


def build(config):
    parent=resolve_path(config['parent_release'])
    root=resolve_path(config['parent_cache'])
    root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((parent/'manifest.json').read_text())
    sources={s['id']:s for s in manifest['sources'] if s['stratum']=='al_native'}
    train=sorted(s for s in sources if sources[s]['split']=='train')
    rng=np.random.default_rng(config['seed'])
    count,remainder=divmod(config['anchors'],len(train))
    tasks=[]
    for si,sid in enumerate(train):
        available=[r for r in manifest['shards'] if r['source']==sid and r['task']['split']=='train']
        rng.shuffle(available)
        remaining=count+int(si<remainder)
        for shard in available:
            n=min(config['anchors_per_shard'],remaining,shard['anchors'])
            tasks.append(dict(shard=shard,rows=rng.choice(shard['anchors'],n,replace=False).tolist(),
                source=sources[sid],group=0,seed=int(rng.integers(2**31))))
            remaining-=n
            if not remaining:break
        if remaining:raise ValueError(f'Not enough existing anchors for {sid}: {remaining}')
    for shard in manifest['shards']:
        if shard['source'] in sources and shard['task']['split']=='selection':
            tasks.append(dict(shard=shard,rows=list(range(shard['anchors'])),source=sources[shard['source']],
                group=0,seed=int(rng.integers(2**31))))
    producer=Path(__file__).parents[1]/'prepare.py'
    identity=digest(dict(config=config,parent=file_hash(parent/'manifest.json'),
                         planner=file_hash(Path(__file__)),graph_producer=file_hash(producer)))
    plan=dict(identity=identity,config=config,parent_manifest_sha256=file_hash(parent/'manifest.json'),
              groups=[['Al','al-lee2003-meam']],tasks=tasks)
    plan_path=root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Expansion plan changed; use a new cache')
    save_json(plan_path,plan)
    receipts=[]
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(prepare_task,(str(root),str(parent),identity,t,4.25)) for t in tasks]
        for future in as_completed(futures):
            receipts.append(future.result())
            save_json(root/'status.json',dict(state='preparing',complete=len(receipts),total=len(tasks)))
    receipts.sort(key=lambda r:r['id'])
    save_json(root/'manifest.json',dict(state='complete',identity=identity,config=config,
        groups=plan['groups'],shards=receipts))
    torch.set_num_threads(1)
    result=prepare(config)
    save_json(root/'status.json',dict(state='complete',anchors=config['anchors'],sources=len(train),
        corrected_cache=str(resolve_path(config['cache'])),corrected_identity=result['identity']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    build(json.loads(Path(args.config).read_text()))
