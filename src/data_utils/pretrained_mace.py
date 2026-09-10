"""80-atom TDA targets and one shuffled pass through the stored neighborhoods."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from src.analysis.liquid_structure import persistence_image
from src.data_utils.temporal_campaign import write_json


class Quadruplets:
    def __init__(self,cfg):
        self.records=json.loads((Path(cfg['cache'])/'manifest.json').read_text())['shards']
        points=cfg['points']
        self.clouds=[np.load(Path(r['directory'])/'clouds.npy',mmap_mode='r')[:,:,:points] for r in self.records]
        self.tda=[np.load(Path(r['directory'])/'tda.npy',mmap_mode='r') for r in self.records]
        self.conditions=[np.load(Path(r['directory'])/'condition.npy') for r in self.records]
        self.pools={split:[np.array([(i,j) for i,r in enumerate(self.records) if r['split']==split and r['material']==m for j in range(r['anchors_count'])],dtype=np.int64) for m in range(3)] for split in ('train','val')}
        self.required_temporal_lag=cfg['temporal_lag_ps']
        self.temporal_eligible=np.array([abs(r['lags'][0]*r['cadence_ps']-self.required_temporal_lag)<1e-9 for r in self.records])


    def temporal_mask(self,indices):
        return self.temporal_eligible[indices[:,0]]

    def get(self,indices):
        x=np.stack([self.clouds[i][j] for i,j in indices]).astype(np.float32);t=np.stack([self.tda[i][j] for i,j in indices]);c=np.stack([self.conditions[i][j] for i,j in indices])
        m=np.array([self.records[i]['material'] for i,j in indices],dtype=np.int64)
        return x,t,c,m

    def all_indices(self, split):
        return np.concatenate(self.pools[split])

    def epoch_steps(self, batch_size):
        return (sum(len(p) for p in self.pools['train'])+batch_size-1)//batch_size

    def epoch(self, split, batch_size, rng):
        rows=self.all_indices(split)
        rows=rows[rng.permutation(len(rows))]
        for start in range(0,len(rows),batch_size):
            yield rows[start:start+batch_size]


def prepare_target_shard(record, cache):
    source=Path(record['directory']);directory=Path(cache)/record['name']
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['tda_points']!=80 or saved['source_directory']!=str(source):
            raise ValueError(f'TDA cache protocol mismatch: {manifest}')
        digest=hashlib.sha256((directory/'tda.npy').read_bytes()).hexdigest()
        if digest!=saved['tda_sha256']:raise ValueError(f'TDA checksum mismatch: {directory}')
        return saved
    started=time.monotonic()
    clouds=np.load(source/'clouds.npy',mmap_mode='r')[:,:,:80]
    if clouds.shape[2:]!=(80,3):raise ValueError(f'Expected 80x3 neighborhoods in {source}: {clouds.shape}')
    targets=np.lib.format.open_memmap(directory/'tda.tmp.npy',mode='w+',dtype='float32',shape=(len(clouds),4,144))
    for i in range(len(clouds)):
        for view in range(4):targets[i,view]=persistence_image(clouds[i,view].astype(np.float32))
    targets.flush();del targets
    (directory/'tda.tmp.npy').replace(directory/'tda.npy')
    for name in ('clouds.npy','condition.npy','ids.npy','frames.npy'):
        destination=directory/name
        if not destination.exists():destination.symlink_to((source/name).resolve())
    saved=dict(record,directory=str(directory),source_directory=str(source),tda_points=80,
               tda_sha256=hashlib.sha256((directory/'tda.npy').read_bytes()).hexdigest(),
               tda_normalization='number of supplied atoms minus one (79)',seconds=time.monotonic()-started)
    write_json(manifest,saved)
    return saved


def prepare(cfg):
    cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True)
    source=Path(cfg['source_manifest']);records=json.loads(source.read_text())['shards']
    completed=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        futures=[pool.submit(prepare_target_shard,r,str(cache)) for r in records]
        for future in as_completed(futures):
            completed.append(future.result())
            status=dict(state='preparing_80_atom_tda',completed_shards=len(completed),total_shards=len(records),elapsed_seconds=time.monotonic()-started)
            write_json(cache/'status.json',status);print(json.dumps(status),flush=True)
    # Preserve source order: indices, targets and sampling share this ordering.
    by_name={r['name']:r for r in completed};ordered=[by_name[r['name']] for r in records]
    write_json(cache/'manifest.json',dict(shards=ordered,config=cfg,tda_points=80,
        source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        protocol='Same stored float16 coordinates as encoder, first 80 atoms; full alpha complex and H0/H1/H2 images; no 65-atom targets reused.'))
    write_json(cache/'status.json',dict(state='complete',shards=len(ordered),target_views=sum(4*r['anchors_count'] for r in ordered),seconds=time.monotonic()-started))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    prepare(json.loads(Path(args.config).read_text()))
