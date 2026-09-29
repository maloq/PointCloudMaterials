"""Source-sharded, resumable descriptors from sealed observations, stored in IDS."""
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from .data import config,population,masks
from .descriptors import patch_descriptors,summarize_context


def parent(c):
    p=resolve_path(c['parent_config'])
    if sha(p)!=c['parent_config_sha256']:raise ValueError('Parent protocol changed')
    return config(p)


def _block(points):
    return np.stack([patch_descriptors(x)[0] for x in points])


def prepare_source(c,item,pool):
    pc=parent(c);dataset=resolve_path(pc['dataset']['root']);folder=dataset/'sources'/str(item['source'])
    dest=resolve_path(c['cache'])/'sources'/str(item['source']);dest.mkdir(parents=True,exist_ok=True)
    binding=dict(dataset=pc['prepared_data']['identity'],input=item['files'],
                 descriptors=sha(Path(__file__).with_name('descriptors.py')),preparation=sha(Path(__file__)))
    identity=digest(binding)
    if (dest/'complete.json').exists():
        receipt=config(dest/'complete.json')
        if receipt['identity']!=identity:raise ValueError('Changed descriptor source identity')
        for name,h in receipt['files'].items():
            if sha(dest/name)!=h:raise ValueError(f'Changed descriptor cache {dest/name}')
        return receipt
    for name in ('rows.npz','positions.npy'):
        if sha(folder/name)!=item['files'][name]:raise ValueError(f'Changed input {folder/name}')
    with np.load(folder/'rows.npz') as a:m={k:a[k] for k in a.files}
    ids=np.flatnonzero((m['kind']<2)&~m['inside_crystal']&~m['crystal_visible_context']&np.isfinite(m['crystal_distance']))
    patches=np.unique(m['indices'][ids]);bank=np.load(folder/'positions.npy',mmap_mode='r')
    # Empty sources still get a receipt; no model-specific removal of held-out rows.
    sample,names=patch_descriptors(np.asarray(bank[0]));width=len(sample)
    _,summaries=summarize_context(np.zeros((1,25,width)),m['actual'][:1])
    columns=[{'name':f'{s}/{n}','family':n.split('/')[0]} for s in summaries for n in names]
    progress=dest/'progress.json';start=0
    if progress.exists():
        saved=config(progress)
        if saved['identity']!=identity:raise ValueError('Changed partial descriptor identity')
        start=saved['patches_done']
    path=dest/'patches.npy'
    x=np.lib.format.open_memmap(path,mode='r+' if progress.exists() else 'w+',dtype=np.float32,shape=(len(patches),width))
    write_json(progress,dict(identity=identity,patches_done=start,total_patches=len(patches)))
    started=time.time()
    for begin in range(start,len(patches),8192):
        end=min(begin+8192,len(patches));chunk=c['preparation']['chunk_size']
        arrays=(np.asarray(bank[patches[i:min(i+chunk,end)]]) for i in range(begin,end,chunk))
        offset=begin
        for value in pool.map(_block,arrays):
            x[offset:offset+len(value)]=value;offset+=len(value)
        x.flush();write_json(progress,dict(identity=identity,patches_done=end,total_patches=len(patches)))
        print(json.dumps(dict(source=item['source'],patches_done=end,total=len(patches),seconds=time.time()-started)),flush=True)
    context=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(columns)))
    for begin in range(0,len(ids),128):
        rows=ids[begin:begin+128];local=np.searchsorted(patches,m['indices'][rows])
        context[begin:begin+len(rows)]=summarize_context(x[local],m['actual'][rows])[0]
    context.flush();del context,x
    np.save(dest/'local_rows.npy',ids);write_json(dest/'columns.json',columns)
    receipt=dict(identity=identity,source=item['source'],rows=len(ids),patches=len(patches),features=len(columns),binding=binding,
                 files={name:sha(dest/name) for name in ('features.npy','local_rows.npy','columns.json','patches.npy')})
    write_json(dest/'complete.json',receipt);return receipt


def prepare(c,task):
    pc=parent(c);manifest=config(resolve_path(pc['dataset']['root'])/'manifest.json')
    sources=sorted(manifest['sources'],key=lambda s:s['patches'],reverse=True)
    selected=sources[task::c['preparation']['tasks']]
    with ProcessPoolExecutor(max_workers=c['preparation']['workers']) as pool:
        for item in selected:prepare_source(c,item,pool)


def seal(c):
    pc=parent(c);_,manifest,_,meta,base=population(pc)
    arm={'population':'clear','source_fraction':1};split,weights,sources=masks(meta,base,arm,pc)
    ids=np.concatenate(list(split.values()));cache=resolve_path(c['cache']);receipts=[]
    global_row=np.full(len(base),-1,np.int64);global_row[ids]=np.arange(len(ids))
    columns=None;x=None;seen=np.zeros(len(ids),bool)
    for item in manifest['sources']:
        folder=cache/'sources'/str(item['source']);receipt=config(folder/'complete.json')
        for name,h in receipt['files'].items():
            if sha(folder/name)!=h:raise ValueError(f'Changed prepared {folder/name}')
        current=config(folder/'columns.json')
        if columns is None:
            columns=current;x=np.lib.format.open_memmap(cache/'features.building.npy',mode='w+',dtype=np.float32,shape=(len(ids),len(columns)))
        if current!=columns:raise ValueError('Descriptor names differ across sources')
        source_ids=np.flatnonzero(meta['source']==item['source']);local=np.load(folder/'local_rows.npy')
        dest=global_row[source_ids[local]]
        if (dest<0).any() or seen[dest].any():raise ValueError('Unmatched or duplicated descriptor rows')
        values=np.load(folder/'features.npy',mmap_mode='r')
        for start in range(0,len(dest),1024):
            v=values[start:start+1024]
            if not np.isfinite(v).all():raise FloatingPointError('Nonfinite prepared descriptors')
            x[dest[start:start+1024]]=v
        seen[dest]=True;receipts.append(receipt)
    if not seen.all():raise ValueError('Missing fixed evaluation rows')
    x.flush();del x;(cache/'features.building.npy').replace(cache/'features.npy')
    write_json(cache/'columns.json',columns)
    np.savez(cache/'rows.npz',ids=ids,source=meta['source'][ids],role=meta['role'][ids],kind=meta['kind'][ids],
             target=meta['crystal_distance'][ids],weights=np.concatenate(list(weights.values())))
    receipt=dict(dataset=manifest['identity'],parent_config_sha256=c['parent_config_sha256'],sources=receipts,
                 train_sources=sources.tolist(),rows={r:len(v) for r,v in split.items()},columns=len(columns),
                 files={n:sha(cache/n) for n in ('features.npy','columns.json','rows.npz')})
    receipt['identity']=digest(receipt);write_json(cache/'manifest.json',receipt)
    print(json.dumps(dict(rows=receipt['rows'],columns=len(columns),identity=receipt['identity'])),flush=True)


def load(c):
    cache=resolve_path(c['cache']);manifest=config(cache/'manifest.json')
    for name,h in manifest['files'].items():
        if sha(cache/name)!=h:raise ValueError(f'Changed sealed descriptor data {name}')
    with np.load(cache/'rows.npz') as a:rows={k:a[k] for k in a.files}
    return np.load(cache/'features.npy',mmap_mode='r'),rows,config(cache/'columns.json'),manifest
