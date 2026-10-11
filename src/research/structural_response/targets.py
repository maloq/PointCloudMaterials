"""Resumable, shared fixed-teacher values and frozen motion codes."""
import time
from pathlib import Path
import torch
from src.research.equivariant_context.cache import RetainedCache
from src.research.continuous_motion.data import atomic_save
from .data import read,root,cache,resolve_path,digest,sha,write_json
from .model import geometry_jvp,reference_contractions


def cached_targets(c,method,model,anchor,bank,future,state,frozen,micro,deadline,progress):
    metadata=dict(spatial_sha256=c['spatial_initialization']['checkpoints'][method],
        data_sha256=sha(cache(c)/'complete.json'),response_identity=c['response_identity'],
        time_scale_ps=c['model']['time_scale_ps'],precision='float32',
        implementation={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('model.py'))})
    pool=RetainedCache(resolve_path(c['feature_cache']),6)

    def stage(name,completed,total):
        write_json(progress/'progress.json',dict(state='preparing_fixed_teacher',stage=name,completed=completed,total=total,updated_unix=time.time()))

    def piece(folder,name,producer):
        path=folder/(name+'.pt')
        if not path.exists():
            if time.time()>deadline-300:raise TimeoutError('Fixed-teacher preparation checkpointed for continuation')
            value=producer();atomic_save(path,value)
        return torch.load(path,weights_only=False,map_location='cpu')

    def sealed(folder,name,producer):
        receipt=folder/(name+'.json');path=folder/(name+'.pt')
        if not receipt.exists():
            atomic_save(path,producer());write_json(receipt,dict(sha256=sha(path)))
        if sha(path)!=read(receipt)['sha256']:raise ValueError('Changed fixed-teacher cache: '+str(path))
        return torch.load(path,weights_only=False,map_location='cpu')

    meta=dict(metadata,kind='fixed_future_targets')
    stage('waiting_for_shared_targets',0,len(bank))
    with pool.lease(digest(meta),deadline=deadline,metadata=meta) as folder:
        def make_future():
            values=[];flat=future.flatten(0,1)
            for start in range(0,len(flat),256):
                def compute():
                    with torch.no_grad():return anchor(flat[start:start+256].cuda()).cpu()
                values.append(piece(folder,f'future-{start:06d}',compute))
                if start%2048==0:stage('future_embeddings',min(start+256,len(flat)),len(flat))
            return torch.cat(values).reshape(len(future),4,128)
        target=sealed(folder,'future',make_future)
        def make_short():
            values=[]
            for i,r in enumerate(bank):
                def compute():
                    slot=r['horizons_fs'].index(100);q=r['future'][:,slot,:,:3].float().cuda()
                    with torch.no_grad():endpoint=anchor(q).mean(0,keepdim=True).cpu()
                    responses=[]
                    for k in range(2):
                        _,j=geometry_jvp(anchor,q,r['tangent'][:,slot,:,:3,k].float().cuda(),create_graph=False)
                        responses.append(j.mean(0,keepdim=True).cpu())
                    return dict(state=r['state'][None].float(),basis=r['basis'][None].float(),
                        target=endpoint,response=torch.stack(responses,-1),role=r['role'],ancestor=r['ancestor'],job=r['job'])
                values.append(piece(folder,f'short-{i:04d}',compute));stage('simulator_tangent_targets',i+1,len(bank))
            return values
        short=sealed(folder,'short',make_short)
        write_json(progress/'fixed-target-cache.json',dict(path=str(folder),metadata=meta))

    if not frozen:return target,None,short
    meta=dict(metadata,kind='frozen_motion_codes')
    stage('waiting_for_shared_frozen_codes',0,len(state))
    with pool.lease(digest(meta),deadline=deadline,metadata=meta) as folder:
        def make_current():
            values=[]
            for start in range(0,len(state),128):
                def compute():
                    with torch.no_grad():return torch.cat([model.encode(x.cuda(),create_graph=False).cpu() for x in state[start:start+128].split(micro)])
                values.append(piece(folder,f'current-{start:06d}',compute));stage('current_motion_codes',min(start+128,len(state)),len(state))
            return torch.cat(values)
        zall=sealed(folder,'current',make_current)
        def make_directions():
            values=[]
            for i,r in enumerate(short):
                def compute():
                    x=r['state'].cuda();basis=r['basis'].cuda();dz=[]
                    z=model.encode(x,create_graph=False).detach().cpu()
                    def enc(x):return model.encode(x,create_graph=True)
                    with reference_contractions(model.geometry):
                        for k in range(2):
                            _,j=torch.autograd.functional.jvp(enc,x,basis[...,k],create_graph=False,strict=True);dz.append(j.detach().cpu())
                    return dict(z=z,dz=torch.stack(dz,-1))
                values.append(piece(folder,f'direction-{i:04d}',compute));stage('frozen_input_derivatives',i+1,len(short))
            return values
        directions=sealed(folder,'directions',make_directions)
        short=[dict(r,**d) for r,d in zip(short,directions,strict=True)]
        write_json(progress/'frozen-code-cache.json',dict(path=str(folder),metadata=meta))
    return target,zall,short
