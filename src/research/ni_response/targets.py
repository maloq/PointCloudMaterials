"""Fixed eSEN endpoint values and exact input JVP of analytic EAM tangents."""
import time
import torch
from .data import cache,load,save,read,sha,write_json,atomic_save
from .model import Motion
from .simulator import load as load_bank

def prepare(c,p,micro,deadline):
    if (cache(c)/'targets.json').exists():load(c,'targets');return
    torch.manual_seed(c['seed']);model=Motion(c).cuda().eval().requires_grad_(False)
    data=load(c,'data');bank=load_bank(c,p);folder=cache(c)/'target-packets';folder.mkdir(parents=True,exist_ok=True)
    futures=[];codes=[]
    for start in range(0,len(data['state']),128):
        if time.time()>deadline-300:raise TimeoutError('Teacher cache continuation required')
        path=folder/f'future-{start:06d}.pt'
        if not path.exists():
            q=data['future'][start:start+128];shape=q.shape[:2]
            with torch.no_grad():
                y=torch.cat([model.geometry(x.cuda()).cpu() for x in q.flatten(0,1).split(micro)]).reshape(*shape,128)
                z=torch.cat([model.base_encode(x.cuda(),create_graph=False).cpu() for x in data['state'][start:start+128].split(micro)])
            atomic_save(path,dict(target=y,z=z))
        v=torch.load(path,weights_only=False,map_location='cpu');futures.append(v['target']);codes.append(v['z'])
        write_json(folder/'progress.json',dict(stage='future_codes',done=min(start+128,len(data['state'])),total=len(data['state'])))
    records=[]
    for i,b in enumerate(bank):
        if time.time()>deadline-300:raise TimeoutError('Response target continuation required')
        path=folder/f'response-{i:04d}.pt'
        if not path.exists():
            q=(b['future'][:,0,:,:3]*c['length_factor']).float().cuda()
            tangent=(b['tangent'][:,0,:,:3,:]*c['length_factor']).float().cuda()
            derivatives=[]
            for k in range(2):
                with torch.enable_grad():
                    value,dv=torch.autograd.functional.jvp(model.geometry,q,tangent[...,k],create_graph=False,strict=True)
                derivatives.append(dv.mean(0).cpu())
            x=(b['state']*c['length_factor']).float()[None];basis=(b['basis']*c['length_factor']).float()[None]
            with torch.no_grad():
                z=model.base_encode(x.cuda(),create_graph=False).cpu()
                pairs=torch.stack([torch.stack([model.base_encode((x+sign*model.response_epsilon*basis[...,k]).cuda(),create_graph=False).cpu() for sign in (-1,1)]) for k in range(2)])
            record=dict(state=x,basis=basis,target=value.detach().mean(0,keepdim=True).cpu(),response=torch.stack(derivatives,-1)[None],z=z,z_pairs=pairs,source=b['source'],ancestor=b['ancestor'],role=b['role'],job=b['job'])
            if not all(torch.isfinite(record[k]).all() for k in ('state','basis','target','response','z','z_pairs')):raise FloatingPointError('Nonfinite Ni target')
            atomic_save(path,record)
        records.append(torch.load(path,weights_only=False,map_location='cpu'))
        write_json(folder/'progress.json',dict(stage='response_targets',done=i+1,total=len(bank)))
    save(c,'targets',dict(future=torch.cat(futures),codes=torch.cat(codes),short=records,checkpoint_sha256=c['checkpoint_sha256'],data_sha256=sha(cache(c)/'data.pt')))
