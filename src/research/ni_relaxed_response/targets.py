"""Paired quenched teacher values; no straight-through relaxation derivatives."""
import time
import torch
from src.research.ni_response.model import Motion
from .data import read,root,cache,load,save,sha,write_json,atomic_save
from .response import folder

def prepare_lane(c,p,lane,micro,deadline):
    torch.manual_seed(c['seed']);model=Motion(c).cuda().eval().requires_grad_(False)
    data=load(c,'data');out=cache(c)/'target-packets';out.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(data['state']),128)[lane::2]:
        if time.time()>deadline-300:raise TimeoutError('Relaxed code preparation continuation')
        path=out/f'future-{start:06d}.pt'
        if not path.exists():
            q=data['future'][start:start+128];shape=q.shape[:2]
            with torch.no_grad():
                y=torch.cat([model.geometry(x.cuda()).cpu() for x in q.flatten(0,1).split(micro)]).reshape(*shape,128)
                z=torch.cat([model.base_encode(x.cuda(),create_graph=False).cpu() for x in data['state'][start:start+128].split(micro)])
            atomic_save(path,dict(target=y,z=z))
        write_json(out/f'progress-{lane}.json',dict(stage='future_codes',last_row=start))
    for i in range(len(p['response_jobs']))[lane::2]:
        if time.time()>deadline-300:raise TimeoutError('Relaxed response target continuation')
        path=out/f'response-{i:04d}.pt'
        if not path.exists():
            f=folder(c,p['response_jobs'][i]);done=read(f/'complete.json')
            if sha(f/'response.pt')!=done['sha256']:raise ValueError('Changed finite response packet')
            b=torch.load(f/'response.pt',weights_only=False,map_location='cpu')
            with torch.no_grad():
                target=model.geometry(b['future'].flatten(0,1).cuda()).reshape(c['teacher']['shots'],5,128).mean(0).cpu()
                z=model.base_encode(b['state'].cuda(),create_graph=False).cpu()
            eps=c['student_difference']['response_epsilon']
            response=torch.stack([(target[2]-target[1])/(2*eps),(target[4]-target[3])/(2*eps)],-1)[None]
            record=dict(state=b['state'][:1],input_pairs=torch.stack([b['state'][1:3],b['state'][3:5]])[:,:,None],z=z[:1],z_pairs=torch.stack([z[1:3],z[3:5]])[:,:,None],target=target[:1],response=response,source=b['source'],ancestor=b['ancestor'],role=b['role'],job=b['job'])
            if not all(torch.isfinite(record[k]).all() for k in ('state','input_pairs','z','z_pairs','target','response')):raise FloatingPointError('Nonfinite quenched targets')
            atomic_save(path,record)
        write_json(out/f'progress-{lane}.json',dict(stage='response_targets',last_job=i))
    write_json(out/f'lane-{lane}.json',dict(identity=p['identity']))

def seal(c,p):
    if (cache(c)/'targets.json').exists():load(c,'targets');return
    data=load(c,'data');out=cache(c)/'target-packets';future=[];codes=[]
    for start in range(0,len(data['state']),128):
        a=torch.load(out/f'future-{start:06d}.pt',weights_only=False,map_location='cpu');future.append(a['target']);codes.append(a['z'])
    short=[torch.load(out/f'response-{i:04d}.pt',weights_only=False,map_location='cpu') for i in range(len(p['response_jobs']))]
    save(c,'targets',dict(future=torch.cat(future),codes=torch.cat(codes),short=short,checkpoint_sha256=c['checkpoint_sha256'],data_sha256=sha(cache(c)/'data.pt')))


def short_objective(model,record,c,arm,seed,*,create_graph=True):
    def predict(z):return model.path(z,[.1],particles=4,seed=seed)[...,:128].mean(1)[:,0]
    with torch.set_grad_enabled(create_graph),torch.autocast('cuda',enabled=False):
        if arm['frozen']:prediction=predict(model.adapt(record['z'].cuda()))
        else:prediction=predict(model.encode(record['state'].cuda(),create_graph=create_graph))
        value=.5*(prediction-record['target'].cuda()).square().mean();derivative=value.new_zeros(())
        if arm['derivative']:
            for k in range(2):
                if arm['frozen']:minus,plus=[predict(model.adapt(record['z_pairs'][k,j].cuda())) for j in range(2)]
                else:minus,plus=[predict(model.encode(record['input_pairs'][k,j].cuda(),create_graph=create_graph)) for j in range(2)]
                dp=(plus-minus)/(2*c['student_difference']['response_epsilon']);dt=record['response'][...,k].cuda()
                derivative+=.25*(dp-dt).square().mean()/dt.square().mean().clamp_min(1e-6)
        return value,derivative
