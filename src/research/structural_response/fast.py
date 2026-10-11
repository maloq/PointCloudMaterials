"""Fast student secants, cached-backbone adaptation and batched latent training.

MEAM teacher tangents remain analytic. Only the neural student's directional
derivatives use declared centered differences, evaluated in float32.
"""
import time
from pathlib import Path
import torch
from torch import nn
from .model import Motion
from .data import read,write_json,resolve_path,root,cache,digest,sha
from src.research.continuous_motion.model import mixture_nll
from src.research.continuous_motion.data import atomic_save
from src.research.equivariant_context.cache import RetainedCache


class FastMotion(Motion):
    def __init__(self,c):
        super().__init__(c)
        self.velocity_epsilon=c['student_difference']['velocity_epsilon']
        self.response_epsilon=c['student_difference']['response_epsilon']
        self.adapter=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.use_adapter=False

    def set_arm(self,arm):
        self.use_adapter=arm['frozen']
        self.geometry.requires_grad_(not self.use_adapter)
        self.adapter.requires_grad_(self.use_adapter)

    def base_encode(self,state,*,create_graph):
        if state.shape[1:]!=(80,6):raise ValueError('Expected local80 Q,V')
        with torch.set_grad_enabled(create_graph),torch.autocast('cuda',enabled=False):
            state=state.float();q=state[...,:3];direction=state[...,3:]*self.time_scale
            eps=self.velocity_epsilon
            s=self.geometry(q)
            m=(self.geometry(q+eps*direction)-self.geometry(q-eps*direction))/(2*eps)
            return torch.cat((s,m),-1)

    def adapt(self,z):
        if not self.use_adapter:return z
        # Analytic JVP of the small residual SiLU adapter, retaining parameter
        # gradients without requesting higher-order MACE autograd graphs.
        with torch.autocast('cuda',enabled=False):
            s,m=z.float().split(128,-1);a=self.adapter[0](s);sigmoid=a.sigmoid()
            out=s+self.adapter[2](torch.nn.functional.silu(a))
            dm=torch.nn.functional.linear(m,self.adapter[0].weight)
            dm=dm*sigmoid*(1+a*(1-sigmoid))
            velocity=m+torch.nn.functional.linear(dm,self.adapter[2].weight)
            return torch.cat((out,velocity),-1)

    def encode(self,state,*,create_graph):
        return self.adapt(self.base_encode(state,create_graph=create_graph and not self.use_adapter))


def short_objective(model,record,c,arm,seed,*,create_graph=True):
    def predict(z):return model.path(z,[.1],particles=4,seed=seed)[...,:128].mean(1)[:,0]
    with torch.set_grad_enabled(create_graph),torch.autocast('cuda',enabled=False):
        if arm['frozen']:
            prediction=predict(model.adapt(record['z'].cuda()))
        else:prediction=predict(model.encode(record['state'].cuda(),create_graph=create_graph))
        value=.5*(prediction-record['target'].cuda()).square().mean();derivative=value.new_zeros(())
        if arm['derivative']:
            eps=model.response_epsilon
            for k in range(2):
                if arm['frozen']:
                    minus,plus=[predict(model.adapt(record['z_pairs'][k,j].cuda())) for j in range(2)]
                else:
                    x=record['state'].cuda();direction=record['basis'][...,k].cuda()
                    minus=predict(model.encode(x-eps*direction,create_graph=create_graph))
                    plus=predict(model.encode(x+eps*direction,create_graph=create_graph))
                dp=(plus-minus)/(2*eps);dt=record['response'][...,k].cuda()
                derivative=derivative+.25*(dp-dt).square().mean()/dt.square().mean().clamp_min(1e-6)
    return value,derivative


def long_backward(model,data,targets,zall,indices,c,micro,seed):
    if zall is None:
        z=torch.cat([model.encode(data['state'][ids].cuda(),create_graph=False)
                     for ids in indices.split(micro)])
        leaf=z.detach().requires_grad_()
    else:leaf=model.adapt(zall[indices].cuda())
    with torch.autocast('cuda',dtype=torch.bfloat16):
        path=model.path(leaf,c['horizons_ps'],particles=4,seed=seed)
        mu,lv=model.prediction(path)
    y=(data['target'][indices].float().cuda()-data['mean'].float().cuda())/data['scale'].float().cuda()
    nll=mixture_nll(mu.float(),lv.float(),y).mean()
    latent=.5*(path[...,:128].mean(1)-targets[indices].cuda()).square().mean()
    loss=nll+c['response_training']['long_structural_weight']*latent
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fast long objective')
    loss.backward()
    if zall is None:
        cotangent=leaf.grad.detach()
        for offset in range(0,len(indices),micro):
            z=model.encode(data['state'][indices[offset:offset+micro]].cuda(),create_graph=True)
            z.backward(cotangent[offset:offset+micro])
    return float(nll.detach()),float(latent.detach())


def cached_targets(c,method,model,anchor,bank,future,state,frozen,micro,deadline,progress):
    """Reuse exact v2 teachers; only student codes use the new secant protocol."""
    old=resolve_path(c['fixed_teacher_source'])
    receipt=read(old/'analyses'/(method+'-derivative_joint')/'technical/fixed-target-cache.json')
    metadata=receipt['metadata'];pool=RetainedCache(resolve_path(c['feature_cache']),6)
    if metadata['spatial_sha256']!=c['spatial_initialization']['checkpoints'][method] or metadata['response_identity']!=c['response_identity'] or metadata['data_sha256']!=sha(cache(c)/'complete.json'):
        raise ValueError('Exact teacher cache belongs to another checkpoint or physical bank')
    with pool.lease(digest(metadata),deadline=deadline,metadata=metadata,shared=True) as folder:
        if str(folder)!=receipt['path']:raise ValueError('Unexpected exact teacher path')
        values=[]
        for name in ('future','short'):
            if sha(folder/(name+'.pt'))!=read(folder/(name+'.json'))['sha256']:raise ValueError('Changed exact teacher target: '+name)
            values.append(torch.load(folder/(name+'.pt'),weights_only=False,map_location='cpu'))
        target,short=values
    if [r['job'] for r in short]!=[r['job'] for r in bank]:raise ValueError('Exact teacher bank order changed')
    short=[dict(r,source=b['source']) for r,b in zip(short,bank,strict=True)]
    write_json(progress/'fixed-target-cache.json',dict(receipt,reused_from=str(old),student_derivatives=c['neural_derivative']))
    if not frozen:return target,None,short
    metadata=dict(kind='fast_student_base_codes',spatial_sha256=c['spatial_initialization']['checkpoints'][method],
        response_identity=c['response_identity'],data_sha256=sha(cache(c)/'complete.json'),
        difference=c['student_difference'],implementation=sha(Path(__file__)))
    def stage(name,done,total):write_json(progress/'progress.json',dict(state='preparing_student_cache',stage=name,completed=done,total=total,updated_unix=time.time()))
    stage('waiting_for_base_codes',0,len(state))
    with pool.lease(digest(metadata),deadline=deadline,metadata=metadata) as folder:
        path=folder/'codes.pt'
        if not path.exists():
            codes=[]
            for start in range(0,len(state),256):
                if time.time()>deadline-300:raise TimeoutError('Student code preparation deadline')
                packet=folder/f'current-{start:06d}.pt'
                if not packet.exists():atomic_save(packet,torch.cat([model.base_encode(x.cuda(),create_graph=False).cpu() for x in state[start:start+256].split(micro)]))
                codes.append(torch.load(packet,weights_only=False,map_location='cpu'));stage('base_codes',min(start+256,len(state)),len(state))
            atomic_save(path,torch.cat(codes));write_json(folder/'codes.json',dict(sha256=sha(path)))
        if sha(path)!=read(folder/'codes.json')['sha256']:raise ValueError('Changed student code cache')
        codes=torch.load(path,weights_only=False,map_location='cpu');records=[]
        for i,r in enumerate(short):
            packet=folder/f'response-{i:04d}.pt'
            if not packet.exists():
                if time.time()>deadline-300:raise TimeoutError('Student response preparation deadline')
                x=r['state'].cuda();direction=r['basis'].cuda();eps=model.response_epsilon
                z=model.base_encode(x,create_graph=False).cpu()
                pairs=torch.stack([torch.stack([model.base_encode(x+sign*eps*direction[...,k],create_graph=False).cpu() for sign in (-1,1)]) for k in range(2)])
                atomic_save(packet,dict(z=z,z_pairs=pairs))
            records.append(dict(r,**torch.load(packet,weights_only=False,map_location='cpu')));stage('response_codes',i+1,len(short))
        write_json(progress/'student-code-cache.json',dict(path=str(folder),metadata=metadata))
    return target,codes,records
