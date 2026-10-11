"""Geometry-only MACE with a single outer taper and live per-observation normalization."""
import math
from contextlib import contextmanager
import torch
from torch import nn
from src.models.encoders.spatial_mace import SpatialMACE
from src.research.continuous_motion.model import MotionModel


class Geometry(SpatialMACE):
    def __init__(self, c):
        super().__init__(d0=1., n_ref=80., radius=8., **c['encoder'])
        width=2*self.channels;dim=c['encoder']['code_dim']
        self.pool_norm=nn.LayerNorm(width)
        self.projection=nn.Linear(width,dim,bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.residual=nn.Sequential(nn.Linear(width,dim),nn.SiLU(),nn.Linear(dim,dim))
        nn.init.normal_(self.residual[-1].weight,std=.01);nn.init.zeros_(self.residual[-1].bias)

    def graph(self,q,keep=None):
        # Geometry and cutoff calculations retain float32 under neural AMP.
        with torch.autocast('cuda',enabled=False):
            q=q.float()-q[:,:1].float();batch,atoms,_=q.shape
            if atoms!=80:raise ValueError('Require nearest80 neighborhoods')
            with torch.no_grad():
                valid=q.norm(dim=-1)<self.radius
                if keep is not None:valid=valid & keep
                allowed=((q[:,None]-q[:,:,None]).norm(dim=-1)<self.cutoff)&valid[:,:,None]&valid[:,None,:]
                allowed &= ~torch.eye(atoms,dtype=torch.bool,device=q.device)[None]
                b,i,j=allowed.nonzero(as_tuple=True)
            x=q.flatten(0,1);edge=torch.stack((b*atoms+i,b*atoms+j));vectors=x[edge[1]]-x[edge[0]]
            attrs=x.new_ones(len(x),1);center=torch.zeros_like(attrs);center[torch.arange(batch,device=x.device)*atoms]=1
            radial,cutoff=self.radial_embedding(vectors.norm(dim=-1,keepdim=True),attrs,edge,self.atomic_numbers)
            if cutoff is not None:raise ValueError('Expected radial-embedded cutoff')
            # Flat through6A, one cosine taper over6–8A, applied ONLY at pooling.
            distance=(q.square().sum(-1)+torch.nn.functional.one_hot(torch.zeros(batch,dtype=torch.long,device=q.device),80)).sqrt()
            weight=.5*(1+torch.cos(math.pi*((distance-6.)/2.).clamp(0,1)))
            weight=weight*valid
            return dict(attrs=attrs,center=center,weight=weight.flatten(),edge=edge,
                radial=radial,angular=self.spherical_harmonics(vectors),
                group=torch.arange(batch,device=x.device).repeat_interleave(80),
                centers=torch.arange(batch,device=x.device)*80,size=batch)

    def atom_features(self,g):
        h=self.node_embedding(g['attrs'])+self.center_embedding(g['center'])
        for k,(interaction,product) in enumerate(zip(self.interactions,self.products,strict=True)):
            message,skip=interaction(node_attrs=g['attrs'],node_feats=h,edge_attrs=g['angular'],
                edge_feats=g['radial'],edge_index=g['edge'],cutoff=None,first_layer=k==0)
            h=product(message,sc=skip,node_attrs=g['attrs'])
        return h

    def forward(self,q,keep=None):
        g=self.graph(q,keep);h=self.atom_features(g)
        with torch.autocast('cuda',enabled=False):
            scalar=h[:,:self.channels].float();w=g['weight'].float()
            pooled=scalar.new_zeros(g['size'],self.channels).index_add(0,g['group'],scalar*w[:,None])
            mass=w.new_zeros(g['size']).index_add(0,g['group'],w)
            u=self.pool_norm(torch.cat((scalar[g['centers']],pooled/mass[:,None].clamp_min(1e-6)),-1))
            return self.projection(u)+self.residual(u)


class Spatial(nn.Module):
    def __init__(self,c):
        from .data import spatial_target_names
        super().__init__();self.geometry=Geometry(c)
        self.descriptors=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,len(spatial_target_names(c))))
        self.masked_predictor=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128))


class Motion(MotionModel):
    def __init__(self,c):
        super().__init__(c,'continuous');self.geometry=Geometry(c)

    def encode(self,state,*,create_graph):
        if state.shape[1:]!=(80,6):raise ValueError(f'Expected local80 Q,V: {state.shape}')
        state=state.float()
        s,m=geometry_jvp(self.geometry,state[...,:3],state[...,3:]*self.time_scale,create_graph=create_graph)
        return torch.cat((s,m),-1)


def geometry_jvp(geometry,q,direction,*,create_graph):
    """Exact reverse-over-reverse JVP with bounded graph microbatches.

    The local forward-AD candidate disagreed after checkpoint switching and is
    not used for scientific targets or fitting. No finite differences are used.
    """
    with torch.enable_grad(),torch.autocast('cuda',enabled=False),reference_contractions(geometry):
        return torch.autograd.functional.jvp(geometry,q.float(),direction.float(),create_graph=create_graph,strict=True)


def fixed_geometry(c,source):
    # cuEq's generic pickle/deepcopy reducer loses the fused convolution wrapper.
    # Reconstruct the declared architecture, then transfer its exact state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        target=Geometry(c).cuda()
    target.load_state_dict(source.state_dict())
    return target.eval().requires_grad_(False)


@contextmanager
def reference_contractions(geometry):
    """Same polynomial and parameters, ordinary torch ops for nested derivatives.

    Avoid cuEq's combinatorial JIT generation for third input/parameter gradients.
    Only parameter-free contraction implementations change, temporarily.
    """
    from cuequivariance_torch.primitives.segmented_polynomial import SegmentedPolynomial
    from cuequivariance_torch.primitives.segmented_polynomial_naive import SegmentedPolynomialNaive
    saved=[];device=next(geometry.parameters()).device
    for module in list(geometry.modules()):
        if not isinstance(module,SegmentedPolynomial) or isinstance(module.m,SegmentedPolynomialNaive):continue
        if not hasattr(module,'_response_reference'):
            implementation=SegmentedPolynomialNaive(module._polynomial_orig,module._math_dtype_orig,module._output_dtype_map_orig).to(device)
            object.__setattr__(module,'_response_reference',implementation)
        saved.append((module,module.m));module.m=module._response_reference
    try:yield
    finally:
        for module,implementation in saved:module.m=implementation


def variance_covariance(z):
    z=z.float();centered=z-z.mean(0);cov=centered.T@centered/(len(z)-1)
    std=(cov.diag()+1e-4).sqrt();variance=torch.relu(1-std).mean()
    off=cov-torch.diag_embed(cov.diag())
    # Per-channel normalization: not the old D^2 mean with another .01 factor.
    covariance=off.square().sum()/z.shape[1]
    return variance,covariance


def spatial_loss(model,z,target,weights,method,teacher=None):
    a,b=z[:2];za,zb=a.float(),b.float()
    invariance=(za-zb).square().mean();va,ca=variance_covariance(za);vb,cb=variance_covariance(zb)
    variance=(va+vb)/2;covariance=(ca+cb)/2
    with torch.autocast('cuda',enabled=False):
        pa=model.descriptors(za);pb=model.descriptors(zb)
        prediction=.25*(((pa-target).square()+(pb-target).square())*weights).sum(-1).mean()
        masked=prediction.new_zeros(())
        if method=='masked_teacher':
            pred=model.masked_predictor(z[2].float())
            masked=.5*(nn.functional.layer_norm(pred,(128,))-nn.functional.layer_norm(teacher.detach().float(),(128,))).square().mean()
        total=25*invariance+25*variance+covariance+prediction+masked
    return total,dict(invariance=invariance,variance=variance,covariance=covariance,descriptor_half_mse=prediction,masked_half_mse=masked)


def views(q,method,seed):
    rng=torch.Generator(device=q.device).manual_seed(seed)
    values=[]
    for _ in range(2):
        jitter=torch.randn(q.shape,generator=rng,device=q.device)*.01
        x=q+jitter;x=x-x[:,:1]
        values.append((x,None))
    if method=='masked_teacher':
        keep=torch.rand(q.shape[:2],generator=rng,device=q.device)>.2
        keep[:,:13]=True
        values.append((values[0][0],keep))
    return values


def spatial_backward(model,ema,q,target,weights,method,seed,micro):
    """Exact full-batch VCReg cotangents; encoder graphs replay one microbatch at a time."""
    inputs=views(q,method,seed);latent=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for x,keep in inputs:
            latent.append(torch.cat([model.geometry(x[i:i+micro],None if keep is None else keep[i:i+micro]) for i in range(0,len(x),micro)]))
        teacher=None if method!='masked_teacher' else torch.cat([ema(x) for x in q.split(micro)])
    leaves=[z.detach().requires_grad_() for z in latent]
    loss,stats=spatial_loss(model,leaves,target,weights,method,teacher)
    loss.backward()
    for (x,keep),leaf in zip(inputs,leaves,strict=True):
        for i in range(0,len(x),micro):
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model.geometry(x[i:i+micro],None if keep is None else keep[i:i+micro])
            z.backward(leaf.grad[i:i+micro])
    return float(loss.detach()),{k:float(v.detach()) for k,v in stats.items()}
