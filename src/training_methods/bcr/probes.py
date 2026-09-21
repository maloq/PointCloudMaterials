"""Independent structural measurements and source-held-out frozen probes."""
import numpy as np
import torch
from torch import nn
from e3nn import o3
from .data import pack,taper,extract_patch


def descriptors(patches,radius):
    """Physical-A radial bins and l=0,2,4,6 multiscale moment Gram blocks.

    Eight Gaussian radial channels x each harmonic degree; all symmetric
    cross-radial contractions retained. Angular third-order q4/q6 contractions
    use Wigner-3j. No averaged-order claim beyond the observed patch.
    """
    radial=[];angular=[];rich=[];covariates=[]
    for patch in patches:
        x=torch.as_tensor(patch[1:],dtype=torch.float64);r=x.norm(dim=-1);w=taper(r,radius)
        bins=torch.exp(-((r[:,None]/radius-torch.linspace(0,1,12,dtype=r.dtype))/.08)**2)
        rstats=torch.stack((r.min(),(w*r).sum()/w.sum(),(w*r*r).sum()/w.sum(),w.sum(),w.sum()/(4/3*np.pi*radius**3)))
        radial.append(torch.cat(((w[:,None]*bins).sum(0),rstats)).numpy())
        ys={l:o3.spherical_harmonics(l,x,normalize=True,normalization='integral') for l in (0,2,4,6)}
        qs={l:(w[:,None]*y).sum(0)/w.sum() for l,y in ys.items()};order=[]
        for l in (4,6):
            q=qs[l];order.append((q.square().sum()*4*np.pi/(2*l+1)).sqrt())
            order.append(torch.einsum('ijk,i,j,k',o3.wigner_3j(l,l,l,dtype=q.dtype),q,q,q)/(q.square().sum().clamp_min(1e-12)**1.5))
        angular.append(torch.stack(order).numpy());rb=torch.exp(-((r[:,None]/radius-torch.linspace(0,1,8,dtype=r.dtype))/.13)**2)
        blocks=[]
        for l,y in ys.items():
            m=torch.einsum('nr,nm,n->rm',rb,y,w)/80. # fixed reference, keeps density
            gram=m@m.T;ii=torch.triu_indices(8,8);blocks.append(gram[ii[0],ii[1]])
        rich.append(torch.cat(blocks).numpy());covariates.append([len(patch),float(rstats[-1]),float(order[2])])
    return dict(radial=np.array(radial),angular=np.array(angular),rich=np.array(rich)),np.array(covariates)


def error_metrics(y,pred,records):
    out={};temp=np.array([r['temperature_K'] for r in records]);roots=np.array([r['root'] for r in records])
    groups={'all':np.ones(len(y),bool),**{f'T{v:g}':temp==v for v in np.unique(temp)}}
    for name,m in groups.items():
        mse=(pred[m]-y[m])**2;var=((y[m]-y[m].mean(0))**2).mean(0)
        out[name]=dict(standardized_rmse=float(np.sqrt(mse.mean())),per_target_rmse=np.sqrt(mse.mean(0)).tolist(),
            per_target_r2=np.divide(mse.mean(0),var,out=np.full_like(var,np.nan),where=var>1e-12).tolist())
        out[name]['per_target_r2']=[1-v if np.isfinite(v) else None for v in out[name]['per_target_r2']]
    out['per_root']={str(r):float(np.sqrt(((pred[roots==r]-y[roots==r])**2).mean())) for r in np.unique(roots)}
    return out


def frozen_probes(train_z,dev_z,test_z,train_y,dev_y,test_y,records,seed=42,updates=500):
    """Transforms and targets fit on train only; fixed ridge grid and MLP budget."""
    mx=train_z.mean(0);sx=train_z.std(0).clip(1e-6);my=train_y.mean(0);sy=train_y.std(0).clip(1e-6)
    x,d,t=[(v-mx)/sx for v in (train_z,dev_z,test_z)];y,dy,ty=[(v-my)/sy for v in (train_y,dev_y,test_y)]
    x,d,t=[np.c_[v,np.ones(len(v))] for v in (x,d,t)];best=None
    for alpha in np.logspace(-6,4,11):
        penalty=np.eye(x.shape[1])*alpha;penalty[-1,-1]=0
        weight=np.linalg.solve(x.T@x+penalty,x.T@y);score=np.square(d@weight-dy).mean()
        if best is None or score<best[0]:best=(score,alpha,weight)
    ridge=t@best[2]
    torch.manual_seed(seed);model=nn.Sequential(nn.Linear(x.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,y.shape[1]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    generator=torch.Generator().manual_seed(seed+1)
    for _ in range(updates):
        idx=torch.randint(len(x),(min(256,len(x)),),generator=generator);loss=(model(xt[idx])-yt[idx]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():mlp=model(torch.tensor(t,dtype=torch.float32)).numpy()
    return dict(ridge=error_metrics(ty,ridge,records),ridge_alpha=float(best[1]),mlp=error_metrics(ty,mlp,records))


def retrieval(z,reference,roots,k=20):
    roots=np.asarray(roots);valid=roots[:,None]!=roots[None,:]
    if np.any(valid.sum(1)<k):raise ValueError('Insufficient cross-root candidates for recall@20')
    a=np.linalg.norm(z[:,None]-z[None,:],axis=-1);b=np.linalg.norm(reference[:,None]-reference[None,:],axis=-1)
    a[~valid]=np.inf;b[~valid]=np.inf
    ai=np.argsort(a,axis=1)[:,:k];bi=np.argsort(b,axis=1)[:,:k]
    return np.array([len(set(i)&set(j))/k for i,j in zip(ai,bi)])


@torch.no_grad()
def perturbations(encoder,patch,full_positions,cell,center,atom_ids,radius,d0,scale,amplitudes,uncertainty,seed=71):
    """Fixed membership and fresh radius extraction are distinct measurements."""
    device=next(encoder.parameters()).device;rng=np.random.default_rng(seed)
    z=encoder(pack([patch],device));result=[]
    for amplitude in amplitudes:
        if amplitude*d0<uncertainty:continue
        delta=rng.normal(size=np.asarray(full_positions).shape)*amplitude*d0;delta[center]=0
        reextracted=extract_patch(full_positions+delta,cell,center,atom_ids,radius)['positions']
        fixed=patch.copy();fixed[1:]+=rng.normal(size=fixed[1:].shape)*amplitude*d0
        for name,x in [('fixed',fixed),('reextracted',reextracted)]:
            change=float((encoder(pack([x],device))-z).norm())
            result.append(dict(amplitude_over_d0=amplitude,membership=name,raw_change=change,scale=scale,sensitivity=change/scale))
    for strain in (-.03,-.01,.01,.03):
        for name,A in [('hydrostatic',np.eye(3)*(1+strain)),('shear',np.array([[1,strain,0],[0,1,0],[0,0,1]]))]:
            change=float((encoder(pack([patch@A.T],device))-z).norm());result.append(dict(perturbation=name,strain=strain,raw_change=change))
    return result


def select_checkpoint(assessments):
    """Do not fabricate a best checkpoint before G2/G3 have been measured."""
    eligible=[r for r in assessments if r['structural_retention_pass'] and r['robustness_pass']]
    if not eligible:raise ValueError('No checkpoint passes structural retention and robustness; selection unresolved')
    return min(eligible,key=lambda r:r['development_nmse'])['checkpoint']


def prototypes():
    """Unit first-shell reference fixtures; never pretraining motif labels."""
    from itertools import product
    phi=(1+5**.5)/2
    fcc=np.array([p for p in product((-1,0,1),repeat=3) if sum(v*v for v in p)==2],float)/2**.5
    bcc=np.array(list(product((-1,1),repeat=3)),float)/3**.5
    angles=np.arange(6)*np.pi/3;hcp=[(np.cos(a),np.sin(a),0) for a in angles]
    for sign in (-1,1):
        hcp.extend([(np.cos(a)/3**.5,np.sin(a)/3**.5,sign*(2/3)**.5) for a in np.arange(3)*2*np.pi/3+np.pi/6])
    ico=np.array([(0,a,b*phi) for a,b in product((-1,1),repeat=2)]+[(a,b*phi,0) for a,b in product((-1,1),repeat=2)]+[(b*phi,0,a) for a,b in product((-1,1),repeat=2)],float)
    ico/=np.linalg.norm(ico,axis=1)[:,None]
    rng=np.random.default_rng(152);disordered=rng.normal(size=(12,3));disordered/=np.linalg.norm(disordered,axis=1)[:,None]
    return {k:np.vstack([np.zeros((1,3)),v]).astype(np.float32) for k,v in dict(FCC=fcc,HCP=np.array(hcp),BCC=bcc,icosahedral=ico,disordered=disordered).items()}
