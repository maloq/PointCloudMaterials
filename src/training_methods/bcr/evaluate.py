"""Held-out reconstruction interventions and root-paired uncertainty."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .data import corrupt,pack
from .objective import per_environment


def code_statistics(z):
    z=z.double();centered=z-z.mean(0);cov=centered.T@centered/(len(z)-1)
    eig=torch.linalg.eigvalsh(cov).clamp_min(0)
    return dict(mean=z.mean(0).tolist(),std=z.std(0).tolist(),norm_mean=float(z.norm(dim=1).mean()),
        covariance_trace=float(eig.sum()),effective_rank=float(eig.sum().square()/eig.square().sum().clamp_min(1e-30)),spectrum=eig.tolist())


def matching_bins(train_covariates):
    """Count, density, order edges fitted on training observations only."""
    return [np.unique(np.quantile(train_covariates[:,i],[.2,.4,.6,.8])).tolist() for i in range(3)]


def derangement(roots,conditions,covariates,bins,seed,relax=0):
    """One-to-one partial derangement; never silently loosen exact matches.

    relax=0: condition, count/density/order bins, other root;
    relax=1: condition/count bins and other root; relax=2: unrestricted other root.
    Unmatched rows are -1, excluded from paired true/swap comparisons.
    """
    rng=np.random.default_rng(seed);n=len(roots)
    encoded=np.stack([np.digitize(covariates[:,i],bins[i]) for i in range(3)],1)
    allowed=np.asarray(roots)[:,None]!=np.asarray(roots)[None,:]
    if relax<2:allowed&=np.asarray(conditions)[:,None]==np.asarray(conditions)[None,:]
    for j in range(3 if relax==0 else 1 if relax==1 else 0):allowed&=encoded[:,j,None]==encoded[None,:,j]
    costs=np.where(allowed,rng.uniform(0,1,(n,n)),1e6)
    rows,cols=linear_sum_assignment(costs);out=np.full(n,-1,dtype=int)
    keep=allowed[rows,cols];out[rows[keep]]=cols[keep]
    return out


def paired_root_gain(true,other,roots,seed=0,draws=1000):
    """Average corruptions per anchor before calling; then equal root means."""
    true=np.asarray(true);other=np.asarray(other);roots=np.asarray(roots)
    valid=np.isfinite(true)&np.isfinite(other);unique=np.unique(roots[valid])
    if not len(unique):return dict(coverage=0.,roots=0,gain=None,ci95=None)
    values=np.array([[true[valid&(roots==r)].mean(),other[valid&(roots==r)].mean()] for r in unique])
    gain=lambda a:float((a[:,1].mean()-a[:,0].mean())/max(a[:,1].mean(),1e-30))
    rng=np.random.default_rng(seed);samples=[gain(values[rng.integers(len(values),size=len(values))]) for _ in range(draws)]
    return dict(coverage=float(valid.mean()),roots=len(unique),true_nmse=float(values[:,0].mean()),other_nmse=float(values[:,1].mean()),
        delta=float((values[:,1]-values[:,0]).mean()),gain=gain(values),ci95=np.quantile(samples,[.025,.975]).tolist() if len(unique)>=2 else None)


@torch.no_grad()
def reconstruction(model,patches,records,levels,draws=2,seed=731,unconditional=None,bins=None,covariates=None,shuffles=4,chunk=8,include_swaps=True):
    device=next(model.parameters()).device;model.eval();z=torch.cat([model.encode(pack(patches[i:i+chunk],device)) for i in range(0,len(patches),chunk)])
    roots=[r['root'] for r in records];conditions=[r['temperature_K'] for r in records];radius=model.config['encoder']['radius'];d0=model.config['encoder']['d0']
    output=dict(code=code_statistics(z),anchors=len(patches),roots=len(set(roots)),levels={})
    for k,level in enumerate(levels):
        rows={name:[] for name in ('weighted','unweighted','interior','middle','outer','unconditional')}
        swapped={relax:[] for relax in range(3)}
        maps={relax:[derangement(roots,conditions,covariates,bins,seed+13*j,relax) for j in range(shuffles)] for relax in range(3)} if bins is not None and include_swaps else {}
        for draw in range(draws):
            errors={name:[] for name in rows};swap_errors={relax:[[] for _ in range(shuffles)] for relax in maps}
            for start in range(0,len(patches),chunk):
                clean=pack(patches[start:start+chunk],device)
                # Anchor-specific keys independent of evaluation chunking.
                ys=[];es=[];ss=[]
                for idx in range(start,min(start+chunk,len(patches))):
                    b={name:v[idx-start:idx-start+1] for name,v in clean.items()}
                    y,e,s,_=corrupt(b,levels,d0,torch.Generator().manual_seed(seed+100003*idx+1009*k+draw),torch.tensor([k]))
                    ys.append(y['positions']);es.append(e);ss.append(s)
                noisy=dict(clean,positions=torch.cat(ys));epsilon=torch.cat(es);sigma=torch.cat(ss)
                code=(model.constant[None].expand(len(sigma),-1) if model.arm=='unconditional' else z[start:start+len(sigma)])
                pred=model.decoder(noisy['positions'],noisy['species'],noisy['center'],noisy['mask'],code,torch.log(sigma/d0))
                for name in rows:
                    if name=='unconditional':
                        value=per_environment(unconditional(clean,noisy,sigma)[0],epsilon,clean,radius) if unconditional else torch.full((len(sigma),),torch.nan,device=device)
                    else:value=per_environment(pred,epsilon,clean,radius,name)
                    errors[name].extend(value.cpu().tolist())
                for relax,mappings in maps.items():
                    for j,mapping in enumerate(mappings):
                        ids=mapping[start:start+len(sigma)];swapz=z[torch.as_tensor(ids.clip(0),device=device)]
                        p=model.decoder(noisy['positions'],noisy['species'],noisy['center'],noisy['mask'],swapz,torch.log(sigma/d0))
                        err=per_environment(p,epsilon,clean,radius).cpu().numpy();err[ids<0]=np.nan;swap_errors[relax][j].extend(err.tolist())
            for name in rows:rows[name].append(errors[name])
            for relax in maps:swapped[relax].extend(swap_errors[relax])
        result={}
        for name,v in rows.items():
            a=np.array(v);finite=np.isfinite(a);total=np.where(finite,a,0).sum(0);count=finite.sum(0)
            result[name]=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
        true=result['weighted'];report={name:float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else None for name,v in result.items()}
        report['coordinate_RMSE_A']=float(level*d0*np.sqrt(true.mean()))
        report['unconditional_gain']=paired_root_gain(true,result['unconditional'],roots)
        for relax,values in swapped.items():
            if not values:continue
            a=np.asarray(values);n=np.isfinite(a).sum(0);avg=np.divide(np.nansum(a,0),n,out=np.full(len(patches),np.nan),where=n>0)
            report[f'swap_relax{relax}']=paired_root_gain(true,avg,roots)
            report[f'per_anchor_swap_relax{relax}_nmse']=avg.tolist()
        report['per_anchor_unconditional_nmse']=result['unconditional'].tolist()
        report['per_anchor_nmse']=true.tolist();output['levels'][str(level)]=report
    return output
