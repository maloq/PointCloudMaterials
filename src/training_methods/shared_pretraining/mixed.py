"""Mixed-domain snapshot batches, within-domain VICReg and three-frame curvature."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.prepare import digest
from src.training_methods.structural_pretraining.objective import Objective,vicreg,physical_correlation_loss


def quotas(weights,size,minimum):
    """Proportional largest-remainder allocation with a minimum for every domain."""
    weights=np.asarray(weights,dtype=float)
    if minimum<2 or size<minimum*len(weights) or np.any(weights<=0):
        raise ValueError('Mixed statistical batch cannot satisfy per-domain minimum')
    active=np.arange(len(weights));result=np.zeros(len(weights),dtype=int);remaining=size
    while len(active):
        desired=remaining*weights[active]/weights[active].sum()
        small=desired<minimum
        if small.any():
            result[active[small]]=minimum;remaining-=minimum*int(small.sum());active=active[~small]
        else:
            counts=np.floor(desired).astype(int)
            counts[np.argsort(-(desired-counts),kind='stable')[:remaining-int(counts.sum())]]+=1
            result[active]=counts;break
    return result


def group_ids(release,indices):
    lookup={k:i for i,k in enumerate(release.group_keys)}
    return np.array([lookup[(r['material'],r['potential'],r['static'])]
                     for i in indices for r in [release.rows[i][2]]],dtype=np.int64)


def prepare(release,step,config,*,pin_memory=True):
    if config['architecture'] not in ('gatr','mace') or config['history_frames']!=1 or config['method']!='vicreg':
        raise ValueError('Mixed triplets require snapshot MACE/GATr and VICReg')
    architecture=config['architecture'];mace=architecture=='mace'
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],step]))
    counts=quotas(release.group_weights,config['batch_size'],config['minimum_group_size'])
    indices=[]
    for group,count in zip(release.group_keys,counts,strict=True):
        if group[2]:raise ValueError('Static group entered dynamic-only training')
        if len(release.groups[group])<count:raise ValueError(f'Insufficient anchors for mixed group {group}')
        indices.extend(rng.choice(release.groups[group],int(count),replace=False).tolist())
    rng.shuffle(indices);temporal=bool(rng.integers(2))
    extra=dict(domain=group_ids(release,indices))
    delta=np.zeros(len(indices),dtype=np.float32)
    endpoints=['anchor','spatial']
    if temporal:
        gaps=[]
        for index in indices:
            name,row,_=release.rows[index];a=release.arrays[name];mapping=a['views'][row]
            if not np.all(a['center_ids'][mapping[[1,2,3]]]==a['center_ids'][mapping[2]]):
                raise ValueError(f'Triplet lost persistent atom identity in {name}, row {row}')
            times=a['times'];gaps.append([float(times[2]-times[1]),float(times[3]-times[2])])
        gaps=np.asarray(gaps,dtype=np.float32)
        if not np.isfinite(gaps).all() or np.any(gaps<=0):raise ValueError('Triplet time gaps must be positive')
        extra['triplet_dt']=gaps;delta=gaps[:,1]
        # The past frame is encoder-only curvature context, never a zero label.
        endpoints=['anchor','future','previous']
    micro=config['microbatch_size'];workers=config.get('preparation_workers',1)
    if workers<1:raise ValueError('preparation_workers must be positive')
    chunks=[(which,start) for which in endpoints for start in range(0,len(indices),micro)]
    def pack(chunk):
        which,start=chunk
        samples=[release.observation(i,which,False,mace) for i in indices[start:start+micro]]
        batch=collate(samples,architecture,bond_order=(mace or config.get('bond_order_weight',0)>0))
        return {k:v.pin_memory() if pin_memory and torch.cuda.is_available() else v for k,v in batch.items()}
    if workers==1:batches=[pack(chunk) for chunk in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map preserves endpoint and row order, regardless of completion order.
            batches=list(pool.map(pack,chunks))
    return batches,temporal,delta.tolist(),indices,['mixed','material-potential',False],False,extra


def backtracking(previous,current,future,delta):
    """Mean squared Euclidean interpolation residual; exact second difference at equal dt.

    2*(h_prev*z_next + h_next*z_prev)/(h_prev+h_next) - 2*z_current
    vanishes for constant velocity at irregular timestamps. It has state units,
    not acceleration units: there is no inverse-dt-squared amplification.
    """
    with torch.autocast(current.device.type,enabled=False):
        previous,current,future=previous.float(),current.float(),future.float()
        h0,h1=delta.float().unbind(-1)
        residual=2*(h0[:,None]*(future-current)+h1[:,None]*(previous-current))/(h0+h1)[:,None]
        return residual.square().sum(-1).mean()


class MixedObjective(Objective):
    def __init__(self,normalization,group_keys,correlation_weight,backtracking_weight):
        super().__init__(normalization,'vicreg',correlation_weight)
        if backtracking_weight<0:raise ValueError('Backtracking weight must be nonnegative')
        self.group_keys=tuple(tuple(k) for k in group_keys);self.backtracking_weight=backtracking_weight

    def forward(self,model,z,targets,temporal,delta):
        domains=targets['domain'];n=len(domains)
        if len(z)!=(3 if temporal else 2)*n:raise ValueError('Wrong snapshot ordering for curvature')
        target={k:targets[k][:2*n] for k in ('physical','tda','tda_valid')}
        if not bool(target['tda_valid'].all()):raise ValueError('Every supervised mixed view requires instantaneous TDA')
        heads=model.heads(z[:2*n],domains.repeat(2));p,h=self.physical_errors(heads,target)
        q0,q1=heads['q'].chunk(2);terms={};reg=z.sum()*0;corr=z.sum()*0
        for group,key in enumerate(self.group_keys):
            mask=domains==group;weight=mask.sum()/n
            if int(mask.sum())<2:raise ValueError(f'VICReg group {key} has fewer than two pairs')
            value,details=vicreg(q0[mask],q1[mask]);reg=reg+weight*value
            both=mask.repeat(2)
            contrast=physical_correlation_loss(heads['physical'][both],target['physical'][both])
            corr=corr+weight*contrast
            for name,v in details.items():terms[name]=terms.get(name,0)+weight*v
            prefix=f'groups/{key[0]}/{key[1]}'
            terms.update({f'{prefix}/physical':p[both].mean(),f'{prefix}/instantaneous_tda':h[both].mean(),
                f'{prefix}/vicreg':details['vicreg'],f'{prefix}/pairs':mask.sum(),
                f'{prefix}/state_std':z[:n][mask].std(0).mean()})
        curve=(backtracking(z[2*n:3*n],z[:n],z[n:2*n],targets['triplet_dt'])
               if temporal else z.new_zeros(()))
        curve_weighted=self.backtracking_weight*curve
        loss=p.mean()+.25*h.mean()+.1*reg+self.physical_correlation_weight*corr+curve_weighted
        terms.update(loss=loss,physical=p.mean(),instantaneous_tda=h.mean(),representation=reg,
            vicreg_weighted=.1*reg,physical_correlation_loss=corr,
            physical_correlation_weighted=self.physical_correlation_weight*corr,
            backtracking=curve,backtracking_weighted=curve_weighted,
            backtracking_loss_fraction=curve_weighted/loss.detach().clamp_min(1e-12),
            labelled_views=target['tda_valid'].sum(),state_std=z[:2*n].std(0).mean(),
            physical_prediction_std=heads['physical'].std(0).mean(),tda_prediction_std=heads['tda'].std(0).mean())
        return loss,terms


class BondObjective(MixedObjective):
    def __init__(self,normalization,group_keys,correlation_weight,backtracking_weight,bond_order_weight):
        super().__init__(normalization,group_keys,correlation_weight,backtracking_weight)
        if bond_order_weight<=0:raise ValueError('Bond-order objective requires a positive weight')
        self.bond_order_weight=bond_order_weight

    def forward(self,model,features,targets,temporal,delta):
        from src.data.structural_pretraining.bond_order import bond_order_errors
        n=2*len(targets['domain'])
        loss,terms=super().forward(model,features[:,:128],targets,temporal,delta)
        prediction=model.bond_order(features[:n,128:])
        errors=bond_order_errors(prediction,targets['bond_order'][:n])
        weighted=self.bond_order_weight*errors.mean();loss=loss+weighted
        terms.update(loss=loss,bond_order=errors.mean(),bond_order_q4=errors[:,0].mean(),
            bond_order_q6=errors[:,1].mean(),bond_order_weighted=weighted,
            backtracking_loss_fraction=terms['backtracking_weighted']/loss.detach().clamp_min(1e-12))
        return loss,terms


def calibration_indices(release,seed,per_group):
    if per_group<2:raise ValueError('At least two training calibration anchors per group are required')
    indices=[]
    for group,key in enumerate(release.group_keys):
        if len(release.groups[key])<per_group:raise ValueError(f'Insufficient training reference for {key}')
        rng=np.random.default_rng(np.random.SeedSequence([seed,48731,group]))
        indices.extend(rng.choice(release.groups[key],per_group,replace=False).tolist())
    return indices


@torch.no_grad()
def calibrate(model,release,config):
    indices=calibration_indices(release,config['seed'],config['calibration_anchors_per_group'])
    was_training=model.training;model.eval();device=next(model.parameters()).device
    states=[];domains=[];micro=min(config['microbatch_size'],64)
    # All three supervised view populations, each treated as independent snapshots.
    for which in ('anchor','spatial','future'):
        for start in range(0,len(indices),micro):
            rows=indices[start:start+micro]
            architecture=config['architecture']
            batch=move(collate([release.observation(i,which,False,architecture=='mace') for i in rows],architecture),device)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                states.append(model.encoder(batch).float())
            domains.extend(group_ids(release,rows).tolist())
    z=torch.cat(states);g=torch.tensor(domains,device=device,dtype=torch.long)
    model.calibrate(z,g);model.train(was_training)
    return dict(indices_sha256=digest(indices),anchors_per_group=config['calibration_anchors_per_group'],
        groups=[list(k) for k in release.group_keys],rows=len(z),
        group_state_std={str(k):float(z[g==i].double().std(0,unbiased=False).mean()) for i,k in enumerate(release.group_keys)})
