"""Radius-complete patches, explicit periodic images and replayable corruption."""
from itertools import product
import numpy as np
import torch
from scipy.spatial import cKDTree


def taper(r, radius):
    u=((r/radius-.8)/.2).clamp(0,1)
    return torch.where(r>=radius,0.,torch.where(r<=.8*radius,1.,(1-u)**3*(1+3*u+6*u*u)))


def extract_patch(positions, cell, center, atom_ids, radius, max_atoms=None):
    """General periodic cell (row vectors), with every image inside radius.

    Fractional wrapping is used only to enumerate images, never as a skew-cell
    minimum-image approximation. KD-tree queries run in Cartesian coordinates.
    """
    x=np.asarray(positions,dtype=np.float64);cell=np.asarray(cell,dtype=np.float64)
    if cell.shape!=(3,3) or abs(np.linalg.det(cell))<1e-10:raise ValueError('Noninvertible 3x3 cell')
    inv=np.linalg.inv(cell);frac=x@inv;wrapped=(frac-np.floor(frac))@cell
    bound=np.ceil(radius*np.linalg.norm(inv,axis=0)).astype(int)+1
    tree=cKDTree(wrapped);rows=[];images=[];vectors=[]
    for shift in product(*(range(-int(b),int(b)+1) for b in bound)):
        shift=np.array(shift,dtype=int);target=wrapped[center]-shift@cell
        ids=tree.query_ball_point(target,radius)
        for i in ids:
            v=wrapped[i]-target
            if np.linalg.norm(v)<radius:
                rows.append(i);images.append(shift);vectors.append(v)
    rows=np.asarray(rows,dtype=int);images=np.asarray(images,dtype=int);vectors=np.asarray(vectors)
    central=(rows==center)&(images==0).all(1)
    if central.sum()!=1:raise ValueError('Exactly one distinguished center image required')
    order=np.r_[np.flatnonzero(central),np.flatnonzero(~central)]
    if max_atoms is not None and len(order)>max_atoms:
        raise ValueError(f'Radius support overflow: {len(order)} > N_max={max_atoms}; no neighbors discarded')
    vectors=vectors[order];vectors[0]=0
    return dict(positions=vectors.astype(np.float32),atom_ids=np.asarray(atom_ids)[rows[order]],images=images[order])


def pack(patches, device='cpu'):
    n=max(len(p) for p in patches);b=len(patches)
    x=torch.zeros(b,n,3,device=device);mask=torch.zeros(b,n,dtype=torch.bool,device=device)
    center=torch.zeros_like(mask);species=torch.zeros(b,n,dtype=torch.long,device=device)
    for i,p in enumerate(patches):
        x[i,:len(p)]=torch.as_tensor(p,device=device);mask[i,:len(p)]=True;center[i,0]=True
    return dict(positions=x,mask=mask,center=center,species=species)


def validate_batch(batch):
    x,m,c=batch['positions'],batch['mask'],batch['center']
    if not torch.isfinite(x).all() or (c&~m).any() or not (c.sum(1)==1).all():
        raise ValueError('Finite coordinates, one valid center per patch required')
    if not (x[c]==0).all():raise ValueError('Center must remain exactly at origin')


def edges(positions,mask,cutoff):
    """Packed directed graph; no neighbor cap, distances from these positions only."""
    b,n,_=positions.shape
    distance=torch.cdist(positions,positions,compute_mode='donot_use_mm_for_euclid_dist')
    valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=positions.device)[None]
    bi,sender,receiver=torch.where(valid&(distance<cutoff))
    packed=torch.full((b,n),-1,dtype=torch.long,device=positions.device)
    packed[mask]=torch.arange(int(mask.sum()),device=positions.device)
    return torch.stack((packed[bi,sender],packed[bi,receiver]))


def corrupt(batch, levels, d0, generator, level_ids=None):
    """One balanced level per anchor; center/padding fixed, no COM subtraction."""
    x=batch['positions'];b=len(x)
    if level_ids is None:
        level_ids=(torch.arange(b)+torch.randint(len(levels),(),generator=generator))%len(levels)
        level_ids=level_ids[torch.randperm(b,generator=generator)]
    level_ids=level_ids.to(x.device)
    sigma=x.new_tensor(levels)[level_ids]*d0
    epsilon=torch.randn(x.shape,generator=generator,dtype=x.dtype).to(x.device)
    epsilon=epsilon*(batch['mask']&~batch['center'])[...,None]
    noisy=dict(batch,positions=x+sigma[:,None,None]*epsilon)
    return noisy,epsilon,sigma,level_ids


def coordinate_uncertainty(positions):
    """Conservative per-component center-relative rounding bound, native dtype.

    Sum half ULPs of neighbor and center. This is an upper bound, not a claim
    to recover the unquantized trajectory or a measured RMS error.
    """
    x=np.asarray(positions)
    gap=np.maximum(abs(np.nextafter(x,np.inf,dtype=x.dtype).astype(float)-x),
                   abs(x.astype(float)-np.nextafter(x,-np.inf,dtype=x.dtype).astype(float)))
    return float(np.max(gap))  # two half-ULPs, center plus neighbor


def allowed_levels(levels,d0,uncertainty):
    kept=[s for s in levels if s*d0>=10*uncertainty]
    if not kept:raise ValueError(f'All noise levels below 10x coordinate uncertainty ({uncertainty:g} A); use higher-precision data')
    return kept


class BalancedStream:
    """Uniform root -> source -> time block -> anchor, all on training records."""
    def __init__(self,records,seed):
        self.rng=np.random.default_rng(seed);self.tree={};self.exposures=0
        for i,r in enumerate(records):
            self.tree.setdefault(r['root'],{}).setdefault(r['source'],{}).setdefault(r['block'],[]).append(i)
    def draw(self,n):
        result=[]
        for _ in range(n):
            branch=self.tree
            for _ in range(3):branch=branch[list(branch)[self.rng.integers(len(branch))]]
            result.append(branch[self.rng.integers(len(branch))])
        self.exposures+=n
        return np.array(result)
    def state_dict(self):return dict(rng=self.rng.bit_generator.state,exposures=self.exposures)
    def load_state_dict(self,state):self.rng.bit_generator.state=state['rng'];self.exposures=state['exposures']


def audit_corruptions(patches,levels,d0,radius,seed=41):
    batch=pack(patches);g=torch.Generator().manual_seed(seed);out={}
    x=batch['positions'];valid=batch['mask']&~batch['center']
    clean_dist=torch.cdist(x,x);n=x.shape[1]
    pair=batch['mask'][:,:,None]&batch['mask'][:,None,:]&~torch.eye(n,dtype=torch.bool)[None]
    nearest=clean_dist.masked_fill(~pair,torch.inf).argmin(-1)
    for k,level in enumerate(levels):
        noisy,eps,sigma,_=corrupt(batch,levels,d0,g,torch.full((len(x),),k))
        y=noisy['positions'];dist=torch.cdist(y,y)
        nearest_noisy=dist.masked_fill(~pair,torch.inf).argmin(-1)
        out[str(level)]=dict(variance=float(eps[valid].var()),nearest_identity_change=float((nearest!=nearest_noisy)[valid].float().mean()),
            collision_fraction=float((dist[pair]<.5*d0).float().mean()),finite=bool(torch.isfinite(y).all()),
            mean_decoder_degree=float(((dist<2*d0)&pair).sum(-1)[batch['mask']].float().mean()),
            min_pair_distance_A=float(dist[pair].min()),boundary_crossings=int(((y.norm(dim=-1)>=radius)&valid).sum()))
    return out


def balanced_subset(records,indices,limit,seed=73,key='root'):
    """Fixed evaluation subset: round-robin groups, no duplicate scarce anchors."""
    rng=np.random.default_rng(seed);groups={}
    for i in indices:groups.setdefault(records[i][key],[]).append(i)
    queues=[list(rng.permutation(v)) for _,v in sorted(groups.items())];chosen=[]
    while len(chosen)<min(limit,len(indices)):
        for q in queues:
            if q and len(chosen)<limit:chosen.append(int(q.pop()))
    return chosen
