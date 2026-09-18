"""Radially matched pairs and source-level conditional predictive comparisons."""
from itertools import combinations
import numpy as np


def matched_pairs(a,config):
    left=[];right=[]
    for source in np.unique(a['source']):
        rows=np.flatnonzero(a['source']==source)
        frames=np.unique(a['frame'][rows])
        for frame in frames:
            current=rows[a['frame'][rows]==frame]
            if len(np.unique(a['atom'][current]))!=len(current):raise ValueError('Duplicate atom/frame observation')
            i,j=np.triu_indices(len(current),1);left.append(current[i]);right.append(current[j])
    left=np.concatenate(left);right=np.concatenate(right)
    radial=np.empty(len(left));full=np.empty(len(left))
    for start in range(0,len(left),10000):
        end=start+10000;l,r=left[start:end],right[start:end]
        radial[start:end]=np.sqrt(np.mean((a['radii80'][l]-a['radii80'][r])**2,axis=1))
        full[start:end]=np.sqrt(np.mean((a['radial_quantiles'][l]-a['radial_quantiles'][r])**2,axis=1))
    rho=a['order'][:,6].astype(float)
    density=np.abs(rho[left]-rho[right])/((rho[left]+rho[right])/2)
    return dict(left=left,right=right,radial_rms_A=radial,full_radial_rms_A=full,density_relative=density)


def pair_mask(pairs,caliper,config):
    return ((pairs['radial_rms_A']<=caliper)&(pairs['full_radial_rms_A']<=caliper)
            &(pairs['density_relative']<=config['match_density_relative']))


def improvement(baseline,candidate,draws):
    """Percent reduction in equal-source mean loss; paired source resampling."""
    baseline=np.asarray(baseline,dtype=float);candidate=np.asarray(candidate,dtype=float)
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all() or np.any(baseline<0):
        raise ValueError('Paired improvement requires finite source errors')
    if baseline.mean()<=0:raise ValueError('Undefined relative improvement against zero loss')
    estimate=100*(1-candidate.mean()/baseline.mean())
    samples=100*(1-candidate[draws].mean(1)/baseline[draws].mean(1))
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(improvement_percent=float(estimate),low=float(lo),high=float(hi),
        baseline_loss=float(baseline.mean()),candidate_loss=float(candidate.mean()))
