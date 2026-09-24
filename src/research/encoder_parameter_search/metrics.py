"""Fit-scaled errors and liquid retrieval without test-variance division."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.research.geoframe_evolution.metrics import classification


def resolution(z, target, density, split, liquid):
    fit=(split==0)&liquid; test=(split==1)&liquid
    result=dict(fit_count=int(fit.sum()),test_count=int(test.sum()))
    if min(fit.sum(),test.sum())<40:
        return dict(result,eligible=False)
    target=np.asarray(target,dtype=np.float64)
    ys=StandardScaler().fit(target[fit]);active=ys.var_>1e-12
    if not active.any(): return dict(result,eligible=False)
    y=ys.transform(target)[:,active]
    constant=float(np.square(y[test]).mean())
    result.update(eligible=True,columns=np.flatnonzero(active).tolist(),constant_nmse=constant)
    for name,x in [('embedding',z),('density',density[:,None]),('joint',np.c_[density,z])]:
        xs=StandardScaler().fit(x[fit]);xx=xs.transform(x)
        prediction=Ridge(alpha=10.,solver='svd').fit(xx[fit],y[fit]).predict(xx[test])
        result[name+'_nmse']=float(np.square(prediction-y[test]).mean())
    # Feature distances are raw exported Euclidean distances, not whitened fits.
    for name,x in [('embedding',z),('density',density[:,None])]:
        indices=NearestNeighbors(n_neighbors=10,n_jobs=1).fit(x[fit]).kneighbors(x[test],return_distance=False)
        result[name+'_neighbor_nmse']=float(np.square(y[fit][indices]-y[test,None]).mean())
    result['conditional_nmse_gain']=result['density_nmse']-result['joint_nmse']
    result['retrieval_gain_vs_density']=result['density_neighbor_nmse']-result['embedding_neighbor_nmse']
    return result


def frame(z, arrays, record):
    n=record['anchor_count'];split=arrays['split'][:n]
    order=arrays['order'][:n,0 if record['material']=='Al' else 1]
    liquid=~np.isin(arrays['ptm'][:n,1],[1,2,3])&(arrays['solid_fraction'][:n,1]<=.1)
    ti=arrays['topology_rows'];context=arrays['context'][:n,1]
    keep=np.isin(context,[0,2,3,5,6])
    return dict(liquid_order=resolution(z[:n],order[:,:6],order[:,6],split,liquid),
        liquid_topology=resolution(z[ti],arrays['topology'],order[ti,6],split[ti],liquid[ti]),
        nonbulk_context=classification(z[:n][keep],context[keep],split[keep],[0,2,3,5,6]))
