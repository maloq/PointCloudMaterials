"""Bounded-memory, source-weighted feature statistics and normalization."""
import numpy as np


class Moments:
    def __init__(self, field):
        self.field=field;self.mass=0.;self.mean=None;self.m2=None

    def add(self, values, weights):
        x=np.asarray(values,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
        mass=float(w.sum())
        if mass<=0:return
        if self.field=='z':
            mean=np.einsum('b,bnd->d',w,x)/(mass*x.shape[1])
            m2=np.einsum('b,bnd->d',w,(x-mean)**2)/x.shape[1]
            if self.mass:
                delta=mean-self.mean;total=self.mass+mass
                self.m2+=m2+delta**2*self.mass*mass/total
                self.mean+=delta*mass/total
            else:self.mean,self.m2=mean,m2
        else:
            m2=np.einsum('b,bncm->c',w,x*x)/(x.shape[1]*x.shape[3])
            self.m2=m2 if self.m2 is None else self.m2+m2
        self.mass+=mass

    def result(self):
        if not np.isclose(self.mass,1.,rtol=1e-10,atol=1e-10):
            raise ValueError(f'Incomplete training weight for {self.field}: {self.mass}')
        scale=np.sqrt(self.m2/self.mass).clip(1e-5)
        if self.field!='z':scale=scale[:,None]
        return dict(mean=self.mean.tolist() if self.field=='z' else 0.,scale=scale.tolist())


def apply(values, statistics):
    if statistics is None:return np.asarray(values,dtype=np.float32)
    result=((np.asarray(values,dtype=np.float64)-np.asarray(statistics['mean']))/
            np.asarray(statistics['scale'])).astype(np.float32)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite normalized context field')
    return result
