"""Short-horizon scores derived from the same fully open-loop 96ps forecast."""
import numpy as np
from src.research.crystallization_information.runtime import score_hazard


def short_logits(cdf):
    p=np.asarray(cdf,dtype=np.float64)[:,np.array([1,4,8,12,16])-1]
    if not np.isfinite(p).all() or (np.diff(p,axis=1)<-1e-6).any():raise ValueError('Invalid forecast CDF')
    previous=np.c_[np.zeros(len(p)),p[:,:-1]]
    hazard=((p-previous)/np.maximum(1-previous,1e-12)).clip(1e-7,1-1e-7)
    return np.log(hazard)-np.log1p(-hazard)


def short_scores(data,test,calibration,prediction,calibration_prediction):
    ids=np.r_[test,calibration];delay=np.r_[prediction['event'],calibration_prediction['event']]+1
    pop=dict(source=data.corpus.source_ids[ids],event=np.searchsorted([1,4,8,12,16],delay),
        delay=.75*delay,temperature=np.array([data.corpus.rows[int(i)][3] for i in ids]))
    return score_hazard(pop,np.arange(len(test)),short_logits(prediction['cdf']),np.arange(len(test),len(ids)),short_logits(calibration_prediction['cdf']))
