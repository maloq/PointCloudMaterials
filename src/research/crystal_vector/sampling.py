"""Independent random draws from the declared population, without label quotas."""
import numpy as np


class RandomBatches:
    def __init__(self,data,size):
        self.ids=data.split['train'];self.probability=data.weights['train'];self.size=size
        if not np.isclose(self.probability.sum(),1.) or (self.probability<0).any():
            raise ValueError('Invalid target-population probabilities')
        self.steps=int(np.ceil(len(self.ids)/size))

    def batch(self,rng):
        return rng.choice(self.ids,self.size,replace=True,p=self.probability)

    def audit(self,distance,inside_crystal=None):
        # Labels quantify expected coverage only; they never influence draws.
        d=distance[self.ids]
        masks={'zero_distance':d==0,'near_0_8':(d>0)&(d<=8),'near_8_20':(d>8)&(d<=20),
            'near_0_20':(d>0)&(d<=20),'far_over20':d>20}
        if inside_crystal is not None:masks['inside_crystal']=inside_crystal[self.ids]
        result={}
        for name,mask in masks.items():
            p=float(self.probability[mask].sum())
            result[name]=dict(rows=int(mask.sum()),probability=p,expected_per_batch=self.size*p,
                count_std=float(np.sqrt(self.size*p*(1-p))),empty_batch_probability=float((1-p)**self.size))
        return dict(method='independent_random_target_population',replacement=True,batch_size=self.size,
            target='half fixed-at-risk, half uniform; equal sources within each population',
            uses_labels_for_sampling=False,per_batch_quotas=False,loss_weights='unit',coverage=result)
