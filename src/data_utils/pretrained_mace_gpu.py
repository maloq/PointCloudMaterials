"""GPU-resident view tensors for the prepared repository MACE quadruplets."""
import numpy as np
import torch


class GPUQuadruplets:
    """Keep stored coordinate precision and fixed train-only target projection."""
    def __init__(self,data,scaling):
        self.offsets=np.r_[0,np.cumsum([r['anchors_count'] for r in data.records])]
        count=int(self.offsets[-1]);views,points=data.clouds[0].shape[1:3]
        self.clouds=torch.empty((count,views,points,3),dtype=torch.float16,device='cuda')
        self.targets=torch.empty((count,views,len(scaling['tda_std'])),dtype=torch.float32,device='cuda')
        self.conditions=torch.empty((count,5),dtype=torch.float32,device='cuda')
        self.materials=torch.empty(count,dtype=torch.long,device='cuda')
        for i,record in enumerate(data.records):
            sl=slice(self.offsets[i],self.offsets[i+1])
            self.clouds[sl].copy_(torch.from_numpy(np.array(data.clouds[i],copy=True)))
            target=((data.tda[i]-scaling['tda_mean'])@scaling['tda_components'].T)/scaling['tda_std']
            self.targets[sl].copy_(torch.from_numpy(target.astype(np.float32)))
            self.conditions[sl].copy_(torch.from_numpy(data.conditions[i]))
            self.materials[sl]=record['material']
        self.bytes=sum(t.numel()*t.element_size() for t in (self.clouds,self.targets,self.conditions,self.materials))

    def get(self,indices):
        rows=torch.from_numpy(self.offsets[indices[:,0]]+indices[:,1]).to('cuda')
        return self.clouds[rows].float(),self.targets[rows],self.conditions[rows],self.materials[rows]
