"""The same conditional JEPA predictor, queried at additional physical time offsets."""
import torch
from ..regularization.model import Model as OriginalModel


class Model(OriginalModel):
    def __init__(self,channels,spec,seed):
        super().__init__(channels,spec,seed)
        self.horizons_ps=tuple(spec['horizons_ps'])

    def future_embeddings(self,current,temperature):
        conditioned=torch.cat((current[:,:128]+self.condition((temperature[:,None]-460.)/100.),current[:,128:]),-1)
        lag=current.new_tensor(self.horizons_ps)[None].expand(len(current),-1)
        position=current.new_zeros((len(current),len(self.horizons_ps),3))
        return self.predict(conditioned,None,position,lag)
