"""Same fast continuous-motion student, initialized from unrelaxed Small eSEN."""
import torch
from torch import nn
from src.models.encoders.spatial_esen import SpatialESEN
from src.models.encoders.esen_runtime import index_permutations
from src.research.structural_response.fast import FastMotion
from .data import resolve_path

class Motion(FastMotion):
    def __init__(self,c):
        nn.Module.__init__(self)
        self.geometry=SpatialESEN(**c['encoder']);self.arm='continuous'
        self.time_scale=c['model']['time_scale_ps'];self.step_ps=c['model']['step_ps'];self.d=128
        hidden=c['model']['hidden']
        for name,out in [('acceleration',128),('diffusion',128),('readout',48)]:
            net=nn.Sequential(nn.Linear(256,hidden),nn.SiLU(),nn.Linear(hidden,out))
            nn.init.normal_(net[-1].weight,std=.001);nn.init.constant_(net[-1].bias,-3. if name=='diffusion' else 0.)
            setattr(self,name,net)
        self.velocity_epsilon=c['student_difference']['velocity_epsilon'];self.response_epsilon=c['student_difference']['response_epsilon']
        self.adapter=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias);self.use_adapter=False
        saved=torch.load(resolve_path(c['checkpoint']),map_location='cpu',weights_only=False)
        self.geometry.load_state_dict(saved['geometry']);index_permutations(self.geometry)
