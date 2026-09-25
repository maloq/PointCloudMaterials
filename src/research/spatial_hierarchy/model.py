"""Region tokens feed back into local MACE before the next spatial operation."""
import numpy as np
import torch
from torch import nn
from e3nn import o3
from src.training_methods.bcr.data import taper
from src.models.encoders.mace_causal import normalize_atom_features
from src.research.structural_state.model import GeometryEncoder
from src.research.structural_state.data import graph_arrays
from src.research.robust_onset.model import Model as BaseModel,GraphBank as BaseBank

TOKEN_DIM=49  # radial6, l2/l4 cross-radial Gram21 each, smooth count1


def region_tokens(patches,radii):
    """Smooth invariant basis; no hard atom counts or count normalization.

    Regions are nested balls centered on the same focal atom. Harmonic moments
    use continuous radial weights. These are coarse geometry summaries, not
    pretrained atom embeddings, crystalline labels or future measurements.
    """
    result=[];indices=torch.triu_indices(6,6)
    for patch in patches:
        x=torch.as_tensor(patch[1:],dtype=torch.float64);r=x.norm(dim=-1)
        if len(x)==0 or not torch.isfinite(x).all() or (r<=0).any():raise ValueError('Invalid regional geometry')
        harmonics=[o3.spherical_harmonics(l,x,normalize=True,normalization='component') for l in (2,4)]
        tokens=[]
        for radius in radii:
            w=taper(r,radius);radial=torch.exp(-((r[:,None]/radius-torch.linspace(0,1,6))/.2)**2)
            norm=128.*(radius/8.)**3;weighted=radial*w[:,None]/norm
            fields=[weighted.sum(0)]
            for y in harmonics:
                moment=weighted.T@y;gram=moment@moment.T/y.shape[1]
                fields.append(gram[indices[0],indices[1]])
            fields.append((w.sum()/norm).reshape(1));tokens.append(torch.cat(fields))
        result.append(torch.stack(tokens))
    return torch.stack(result).numpy().astype(np.float32)


class ContextBank(BaseBank):
    def __init__(self,arrays,encoder,device):
        super().__init__(arrays,encoder,device)
        if arrays['context'].shape!=(len(self.offsets)-1,3,TOKEN_DIM):raise ValueError('Context/local graph alignment mismatch')
        self.context=torch.as_tensor(arrays['context'],device=device)

    def batch(self,indices):
        graph=super().batch(indices)
        graph['context']=self.context[torch.as_tensor(np.asarray(indices),device=self.device)]
        return graph


class ContextEncoder(GeometryEncoder):
    def __init__(self,radii,early,**config):
        super().__init__(**config);self.radii=tuple(radii);self.early=early;c=self.channels
        self.register_buffer('context_mean',torch.zeros(3,TOKEN_DIM))
        self.register_buffer('context_scale',torch.ones(3,TOKEN_DIM))
        self.token_embed=nn.Sequential(nn.Linear(TOKEN_DIM,c),nn.SiLU(),nn.Linear(c,c))
        self.level=nn.Parameter(torch.randn(3,c)*.01)
        self.feedback=nn.ModuleList([nn.Linear(c,c) for _ in self.interactions])
        self.query=nn.ModuleList([nn.Linear(c,c,bias=False) for _ in self.interactions])
        self.value=nn.ModuleList([nn.Linear(c,c) for _ in self.interactions])
        self.scalar_update=nn.ModuleList([nn.Linear(c,c) for _ in self.interactions])
        self.gates=nn.ModuleList([nn.Linear(c,3*c) for _ in self.interactions])
        self.final_query=nn.Linear(2*c,c,bias=False)
        self.final_fusion=nn.Linear(c,2*c)

    def pooled_graph(self,graph):
        attrs,w,g=graph['attrs'],graph['weight'],graph['group'];c=self.channels;b=graph['size']
        tokens=self.token_embed((graph['context']-self.context_mean)/self.context_scale)+self.level
        h=self.node_embedding(attrs)+self.center_embedding(graph['center'])
        for k,(interaction,product) in enumerate(zip(self.interactions,self.products,strict=True)):
            # Bottom-up evidence from the local atoms refines each parent token.
            summaries=[]
            for radius in self.radii:
                weight=taper(graph['radius'],radius)
                summaries.append(h.new_zeros(b,c).index_add(0,g,h[:,:c]*weight[:,None])/self.n_ref)
            tokens=tokens+.1*torch.tanh(self.feedback[k](torch.stack(summaries,1)))
            if self.early:
                attention=torch.softmax(torch.einsum('nc,nsc->ns',self.query[k](h[:,:c]),tokens[g])/c**.5,dim=-1)
                message=torch.einsum('ns,nsc->nc',attention,self.value[k](tokens)[g])
                scalar=h[:,:c]+.1*torch.tanh(self.scalar_update[k](message))
                if h.shape[1]==c:
                    h=scalar
                else:
                    gates=1+.1*torch.tanh(self.gates[k](message))
                    h=torch.cat([scalar*gates[:,:c],
                        (h[:,c:4*c].reshape(-1,c,3)*gates[:,c:2*c,None]).flatten(1),
                        (h[:,4*c:].reshape(-1,c,5)*gates[:,2*c:,None]).flatten(1)],1)
            m,skip=interaction(node_attrs=attrs,node_feats=h,edge_attrs=graph['angular'],edge_feats=graph['radial'],
                edge_index=graph['edge'],cutoff=graph['cutoff'],first_layer=k==0)
            h=normalize_atom_features(product(m,sc=skip,node_attrs=attrs))*w[:,None]
        pool=h.new_zeros(b,c).index_add(0,g,h[:,:c]*w[:,None])/self.n_ref
        focal=torch.cat((h[graph['centers'],:c],pool),1)
        attention=torch.softmax(torch.einsum('bc,bsc->bs',self.final_query(focal),tokens)/c**.5,-1)
        message=torch.einsum('bs,bsc->bc',attention,tokens)
        return focal+.1*torch.tanh(self.final_fusion(message))


class Model(BaseModel):
    def __init__(self,encoder_config,temperatures,radii,early):
        super().__init__(encoder_config,False,temperatures)
        self.encoder=ContextEncoder(radii,early,**encoder_config)

    @property
    def input_radius(self):
        # Common 16-A candidate support in every control. Local/12-A tokens have
        # exactly zero contribution from points outside their largest radius.
        return 16.

    def make_bank(self,arrays,device):
        return ContextBank(arrays,self.encoder,device)

    def inference_arrays(self,patches):
        core=[x[np.linalg.norm(x,axis=1)<self.encoder.radius] for x in patches]
        return dict(graph_arrays(core,self.encoder.cutoff),context=region_tokens(patches,self.encoder.radii))
