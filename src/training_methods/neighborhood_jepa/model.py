"""Atom-level high-order features, invariant export, and equivariant query decoding."""
import torch
from torch import nn
from e3nn import o3
from src.models.encoders.structural import StructuralMACE

ELLS=(1,2,4,6)
CHANNELS=4
EQ_DIM=CHANNELS*sum(2*l+1 for l in ELLS)
IRREPS=o3.Irreps(' + '.join(f'{CHANNELS}x{l}{"e" if l%2==0 else "o"}' for l in ELLS))


def split_tensors(e):
    return [x.reshape(*e.shape[:-1],CHANNELS,2*l+1) for x,l in zip(e.split([CHANNELS*(2*l+1) for l in ELLS],-1),ELLS)]


def invariants(e):
    # Channel Gram matrices retain alignment information; rotation-invariant.
    return torch.cat([(x@x.transpose(-1,-2)/(2*l+1)).flatten(-2) for x,l in zip(split_tensors(e),ELLS)],-1)


class NeighborhoodEncoder(nn.Module):
    """One independently evaluated local snapshot; no teacher or history inside E."""
    def __init__(self,architecture,channels=32):
        super().__init__();self.architecture=architecture
        if architecture!='mace':raise ValueError('This protocol trains MACE only')
        self.base=StructuralMACE(channels=channels,readout_hidden=104*channels//32)
        width=channels
        self.angular_weights=nn.Sequential(nn.Linear(width+1,2*channels),nn.SiLU(),nn.Linear(2*channels,len(ELLS)*CHANNELS))
        self.harmonics=o3.SphericalHarmonics(list(ELLS),normalize=True,normalization='component')
        self.compress=nn.Sequential(nn.Linear(128+len(ELLS)*CHANNELS**2,6*channels),nn.LayerNorm(6*channels),nn.SiLU(),nn.Linear(6*channels,128))
        self.output_norm=nn.LayerNorm(128,elementwise_affine=False)

    def forward(self,batch):
        base=self.base
        atoms=base.atom_features(batch)
        x,g,w=atoms['positions'],atoms['graph'],atoms['weights']
        scalar,count=atoms['scalars'],atoms['count']
        z=base.output_norm(base.pool(atoms['features'],x,g,count))
        # Tensor creation, contractions and export are FP32. Harmonics precede pooling.
        with torch.autocast(x.device.type,enabled=False):
            radius=x.norm(dim=-1);weights=w.float()*(radius>1e-6)
            coeff=self.angular_weights(torch.cat((scalar.float(),(radius/8)[:,None]),-1)).reshape(-1,len(ELLS),CHANNELS)
            harmonics=self.harmonics(x).float().split([2*l+1 for l in ELLS],-1)
            denom=x.new_zeros(count).index_add(0,g,weights).clamp_min(1e-8)
            tensors=[]
            for k,y in enumerate(harmonics):
                atom=(coeff[:,k,:,None]*y[:,None]*weights[:,None,None]).flatten(1)
                pooled=x.new_zeros(count,atom.shape[-1]).index_add(0,g,atom)/denom[:,None]
                tensors.append(pooled)
            eq=torch.cat(tensors,-1)
            inv=self.output_norm(z.float()+self.compress(torch.cat((z.float(),invariants(eq)),-1)))
        return torch.cat((inv,eq),-1)

    def export(self,batch):
        encoded=self(batch)
        return {'invariant':encoded[:,:128],'equivariant':encoded[:,128:]}


class NeighborhoodModel(nn.Module):
    def __init__(self,architecture,previous_context=False,groups=1):
        super().__init__();self.encoder=NeighborhoodEncoder(architecture);self.previous_context=previous_context
        # Previous invariant context cannot bypass the current exported z entirely:
        # auxiliary physical decoders always read each independent z.
        context_dim=128*(2 if previous_context else 1)+4+len(ELLS)*CHANNELS+(1 if previous_context else 0)
        self.query=nn.Sequential(nn.Linear(context_dim,192),nn.LayerNorm(192),nn.SiLU(),nn.Linear(192,192),nn.SiLU())
        self.inv_prediction=nn.Linear(192,128)
        self.tensor_mix=o3.Linear(IRREPS,IRREPS)
        self.tensor_past=o3.Linear(IRREPS,IRREPS) if previous_context else None
        self.gates=nn.Linear(192,2*len(ELLS)*CHANNELS)
        self.harmonics=o3.SphericalHarmonics(list(ELLS),normalize=True,normalization='component')
        self.domain_embedding=nn.Embedding(groups,16)
        self.physical_decoder=nn.Sequential(nn.Linear(144,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,85))
        self.tda_decoder=nn.Sequential(nn.Linear(144,192),nn.LayerNorm(192),nn.SiLU(),nn.Linear(192,144))
        self.projector=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,64))
        self.bond=o3.Linear(IRREPS,'1x4e + 1x6e')

    def physical(self,z,group):
        return self.physical_decoder(torch.cat((z,self.domain_embedding(group)),-1))

    def tda(self,z,group):
        return self.tda_decoder(torch.cat((z,self.domain_embedding(group)),-1))

    def predict(self,current,previous,position,lag,previous_lag=None):
        """Only current-relative query positions and observed previous/current states."""
        b,q,_=position.shape;radius=position.norm(dim=-1)
        time=torch.stack((lag,lag.abs(),torch.sign(lag)),dim=-1)
        parts=[current[:,:128,None].transpose(1,2).expand(b,q,128)]
        if self.previous_context:
            if previous_lag is None:raise ValueError('Previous context requires its physical time offset')
            parts.extend((previous[:,:128,None].transpose(1,2).expand(b,q,128),previous_lag[:,None,None].expand(b,q,1)))
        angular=self.harmonics(position).split([2*l+1 for l in ELLS],-1)
        align=torch.cat([(v[:,None]*y[:,:,None]).mean(-1) for v,y in zip(split_tensors(current[:,128:]),angular)],-1)
        h=self.query(torch.cat((*parts,(radius/8)[...,None],time,align),-1))
        inv=self.inv_prediction(h);gates=self.gates(h).reshape(b,q,len(ELLS),2,CHANNELS)
        eq=self.tensor_mix(current[:,128:])
        if self.previous_context:eq=eq+self.tensor_past(previous[:,128:])
        angular=self.harmonics(position).split([2*l+1 for l in ELLS],-1);values=[]
        for k,(v,y) in enumerate(zip(split_tensors(eq),angular)):
            radial_mask=(radius>1e-6)[...,None,None]
            value=gates[:,:,k,0,:,None]*v[:,None]+gates[:,:,k,1,:,None]*y[:,:,None]*radial_mask
            values.append(value.flatten(-2))
        return inv,torch.cat(values,-1)
