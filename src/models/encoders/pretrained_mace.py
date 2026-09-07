"""MLIP-initialized MACE with an exact central-atom receptive-field readout."""
import torch
from torch import nn
from mace.cli.convert_e3nn_cueq import run as convert_cueq


class PretrainedMACEEncoder(nn.Module):
    """MACE-MP-0b2 small, 256 scalar channels from its two trained layers.

    The first interaction retains all incoming edges of atoms within 5 A of
    the center. Only those atoms need the first product. The second interaction
    retains incoming edges to the center; only the center needs its product.
    Without outer_radius_A this equals the full two-layer MACE descriptor.
    With outer_radius_A only the finite, smoothly windowed context contributes;
    it deliberately changes the second-layer descriptor for compact inputs.
    No energy, force, projector, or prediction head forms part of this output.
    """
    def __init__(self,checkpoint,accelerated=True,outer_radius_A=None):
        super().__init__()
        backbone=torch.load(checkpoint,map_location='cpu',weights_only=False).float()
        assert float(backbone.r_max)==5. and int(backbone.num_interactions)==2
        assert str(backbone.products[0].linear.irreps_out)=='128x0e'
        self.backbone=convert_cueq(backbone,device='cuda',layout='ir_mul') if accelerated else backbone
        self.backbone.readouts.requires_grad_(False)
        self.outer_radius_A=outer_radius_A
        if outer_radius_A is not None and outer_radius_A<=5.:
            raise ValueError("The compact context radius must exceed the pretrained 5 A interaction cutoff")
        self.register_buffer('element_indices',torch.tensor([(self.backbone.atomic_numbers==a).nonzero().item() for a in (13,12,73)],dtype=torch.long))
        self.register_buffer('feature_mean',torch.zeros(256))
        self.register_buffer('feature_std',torch.ones(256))

    def raw_features(self,x,material):
        b,n,_=x.shape;device=x.device;model=self.backbone
        # Center-first physical offsets: either complete two-hop support, or
        # the explicitly configured smooth finite context used for compact inputs.
        inside=x.square().sum(-1)<25.
        inner=inside.flatten().nonzero().flatten()
        centers=torch.arange(b,device=device)*n
        remap=torch.full((b*n,),-1,device=device,dtype=torch.long)
        remap[inner]=torch.arange(len(inner),device=device)
        d=torch.cdist(x,x)
        valid=(d<5.)&~torch.eye(n,device=device,dtype=torch.bool)[None]&inside[:,None,:]
        if self.outer_radius_A is not None:
            valid=valid&(x.square().sum(-1)<self.outer_radius_A**2)[:,:,None]
        ids=valid.nonzero()
        edges=(ids[:,1:]+ids[:,0,None]*n).T.contiguous()
        positions=x.reshape(-1,3)
        attrs=torch.nn.functional.one_hot(self.element_indices[material],len(model.atomic_numbers)).to(x.dtype).repeat_interleave(n,0)
        vectors=positions[edges[1]]-positions[edges[0]]
        lengths=vectors.norm(dim=-1,keepdim=True)
        angular=model.spherical_harmonics(vectors)
        radial,cutoff=model.radial_embedding(lengths,attrs,edges,model.atomic_numbers)
        if self.outer_radius_A is not None:
            # The direct 5 A neighborhood is unchanged. Fade sources in the
            # outer context smoothly to zero before the 80-neighbor boundary.
            radius=positions[edges[0]].norm(dim=-1)
            u=((radius-5.)/(self.outer_radius_A-5.)).clamp(0,1)
            taper=(1-u**3*(10-u*(15-6*u)))[:,None]
            cutoff=taper if cutoff is None else cutoff*taper
        h=model.node_embedding(attrs)
        h,sc=model.interactions[0](node_attrs=attrs,node_feats=h,edge_attrs=angular,edge_feats=radial,edge_index=edges,cutoff=cutoff,first_layer=True)
        h=model.products[0](node_feats=h[inner],sc=None if sc is None else sc[inner],node_attrs=attrs[inner])
        first=h[remap[centers]]
        center_edges=(edges[1]%n)==0
        h,sc=model.interactions[1](node_attrs=attrs[inner],node_feats=h,edge_attrs=angular[center_edges],edge_feats=radial[center_edges],edge_index=remap[edges[:,center_edges]],cutoff=None if cutoff is None else cutoff[center_edges],first_layer=False)
        center_indices=remap[centers]
        last=model.products[1](node_feats=h[center_indices],sc=None if sc is None else sc[center_indices],node_attrs=attrs[centers])
        return torch.cat((first,last),1)

    def forward(self,x,material):
        return (self.raw_features(x,material)-self.feature_mean)/self.feature_std
