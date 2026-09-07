"""Encoder-only checkpoint bridge to the standard post-training pipeline."""
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch import nn

from src.models.encoders.atomic_graph import DensityMLPEncoder


class DensityEncoderAnalysis(nn.Module):
    """Expose the frozen raw density encoder through the pipeline's triple output."""
    def __init__(self,cfg):
        super().__init__()
        if cfg.data.normalize:
            raise ValueError('Density encoder analysis requires physical Angstrom offsets: data.normalize=false')
        self.encoder=DensityMLPEncoder()

    def forward(self,points):
        # Static sampling preserves the center but need not place it first.
        order=points.square().sum(-1).argsort(1)
        points=points.gather(1,order[:,:,None].expand(-1,-1,3))
        material=torch.zeros(len(points),dtype=torch.long,device=points.device)
        z=self.encoder(points,material,None,None)
        return z,None,None


def export_encoder(source: Path,destination: Path,data_config: Path):
    """Export encoder weights and their learned density scaler; no output/projector heads."""
    payload=torch.load(source,map_location='cpu',weights_only=False)
    if payload['hypothesis']['name']!='density_predictive':
        raise ValueError(f'Expected predictive density checkpoint, got {payload["hypothesis"]}')
    data=OmegaConf.load(data_config)
    cfg=OmegaConf.create(dict(model_type='density_encoder',representation_source='encoder',
        batch_size=8192,num_workers=4,max_samples=0,split_seed=123,data=OmegaConf.to_container(data,resolve=True)))
    module=DensityEncoderAnalysis(cfg)
    prefix='representation.encoder.'
    module.encoder.load_state_dict({k[len(prefix):]:v for k,v in payload['model'].items() if k.startswith(prefix)},strict=True)
    destination.parent.mkdir(parents=True,exist_ok=False)
    (destination.parent/'.hydra').mkdir()
    OmegaConf.save(cfg,destination.parent/'.hydra/config.yaml')
    provenance=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        representation='DensityMLPEncoder raw 128D output; no representation.output Linear/LayerNorm, forecast head, motion head or EMA teacher',
        training=payload['training'])
    torch.save(dict(state_dict=module.state_dict(),provenance=provenance),destination)
    (destination.parent/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
