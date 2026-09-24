"""Native invariant export of order-regularized neighborhood JEPA checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from .structural_adapter import StructuralMACEAnalysis, _training_contraction_order, snapshot_batch
from src.training_methods.neighborhood_jepa.regularization.model import Encoder
from src.data.structural_pretraining.batches import collate
from src.data.structural_pretraining.support import REFERENCE_RADIUS, SUPPORT, support_weights
from src.experiment_runner.registry import sha256
from src.experiment_runner.artifacts import write_json

PROTOCOL = 'neighborhood_jepa_regularization_static_v1'
RELAXED_PROTOCOL = 'paired_relaxed_snapshot_static_v1'


class NeighborhoodAnalysis(StructuralMACEAnalysis):
    # Compiled BF16 fusion changes this encoder across graph batch sizes.
    # Eager execution retains the native operations and passes singleton replay.
    execution = "eager"

    def __init__(self, cfg):
        torch.nn.Module.__init__(self)
        if cfg.protocol not in (PROTOCOL, RELAXED_PROTOCOL):
            raise ValueError(f'Unsupported neighborhood static protocol: {cfg.protocol}')
        _training_contraction_order()
        self.protocol = cfg.protocol
        self.precision = cfg.structural_precision
        self.scales = dict(cfg.structural_scales)
        self.encoder = Encoder(cfg.encoder_channels, cfg.export_norm)
        self._compiled = False

    def encode(self, batch):
        # The native packed output is invariant128 + typed angular120.
        with torch.autocast(batch["positions"].device.type, dtype=torch.bfloat16,
                            enabled=self.precision == "bf16"):
            return self.encoder(batch).float()[:, :128]


def export(config):
    source = Path(config['checkpoint'])
    if sha256(source) != config['checkpoint_sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {source}')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    training_protocol = saved['manifest']['protocol']
    if training_protocol not in ('neighborhood_jepa_regularization_order_v3', 'paired_relaxed_input_target_mace_v1'):
        raise ValueError('Expected the order-regularized neighborhood JEPA producer')
    if saved['spec']['name'] != config['variant']:
        raise ValueError('Checkpoint treatment differs from requested variant')
    data = OmegaConf.load(config['data_config'])
    protocol = RELAXED_PROTOCOL if training_protocol == 'paired_relaxed_input_target_mace_v1' else PROTOCOL
    if protocol == RELAXED_PROTOCOL and data.structural_encoder.candidate_neighbors != 80:
        raise ValueError('Relaxed snapshot recipe must specify nearest80 candidates')
    cfg = OmegaConf.create(dict(model_type='neighborhood_jepa_encoder', protocol=protocol,
        representation_source='encoder', structural_scales=config['scales'], batch_size=128,
        num_workers=2, max_samples=0, split_seed=123,
        structural_precision=saved['manifest']['config']['precision'],
        encoder_channels=saved['manifest']['config']['encoder_channels'],
        export_norm=saved['spec']['export_norm'], observation_support=SUPPORT,
        data=OmegaConf.to_container(data, resolve=True)))
    output = Path(config['output'])/'technical/encoder'
    output.mkdir(parents=True, exist_ok=False)
    (output/'.hydra').mkdir()
    state = {k: v for k, v in saved['model'].items() if k.startswith('encoder.')}
    torch.save(dict(state_dict=state), output/'encoder.ckpt')
    OmegaConf.save(cfg, output/'.hydra/config.yaml')
    write_json(output/'provenance.json', dict(source=str(source), source_sha256=sha256(source),
        step=saved['step'], selection=saved['manifest']['selection'], spec=saved['spec'],
        representation=f"Native invariant128, export_norm={saved['spec']['export_norm']}; no predictor."))


@torch.no_grad()
def verify(config):
    from src.utils.model_utils import load_model_from_checkpoint
    from src.training_methods.shared_pretraining.compilation import compile_encoder
    output = Path(config['output'])/'technical'
    cfg = OmegaConf.load(output/'encoder/.hydra/config.yaml')
    model = load_model_from_checkpoint(output/'encoder/encoder.ckpt', cfg, device='cuda:0', module=NeighborhoodAnalysis)
    saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    native = Encoder(cfg.encoder_channels, cfg.export_norm).cuda().eval()
    native.load_state_dict({k.removeprefix('encoder.'): v for k,v in saved['model'].items() if k.startswith('encoder.')}, strict=True)
    for k, v in native.state_dict().items():
        torch.testing.assert_close(v, model.encoder.state_dict()[k], rtol=0, atol=0)
    candidate_neighbors = cfg.data.structural_encoder.get('candidate_neighbors')
    points = np.load(config['verification_frame'])
    tree = cKDTree(points)
    radius = 8*config['scales']['Al']/REFERENCE_RADIUS
    ids = np.flatnonzero(np.minimum(points-tree.mins,tree.maxes-points).min(1)>radius)
    ids = ids[np.linspace(0,len(ids)-1,8,dtype=int)]
    batch = snapshot_batch(points, tree, points[ids], scale=config['scales']['Al'], material='Al', architecture='mace', candidate_neighbors=candidate_neighbors)
    # Independently prepare native observation dictionaries and use training collate.
    observations=[]
    for atom in ids:
        if candidate_neighbors is None:
            rows=np.asarray(tree.query_ball_point(points[atom],radius,return_sorted=True))
            x=(points[rows].astype(np.float64)-points[atom].astype(np.float64)).astype(np.float32)*(REFERENCE_RADIUS/config['scales']['Al'])
            keep=np.linalg.norm(x,axis=1)<8
        else:
            _,rows=tree.query(points[atom],k=80)
            physical=(points[rows].astype(np.float64)-points[atom].astype(np.float64)).astype(np.float32)
            keep=np.linalg.norm(physical,axis=1)*(REFERENCE_RADIUS/config['scales']['Al'])<8
            x=physical*(REFERENCE_RADIUS/config['scales']['Al'])
        rows=rows[keep]; x=x[keep]
        pairs=cKDTree(x).query_pairs(5.,output_type='ndarray')
        observations.append(dict(positions=x[None],weights=support_weights(x)[None],center=int(np.flatnonzero(rows==atom).item()),times=np.array([0.],np.float32),species=1,log_scale=np.log(config['scales']['Al']/REFERENCE_RADIUS),physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False,edges=np.concatenate((pairs,pairs[:,::-1]),axis=0).T))
    native_batch=collate(observations,'mace')
    for k,v in batch.items():
        torch.testing.assert_close(v,native_batch[k],rtol=0,atol=0)
    batch={k:v.cuda() for k,v in batch.items()}
    native_batch={k:v.cuda() for k,v in native_batch.items()}
    torch.set_float32_matmul_precision("highest")
    with torch.autocast('cuda',dtype=torch.bfloat16): reference=native(native_batch).float()[:,:128]
    actual=model.encode(batch)
    tolerance = config.get('native_output_tolerance', dict(rtol=2e-5, atol=2e-6))
    torch.testing.assert_close(actual,reference,**tolerance)
    reverse=snapshot_batch(points,tree,points[ids[::-1]],scale=config['scales']['Al'],material='Al',architecture='mace', candidate_neighbors=candidate_neighbors)
    replay=model.encode({k:v.cuda() for k,v in reverse.items()}).flip(0)
    if cfg.data.structural_encoder.get('enforce_batch_replay', True):
        torch.testing.assert_close(actual,replay,rtol=2e-5,atol=2e-6)
    singleton=[]
    for atom in ids:
        one=snapshot_batch(points,tree,points[[atom]],scale=config['scales']['Al'],material='Al',architecture='mace', candidate_neighbors=candidate_neighbors)
        singleton.append(model.encode({k:v.cuda() for k,v in one.items()}))
    singleton=torch.cat(singleton)
    if cfg.data.structural_encoder.get('enforce_batch_replay', True):
        torch.testing.assert_close(actual,singleton,rtol=2e-5,atol=2e-6)
    if not torch.isfinite(actual).all(): raise FloatingPointError('Nonfinite native invariant embedding')
    write_json(output/'static-verification.json',dict(state='complete',step=saved['step'],input_tensors_exact=len(batch),native_output_tolerance=tolerance,execution="eager",singleton_max_error=float((actual-singleton).abs().max()),native_eager_max_error=float((actual-reference).abs().max()),reorder_max_error=float((actual-replay).abs().max()),representation='invariant128'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--stage',choices=('export','verify'),required=True)
    args=parser.parse_args(); {'export':export,'verify':verify}[args.stage](json.loads(Path(args.config).read_text()))
