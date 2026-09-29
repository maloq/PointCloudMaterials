"""Checkpoint-faithful native MACE export for the standard static gallery."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import torch

from src.experiment_runner.artifacts import analysis_artifacts, write_json
from src.experiment_runner.result_records import file_hash
from src.project_runtime.paths import REPO, resolve_path
from src.research.supervised_onset.model import CapacityEncoder
from src.research.encoder_context.geometry import graph
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.utils.model_utils import load_model_from_checkpoint
from .structural_adapter import StructuralSnapshotAnalysis

PROTOCOL = 'native_capacity_mace_static_v1'
PRODUCERS = ('src/models/encoders/spatial_mace.py', 'src/models/encoders/mace_backend.py',
             'src/research/supervised_onset/model.py', 'src/research/encoder_context/geometry.py',
             'src/training_methods/bcr/data.py')


class NativeMACEAnalysis(StructuralSnapshotAnalysis):
    architecture = 'native_mace'
    precision = 'float32'
    execution = 'compiled'
    representation = 'Native 128-D trained projection plus residual of normalized center/pooled scalars; no prediction head.'
    batch_replay_tolerance = dict(rtol=2e-4, atol=2e-4)

    def __init__(self, cfg):
        super().__init__()
        if cfg.protocol != PROTOCOL:
            raise ValueError(f'Unknown native MACE export protocol: {cfg.protocol}')
        self.protocol = cfg.protocol
        self.encoder = CapacityEncoder(**OmegaConf.to_container(cfg.native_encoder, resolve=True))
        self.normalization = OmegaConf.to_container(cfg.coordinate_normalization, resolve=True)
        self._compiled = False

    def encode_positions(self, positions):
        native = graph(positions, self.encoder)
        if not self._compiled:
            compile_spatial_encoder(self.encoder, native)
            self._compiled = True
        return self.encoder(native)


def patches(points, tree, centers, factor):
    distance, atoms = tree.query(centers)
    if np.any(distance != 0):
        raise ValueError('Native MACE requires exact focal source atoms')
    _, neighbors = tree.query(centers, k=80, workers=1)
    if not np.array_equal(neighbors[:, 0], atoms):
        raise ValueError('Duplicate source positions or invalid focal-atom order')
    return ((points[neighbors].astype(np.float64)-points[atoms, None].astype(np.float64))
            .astype(np.float32)*np.float32(factor))


@torch.no_grad()
def encode_frame(model, points, centers, settings, *, progress=None):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected native float32 source coordinates, got {points.dtype}, {points.shape}')
    torch.set_float32_matmul_precision('highest')
    material = settings['material']
    norm = model.normalization
    factor = norm['reference_scale_A']/norm['scales_A'][material]
    radius = model.encoder.radius/factor
    tree = cKDTree(points)
    margin = float(np.minimum(centers-tree.mins, tree.maxes-centers).min())
    if margin <= radius:
        raise ValueError(f'Interior margin {margin} does not support native radius {radius}')
    chunk = int(settings['batch_size']); device = next(model.parameters()).device
    result = np.empty((len(centers), 128), np.float32); started = time.monotonic()
    for start in range(0,len(centers),chunk):
        x = patches(points,tree,centers[start:start+chunk],factor)
        padded = np.full((chunk,80,3),100.,np.float32); padded[:len(x)] = x
        z = model.encode_positions(torch.as_tensor(padded,device=device))[:len(x)]
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite native MACE export at center {start}')
        result[start:start+len(x)] = z.cpu().numpy()
        if progress and (start == 0 or (start//chunk+1)%100 == 0):
            progress(centers_done=start+len(x),centers_total=len(centers),elapsed_seconds=time.monotonic()-started)
    return result, dict(centers=len(centers),material=material,coordinate_scale=factor,
        support_radius_A=radius,candidate_atoms=80,edge_cutoff_normalized=model.encoder.cutoff,
        minimum_boundary_margin_A=margin,elapsed_seconds=time.monotonic()-started,
        encoder_inputs=['center-relative normalized coordinates','constant atom channel','focal-atom indicator'],
        predictor_inputs=[],history=False,motion=False,explicit_conditions=[],
        observation='full-cell-relaxed static snapshot; no per-patch relaxation')


def source(config):
    path = resolve_path(config['checkpoint'])
    if file_hash(path) != config['checkpoint_sha256']:
        raise ValueError(f'Native checkpoint changed: {path}')
    producer = resolve_path(config['producer'])
    for name in PRODUCERS:
        if file_hash(REPO/name) != file_hash(producer/name):
            raise ValueError(f'Native inference producer differs from frozen training code: {name}')
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if config.get('checkpoint_kind', 'self_supervised') == 'distance_encoder':
        training = saved['config']
        completion = json.loads((path.parent/'complete.json').read_text())
        if (training['protocol'] != 'joint_mace_distance_early_v1'
                or saved['epoch'] != training['training']['epochs']
                or file_hash(path) != completion['best_sha256']):
            raise ValueError('Require the completed likelihood-selected CD-MACE128 checkpoint')
        for key, value in saved['encoder'].items():
            torch.testing.assert_close(value, saved['model']['encoder.'+key], rtol=0, atol=0)
        saved = dict(saved, coordinate_normalization=training['structural_dataset']['normalization'],
            step=saved['update'], selection='Minimum distance + early-CDF predictive likelihood after epoch 12',
            method=training['protocol'], structural_identity=training['structural_dataset']['identity'])
    elif saved['method'] not in ('vicreg','epi_variance'):
        raise ValueError('Require a declared native self-supervised or distance-likelihood encoder')
    if saved['encoder_config']['code_dim'] != 128:
        raise ValueError('Require the native 128-dimensional export')
    return saved


def export(config):
    saved = source(config)
    folder = resolve_path(config['output'])/'technical/encoder'
    folder.mkdir(parents=True,exist_ok=False); (folder/'.hydra').mkdir()
    data = OmegaConf.load(resolve_path(config['data_config']))
    cfg = OmegaConf.create(dict(model_type='native_mace_encoder',protocol=PROTOCOL,
        representation_source='encoder',native_encoder=saved['encoder_config'],
        coordinate_normalization=saved['coordinate_normalization'],batch_size=256,
        num_workers=2,max_samples=0,split_seed=123,experiment_name=config.get('name','mace-epi-epoch12'),
        seed_everything=123,data=OmegaConf.to_container(data,resolve=True)))
    torch.save(dict(state_dict={'encoder.'+k:v for k,v in saved['encoder'].items()}),folder/'encoder.ckpt')
    OmegaConf.save(cfg,folder/'.hydra/config.yaml')
    write_json(folder/'provenance.json',dict(source=config['checkpoint'],source_sha256=config['checkpoint_sha256'],
        exported_sha256=file_hash(folder/'encoder.ckpt'),epoch=saved['epoch'],step=saved['step'],
        selection=saved['selection'],method=saved['method'],identity=saved['identity'],
        structural_identity=saved['structural_identity'],representation=NativeMACEAnalysis.representation,
        producer_hashes={p:file_hash(REPO/p) for p in PRODUCERS},
        coordinate_normalization=saved['coordinate_normalization'],
        inference_precision='compiled float32, highest matmul precision; matches native frozen-encoder local evaluation',
        training_precision=saved.get('config',{}).get('runtime',{}).get('precision','float32')))
    evidence = analysis_artifacts(resolve_path(config['output']))/'encoder-provenance.json'
    write_json(evidence,json.loads((folder/'provenance.json').read_text()))


@torch.no_grad()
def verify(config):
    saved = source(config); root = resolve_path(config['output'])/'technical'
    device = config['device']; torch.cuda.set_device(device); torch.set_num_threads(4)
    cfg = OmegaConf.load(root/'encoder/.hydra/config.yaml')
    model = load_model_from_checkpoint(root/'encoder/encoder.ckpt',cfg,device=device,module=NativeMACEAnalysis)
    for key,value in model.encoder.state_dict().items():
        torch.testing.assert_close(value.cpu(),saved['encoder'][key],rtol=0,atol=0)
    reference = resolve_path(config['verification_reference'])
    manifest = json.loads((reference/'manifest.json').read_text())
    record = manifest['frames'][0]; path = resolve_path(record['file'])
    if file_hash(path) != record['input_sha256']:
        raise ValueError(f'Static verification source changed: {path}')
    with np.load(reference/'frame-00.npz') as values: centers = values['coords'][:16]
    points = np.load(path); direct = patches(points,cKDTree(points),centers,1.)
    with np.load(resolve_path(config['verification_inputs'])/'frame-00.npz') as values:
        expected_positions = values['nearest80'][:16]
    from src.research.encoder_quality.common import static_coordinate_agreement
    coordinate_check = static_coordinate_agreement(direct,expected_positions)
    native = CapacityEncoder(**saved['encoder_config']).to(device)
    native.load_state_dict(saved['encoder'],strict=True); native.eval().requires_grad_(False)
    from src.research.encoder_quality.run import encode
    expected = encode(native,expected_positions,256,device,compile_first=True)
    actual,_ = encode_frame(model,points,centers,dict(material='Al',batch_size=256))
    repeated,_ = encode_frame(model,points,centers[::-1],dict(material='Al',batch_size=1))
    np.testing.assert_allclose(actual,expected,**model.batch_replay_tolerance)
    np.testing.assert_allclose(actual,repeated[::-1],**model.batch_replay_tolerance)
    receipt = dict(state='verified',checkpoint_sha256=config['checkpoint_sha256'],
        coordinate_check=coordinate_check,native_max_absolute_error=float(abs(actual-expected).max()),
        batch_replay_max_absolute_error=float(abs(actual-repeated[::-1]).max()),
        tolerance=model.batch_replay_tolerance,centers=len(centers),device=device)
    write_json(root/'encoder/verification.json',receipt); print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=('export','verify'),required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    {'export':export,'verify':verify}[args.stage](config)
