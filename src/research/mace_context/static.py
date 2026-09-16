"""Export and verify the selected joint checkpoint for standard static analysis."""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

from src.analysis.mace_context_adapter import MACEContextAnalysis, encode_frame
from src.experiment_runner.artifacts import write_json
from src.experiment_runner.registry import sha256
from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path


def export(config):
    source = Path(config['checkpoint'])
    if sha256(source) != config['checkpoint_sha256']:
        raise ValueError(f'Selected joint checkpoint checksum mismatch: {source}')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if (saved['protocol'], saved['variant'], saved['epoch']) != ('mace_context_recovery_joint_v1', 'dual_physics', 12):
        raise ValueError('Static export requires the selected dual_physics epoch-12 joint checkpoint')
    directory, name = resolve_config_path(config['initial_checkpoint'])
    cfg = OmegaConf.load(Path(directory)/f'{name}.yaml')
    with open_dict(cfg):
        cfg.model_type = 'mace_context_encoder'
        cfg.protocol = 'mace_context_recovery_static_v1'
        cfg.pretrained_checkpoint = cfg.encoder.kwargs.pretrained_checkpoint
        cfg.performance = cfg.encoder.kwargs.performance
        cfg.performance.compile_radial_mlp = False
        cfg.representation_source = 'encoder'
        cfg.context_checkpoint_sha256 = config['checkpoint_sha256']
    weights = {key.removeprefix('encoder.'): value for key, value in saved['model_state'].items()
               if key.startswith('encoder.mace.')}
    if not weights:
        raise ValueError('Joint checkpoint contains no encoder.mace state')
    output = Path(config['output'])/'technical/encoder'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'encoder.ckpt').exists():
        raise FileExistsError(f'Preserve existing export: {output}')
    (output/'.hydra').mkdir()
    torch.save(dict(state_dict=weights), output/'encoder.ckpt')
    OmegaConf.save(cfg, output/'.hydra/config.yaml')
    write_json(output/'provenance.json', dict(source=str(source), source_sha256=sha256(source),
        epoch=saved['epoch'], variant=saved['variant'], exported_sha256=sha256(output/'encoder.ckpt'),
        representation='512 raw invariant channels: smooth-inner 256 + tracked-center 256',
        training='Exact jointly trained MACE parameters; projector and supervised heads omitted from inference.',
        state_keys=list(weights)))


@torch.inference_mode()
def verify(config):
    torch.cuda.set_device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    root = Path(config['output'])/'technical'
    checkpoint = root/'encoder/encoder.ckpt'
    cfg = OmegaConf.load(root/'encoder/.hydra/config.yaml')
    model = load_model_from_checkpoint(checkpoint, cfg, device=config['device'], module=MACEContextAnalysis)
    original = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)['model_state']
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), original['encoder.'+key], rtol=0, atol=0)
    records = {}
    for material, case in config['verification'].items():
        from scipy.spatial import cKDTree
        points = np.load(case['source'])
        tree = cKDTree(points)
        # Local centers exercise overlapping readouts and spatially separated ones.
        locations = (points.min(0)+points.max(0))/2 + np.array([[0,0,0], [4,0,0], [0,4,0], [0,0,4], [40,0,0], [-40,0,0]])
        centers = points[tree.query(locations)[1]]
        analysis = OmegaConf.load(case['analysis_config'])
        data = OmegaConf.load(analysis.inputs.data_config)
        settings = OmegaConf.to_container(data.context_encoder, resolve=True)
        z, record = encode_frame(model, points, centers, settings)
        settings = dict(settings, node_batch_size=137)
        zz, other = encode_frame(model, points, centers[::-1], settings)
        error = float(np.square(z.astype(np.float64)-zz[::-1]).sum()/np.square(z.astype(np.float64)).sum())
        if error > 1e-8:
            raise AssertionError(f'{material}: reordered centers / changed node batch disagree: {error}')
        # Measure realistic throughput on an interior subvolume, after CUDA warmup.
        center = (points.min(0)+points.max(0))/2
        selected = np.flatnonzero(np.max(np.abs(points-center), axis=1) < 20)
        _, timing = encode_frame(model, points, points[selected[::10]],
                                OmegaConf.to_container(data.context_encoder, resolve=True))
        records[material] = dict(**record, reordered_batch_relative_squared_error=error,
                                 benchmark=timing, verification_centers=centers.tolist())
        write_json(root/'static-verification.json', dict(state='running', results=records))
        print(f'[static verify] {material}: {records[material]}', flush=True)
    write_json(root/'static-verification.json', dict(state='complete', results=records,
        checkpoint_state='Every exported tensor matches joint model_state exactly'))
