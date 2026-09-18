"""Native structural MACE/GATr snapshots in the standard static-analysis workflow."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import torch
from torch import nn

from src.data.predictive_memory.targets import taper
from src.data.static import PointCloudDataset
from src.data.structural_pretraining.prepare import ELEMENTS, REFERENCE_RADIUS, offsets
from src.experiment_runner.artifacts import write_json
from src.experiment_runner.registry import sha256
from src.models.encoders.structural import ATOMIC_NUMBERS, ARCHITECTURE_REVISION, StructuralGATr, StructuralMACE
from src.utils.model_utils import load_model_from_checkpoint


PROTOCOL = 'structural_gatr_snapshot_static_v1'
SHARED_PROTOCOL = 'structural_gatr_snapshot_static_v6'
MACE_PROTOCOL = 'structural_mace_snapshot_static_v6'


def _training_contraction_order():
    # shared_pretraining.compilation applies this policy to both backbones.
    from gatr.utils.einsum import enable_cached_einsum
    import torch.backends.opt_einsum
    enable_cached_einsum(False)
    torch.backends.opt_einsum.enabled = False


class StructuralSnapshotAnalysis(nn.Module):
    execution = 'eager'

    def forward(self, points):
        raise ValueError('Structural encoders require full source neighborhoods and tracked centers')

    def encode(self, batch):
        device = batch['positions'].device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            return self.encoder(batch).float()


class StructuralGATrAnalysis(StructuralSnapshotAnalysis):
    architecture = 'gatr'

    def __init__(self, cfg):
        super().__init__()
        if cfg.protocol not in (PROTOCOL, SHARED_PROTOCOL):
            raise ValueError(f'Unsupported structural analysis protocol: {cfg.protocol}')
        self.protocol = cfg.protocol
        self.precision = 'float32'
        if cfg.protocol == SHARED_PROTOCOL:
            if cfg.architecture_revision != ARCHITECTURE_REVISION:
                raise ValueError(f'Checkpoint architecture differs from current implementation: {cfg.architecture_revision}')
            self.precision = cfg.structural_precision
            if self.precision not in ('float32', 'bf16'):
                raise ValueError(f'Unsupported structural inference precision: {self.precision}')
            # Match the contraction order used by the trained compiled encoder.
            _training_contraction_order()
        self.encoder = StructuralGATr()
        self.scales = dict(cfg.structural_scales)


class StructuralMACEAnalysis(StructuralSnapshotAnalysis):
    architecture = 'mace'
    execution = 'compiled'

    def __init__(self, cfg):
        super().__init__()
        if cfg.protocol != MACE_PROTOCOL or cfg.architecture_revision != ARCHITECTURE_REVISION:
            raise ValueError(f'Unsupported structural MACE protocol/revision: {cfg.protocol}, {cfg.architecture_revision}')
        self.protocol = cfg.protocol
        self.precision = cfg.structural_precision
        if self.precision not in ('float32', 'bf16'):
            raise ValueError(f'Unsupported structural inference precision: {self.precision}')
        _training_contraction_order()
        self.encoder = StructuralMACE(backend=cfg.structural_backend)
        self.scales = dict(cfg.structural_scales)
        if cfg.structural_execution != self.execution:
            raise ValueError('Structural MACE analysis must preserve compiled training execution')
        self._compiled = False

    def encode(self, batch):
        if not self._compiled:
            from src.training_methods.shared_pretraining.compilation import compile_encoder
            compile_encoder(self.encoder, batch, self.precision)
            self._compiled = True
        return super().encode(batch)


def snapshot_batch(points, tree, centers, *, scale, material, architecture='gatr'):
    """Reproduce prepare.offsets and Release.observation for physical static points."""
    if architecture not in ('gatr', 'mace'):
        raise ValueError(f'Unsupported structural architecture: {architecture}')
    distance, atom_rows = tree.query(centers)
    if not np.all(distance == 0):
        raise ValueError(f'Analysis centers must be exact source atoms; max error={distance.max()}')
    factor = REFERENCE_RADIUS / scale
    radius = 17. / factor
    margin = float(np.minimum(centers-tree.mins, tree.maxes-centers).min())
    if margin <= radius:
        raise ValueError(f'Static center margin {margin:.6f} cannot support {radius:.6f} Angstrom')
    groups = tree.query_ball_point(centers, radius, return_sorted=True, workers=1)
    n = max(map(len, groups))
    x = np.zeros((len(centers), 1, n, 3), np.float32)
    w = np.zeros((len(centers), 1, n), np.float32)
    species = np.zeros((len(centers), n), np.int64)
    center_indices = []
    packed_positions, packed_weights, edges, node_graph = [], [], [], []
    packed_offset = 0
    for i, (atom, rows) in enumerate(zip(atom_rows, groups, strict=True)):
        rows = np.asarray(rows, dtype=np.int64)
        # Float64 subtraction then FP32 storage, then fixed material scaling:
        # exactly the native preparation and training observation order.
        local = offsets(points, atom, rows, None) * factor
        x[i, 0, :len(rows)] = local
        w[i, 0, :len(rows)] = taper(np.linalg.norm(local, axis=-1), 15., 17.)
        species[i, :len(rows)] = ATOMIC_NUMBERS.index(ELEMENTS[material])
        center_indices.append(int(np.flatnonzero(rows == atom).item()))
        if architecture == 'mace':
            # Match Release.observation: edges are built after fixed scaling,
            # on each complete, unpadded observation, in query_pairs order.
            pairs = cKDTree(local).query_pairs(5., output_type='ndarray')
            if len(pairs) and np.any(np.linalg.norm(local[pairs[:, 0]]-local[pairs[:, 1]], axis=-1) <= 0):
                raise ValueError('Coincident atoms in the normalized static MACE graph')
            edges.append(np.concatenate((pairs, pairs[:, ::-1]), axis=0).T + packed_offset)
            packed_positions.append(local)
            packed_weights.append(w[i, 0, :len(rows)])
            node_graph.append(np.full(len(rows), i, np.int64))
            packed_offset += len(rows)
    batch = dict(positions=torch.from_numpy(x), weights=torch.from_numpy(w),
                species=torch.from_numpy(species), centers=torch.tensor(center_indices),
                times=torch.zeros((len(centers), 1)),
                log_scale=torch.full((len(centers),), np.log(scale / REFERENCE_RADIUS)))
    if architecture == 'mace':
        batch.update(packed_positions=torch.from_numpy(np.concatenate(packed_positions)),
            packed_weights=torch.from_numpy(np.concatenate(packed_weights)),
            packed_species=torch.full((packed_offset,), ATOMIC_NUMBERS.index(ELEMENTS[material]), dtype=torch.long),
            node_graph=torch.from_numpy(np.concatenate(node_graph)),
            edges=torch.from_numpy(np.concatenate(edges, axis=1)))
    return batch


@torch.no_grad()
def encode_frame(model, points, centers, settings, *, progress=None):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected native float32 static coordinates, got {points.dtype} {points.shape}')
    # Match training FP32 arithmetic, including disabling TF32.
    torch.set_float32_matmul_precision('highest')
    material = settings['material']
    scale = model.scales[material]
    tree = cKDTree(points, balanced_tree=False)
    device = next(model.parameters()).device
    batch_size = int(settings['batch_size'])
    result = np.empty((len(centers), 128), np.float32)
    started = time.monotonic()
    for start in range(0, len(centers), batch_size):
        batch = snapshot_batch(points, tree, centers[start:start+batch_size], scale=scale, material=material,
                               architecture=model.architecture)
        z = model.encode({k: v.to(device) for k, v in batch.items()})
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite structural features at centers {start}:{start+batch_size}')
        result[start:start+len(z)] = z.cpu().numpy()
        if progress is not None and (start == 0 or (start // batch_size + 1) % 100 == 0):
            progress(centers_done=min(start+batch_size, len(centers)), centers_total=len(centers),
                     elapsed_seconds=time.monotonic()-started)
    return result, dict(centers=len(centers), material=material, material_radius_A=scale,
        coordinate_scale=REFERENCE_RADIUS/scale, support_radius_A=17*scale/REFERENCE_RADIUS,
        minimum_boundary_margin_A=float(np.minimum(centers-points.min(0), points.max(0)-centers).min()),
        elapsed_seconds=time.monotonic()-started)


def collect_structural_inference(model, dataloader, cfg, out_dir, *, max_batches, max_samples):
    dataset = dataloader.dataset
    if not isinstance(dataset, PointCloudDataset) or dataset._cache_coord_arrays is None:
        raise TypeError('Structural inference requires cached static centers and full source coordinates')
    settings = OmegaConf.to_container(cfg.data.structural_encoder, resolve=True)
    metadata = json.loads((Path(cfg.data.sample_cache.cache_dir)/'metadata.json').read_text())
    sources = {s['name']: s for s in metadata['request']['sources']}
    total = len(dataset)
    if max_batches is not None:
        total = min(total, max_batches*int(dataloader.batch_size))
    if max_samples is not None:
        total = min(total, max_samples)
    result = np.empty((total, 128), np.float32)
    coordinates = np.empty((total, 3), np.float32)
    records, offset = [], 0
    for shard, coords in zip(metadata['shards'], dataset._cache_coord_arrays, strict=True):
        if offset == total:
            break
        coords = coords[:min(len(coords), total-offset)]
        path = Path(sources[shard['source']]['root'])/shard['file']

        def progress(**fields):
            write_json(Path(out_dir)/'structural-inference-status.json', dict(state='running',
                frame=shard['file'], completed_centers=offset, total_centers=total, **fields))
            print(f'[structural] {shard["file"]}: {fields}', flush=True)

        points = np.load(path, allow_pickle=False)
        z, record = encode_frame(model, points, coords, settings, progress=progress)
        # Every frame checks padding and batch-order independence on real inputs.
        selected = np.unique(np.linspace(0, len(coords)-1, 6, dtype=int))
        replay, _ = encode_frame(model, points, coords[selected[::-1]], dict(settings, batch_size=1))
        np.testing.assert_allclose(z[selected], replay[::-1], rtol=2e-5, atol=2e-6,
                                   err_msg=f'{model.architecture} batch/padding replay disagrees: {path}')
        result[offset:offset+len(coords)] = z
        coordinates[offset:offset+len(coords)] = coords
        records.append(dict(file=str(path), source_sha256=sha256(path),
            batch_replay_max_absolute_error=float(np.max(np.abs(z[selected]-replay[::-1]))), **record))
        offset += len(coords)
        write_json(Path(out_dir)/'structural-inference-protocol.json', dict(protocol=model.protocol,
            settings=settings, frames=records, completed_centers=offset,
            representation='Raw trained 128-channel center encoder state; no projector or target heads.',
            boundaries='Full nonperiodic source neighborhoods; verified interior centers; no inferred box.',
            precision=model.precision, execution=model.execution,
            input_precision='Native float64 subtraction to float32 offsets, fixed material scaling; TF32 disabled.',
            encoder_precision='Checkpoint-native precision boundaries; geometry and exported states remain FP32.'))
    write_json(Path(out_dir)/'structural-inference-status.json', dict(state='complete', total_centers=total))
    return dict(inv_latents=result, coords=coordinates, eq_latents=np.empty(0),
                phases=np.empty(0), instance_ids=np.empty(0), anchor_frame_indices=np.empty(0))


def export(config):
    source = Path(config['checkpoint'])
    if sha256(source) != config['checkpoint_sha256']:
        raise ValueError(f'Selected structural checkpoint checksum mismatch: {source}')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    architecture = saved['architecture']
    if architecture not in ('gatr', 'mace') or (saved['input_frames'], saved['state_dim'], saved['identity']['config']['method']) != (1, 128, 'vicreg'):
        raise ValueError('This static export requires a native snapshot MACE/GATr–VICReg encoder')
    if tuple(saved['atomic_numbers']) != ATOMIC_NUMBERS:
        raise ValueError('Checkpoint species vocabulary differs from the native producer')
    training_protocol = saved['identity']['protocol']
    if training_protocol not in ('structural_neighbors_v1', 'shared_pretraining_v6'):
        raise ValueError(f'Unsupported structural training protocol: {training_protocol}')
    shared = training_protocol == 'shared_pretraining_v6'
    if architecture == 'mace' and not shared:
        raise ValueError('Static structural MACE export requires the shared v6 training protocol')
    if shared and saved['identity']['architecture_revision'] != ARCHITECTURE_REVISION:
        raise ValueError('Selected encoder architecture revision differs from the current producer')
    for filename, expected in saved['identity']['implementation']['files'].items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f'Training implementation changed; inspect before export: {filename}')
    best = torch.load(source.with_name('best.pt'), map_location='cpu', weights_only=False)
    if best['step'] != saved['step']:
        raise ValueError('Selected encoder and best checkpoint disagree on update')
    for key, value in saved['encoder'].items():
        torch.testing.assert_close(value, best['model']['encoder.'+key], rtol=0, atol=0)
    data = OmegaConf.load(config['data_config'])
    protocol = MACE_PROTOCOL if architecture == 'mace' else SHARED_PROTOCOL if shared else PROTOCOL
    cfg = OmegaConf.create(dict(model_type=f'structural_{architecture}_encoder', protocol=protocol,
        representation_source='encoder', structural_scales=saved['scales'], batch_size=128,
        num_workers=2, max_samples=0, split_seed=123, data=OmegaConf.to_container(data, resolve=True)))
    if shared:
        cfg.architecture_revision = saved['identity']['architecture_revision']
        cfg.structural_precision = saved['identity']['config']['precision']
    if architecture == 'mace':
        # StructuralModel's verified v6 producer constructs this backend.
        cfg.structural_backend = 'cueq'
        cfg.structural_execution = 'compiled'
    output = Path(config['output'])/'technical/encoder'
    output.mkdir(parents=True, exist_ok=False)
    (output/'.hydra').mkdir()
    torch.save(dict(state_dict={'encoder.'+k: v for k, v in saved['encoder'].items()}), output/'encoder.ckpt')
    OmegaConf.save(cfg, output/'.hydra/config.yaml')
    write_json(output/'provenance.json', dict(source=str(source), source_sha256=sha256(source),
        exported_sha256=sha256(output/'encoder.ckpt'), step=saved['step'], identity=saved['identity'],
        selection_score=best['best'], scales=saved['scales'], representation='Raw encoder z128'))


@torch.no_grad()
def verify(config):
    from src.data.structural_pretraining.batches import Release, collate, move
    from src.project_runtime.paths import resolve_path
    source = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    root = Path(config['output'])/'technical'
    cfg = OmegaConf.load(root/'encoder/.hydra/config.yaml')
    module = {'gatr': StructuralGATrAnalysis, 'mace': StructuralMACEAnalysis}[source['architecture']]
    model = load_model_from_checkpoint(root/'encoder/encoder.ckpt', cfg,
        device=config['device'], module=module)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    for key, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value.cpu(), source['encoder'][key], rtol=0, atol=0)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    training = source['identity']['config']
    release = Release(resolve_path(training['release']),
        materials=training['materials'] if model.protocol in (SHARED_PROTOCOL, MACE_PROTOCOL) else None)
    index = next(i for i, (_, _, r) in enumerate(release.rows) if r['static'] and r['material'] == 'Al')
    name, row, record = release.rows[index]
    observation = release.observation(index, 'anchor', False, model.architecture == 'mace')
    native = collate([observation], model.architecture)
    static_source = next(s for s in release.manifest['sources'] if s['id'] == record['source'])
    points = np.load(static_source['path'])
    center = release.arrays[name]['center_ids'][release.arrays[name]['views'][row, 2]]
    batch = snapshot_batch(points, cKDTree(points, balanced_tree=False), points[[center]],
                           scale=model.scales['Al'], material='Al', architecture=model.architecture)
    for key, value in batch.items():
        torch.testing.assert_close(value, native[key], rtol=0, atol=0)
    expected = model.encode(move(native, config['device'])).cpu().numpy()
    repeated = model.encode(move(native, config['device'])).cpu().numpy()
    # CuEq/scatter CUDA reductions are not bitwise repeatable. Retain GATr's
    # exact single-input check; MACE uses the same tight numeric tolerance as
    # compiled-state and batch replay, with repeated-native error recorded.
    native_tolerance = dict(rtol=2e-5, atol=2e-6) if model.architecture == 'mace' else dict(rtol=0, atol=0)
    np.testing.assert_allclose(repeated, expected, **native_tolerance)
    result, record = encode_frame(model, points, points[[center]], dict(material='Al', batch_size=1))
    np.testing.assert_allclose(result, expected, **native_tolerance)
    data = OmegaConf.load(config['data_config'])
    frame = np.load(Path(data.data_path)/data.data_files[0])
    location = (frame.min(0)+frame.max(0))/2
    ids = cKDTree(frame).query(location+np.array([[0,0,0], [4,0,0], [0,4,0], [0,0,4], [40,0,0], [-40,0,0]]))[1]
    settings = OmegaConf.to_container(data.structural_encoder, resolve=True)
    z, timing = encode_frame(model, frame, frame[ids], settings)
    zz, _ = encode_frame(model, frame, frame[ids[::-1]], dict(settings, batch_size=1))
    np.testing.assert_allclose(z, zz[::-1], rtol=2e-5, atol=2e-6)
    selected_replay = None
    if model.protocol in (SHARED_PROTOCOL, MACE_PROTOCOL):
        # The trainer saved these states from its compiled best-checkpoint
        # selection pass, in evaluate() order with microbatch size 64.
        cached = np.load(Path(config['checkpoint']).with_name('selection_predictions.npz'))
        states = []
        for start in range(0, len(cached['indices']), 64):
            samples = [release.observation(int(i), 'anchor', False, model.architecture == 'mace')
                       for i in cached['indices'][start:start+64]]
            states.append(model.encode(move(collate(samples, model.architecture), config['device'])).cpu().numpy())
        replay = np.concatenate(states)
        np.testing.assert_allclose(replay, cached['state'], rtol=2e-5, atol=2e-6,
            err_msg='Static adapter disagrees with saved compiled selection states')
        selected_replay = dict(rows=len(replay),
            max_absolute_error=float(np.max(np.abs(replay-cached['state']))),
            reference_sha256=sha256(Path(config['checkpoint']).with_name('selection_predictions.npz')))
    write_json(root/'static-verification.json', dict(state='complete', step=source['step'],
        exported_tensors='Exact match to selected encoder and best training checkpoint.',
        native_training_input=f'All {len(batch)} input tensors exactly match Release.observation + collate.',
        native_training_output_max_absolute_error=float(np.max(np.abs(result-expected))),
        repeated_native_output_max_absolute_error=float(np.max(np.abs(repeated-expected))),
        native_output_tolerance=native_tolerance,
        reordered_batch_max_absolute_error=float(np.max(np.abs(z-zz[::-1]))),
        native_example=record, static_example=timing, compiled_selection_replay=selected_replay,
        precision=model.precision, execution=model.execution, torch_version=torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=('export', 'verify'))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    {'export': export, 'verify': verify}[args.stage](config)


if __name__ == '__main__':
    main()
