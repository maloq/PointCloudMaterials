"""A100 extraction of frozen geometric streams, trajectories and spatial panels."""
from src.data.structural_pretraining.support import OUTER_RADIUS
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from src.data.structural_pretraining.prepare import source_arrays, chart, offsets, file_hash, save_json, REFERENCE_RADIUS
from .model import Capture, vector_parts, geometry_baselines, STAGES, SECTORS, VECTOR_INDICES


def encode_rows(model, positions, centers, batch_size):
    z, mv = [], []
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            batch = model.make_batch(positions[start:start+batch_size], centers[start:start+batch_size])
            a, b = model.evaluate(batch)
            z.append(a.cpu().numpy()); mv.append(b.cpu().numpy())
    return np.concatenate(z), np.concatenate(mv)


@torch.no_grad()
def numerical_audit(model, positions, centers, root, seed):
    batch = model.make_batch(positions[:16], centers[:16])
    z, mv = model.evaluate(batch)
    _, repeated = model.evaluate(batch)
    original = vector_parts(mv)
    trials = []
    for index, rotation in enumerate(Rotation.random(5, random_state=seed).as_matrix()):
        r = torch.tensor(rotation, device='cuda', dtype=torch.float64)
        rotated = dict(batch, positions=(batch['positions'].double()@r.T).float())
        zr, mvr = model.evaluate(rotated)
        actual = vector_parts(mvr).double()
        expected = original.double()@r.T
        error = (actual-expected).square().mean((0, 2, 4)).sqrt()
        scale = original.double().square().mean((0, 2, 4)).sqrt()
        relative = torch.where(scale > 0, error/scale, torch.zeros_like(error))
        if relative.max() > .01:
            raise ValueError(f'Rotation audit failed: {relative.cpu().numpy()}')
        trials.append(dict(rotation=index, max_z_absolute=float((zr-z).abs().max()),
            vector_relative_rms=relative.cpu().tolist(), vector_absolute_rms=error.cpu().tolist()))
    def erase_output(module, args, output):
        return torch.zeros_like(output[0]), output[1]
    handle = model.encoder.spatial[1].register_forward_hook(erase_output)
    erased_z, _ = model.evaluate(batch); handle.remove()
    torch.testing.assert_close(z, erased_z, rtol=0, atol=0)
    def erase_direction(module, args, kwargs):
        x = args[0].clone(); x[..., list(VECTOR_INDICES)] = 0
        return (x, *args[1:]), kwargs
    handle = model.encoder.spatial[1].mlp.register_forward_pre_hook(erase_direction, with_kwargs=True)
    ablated_z, _ = model.evaluate(batch); handle.remove()
    save_json(root/'numerical-audit.json', dict(rotations=trials,
        repeated_multivector_max_abs=float((mv-repeated).abs().max()),
        final_mv_erasure_z_max_abs=float((z-erased_z).abs().max()),
        pre_mlp_direction_erasure_z_rms=float((z.double()-ablated_z.double()).square().mean().sqrt()),
        description='Read-only hooks. Last returned multivector is discarded; pre-MLP erasure is an out-of-distribution diagnostic, not a physical intervention.'))


def cage_rotations(a, positions, atom_ids):
    """Proper least-squares rotations of common nearest-80 atom identities."""
    nf, nc = len(a['frames']), len(a['centers'])
    near = a['nearest_ids'].reshape(nf, nc, 80)
    rotations = np.empty((nf-1, nc, 3, 3), np.float32)
    residual = np.empty((nf-1, nc), np.float32)
    common_count = np.empty((nf-1, nc), np.int16)
    for t in range(nf-1):
        for c in range(nc):
            common = np.intersect1d(near[t, c], near[t+1, c], assume_unique=True)
            common = common[common != a['centers'][c]]
            if len(common) < 12:
                raise ValueError(f'Insufficient common cage atoms: time={t}, center={c}')
            clouds = []
            for row in (t*nc+c, (t+1)*nc+c):
                lo, hi = a['offsets'][row:row+2]
                ids = atom_ids[lo:hi]
                indices = np.searchsorted(ids, common)
                np.testing.assert_array_equal(ids[indices], common)
                clouds.append(positions[lo:hi][indices].astype(float))
            x, y = clouds
            u, _, vt = np.linalg.svd(x.T@y)
            correction = np.diag([1., 1., np.linalg.det(u@vt)])
            r = u@correction@vt
            rotations[t, c] = r
            residual[t, c] = np.sqrt(np.mean(np.sum((x@r-y)**2, axis=-1)))
            common_count[t, c] = len(common)
    return dict(cage_rotation=rotations, cage_fit_rms_A=residual, cage_common_atoms=common_count)


def extract_temporal(model, config, parent):
    root = Path(config['output'])/'technical'
    audited = False
    for source in parent['sources']:
        dest = root/'temporal'/str(source['id'])
        if (dest/'complete.json').exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        folder = Path(config['parent_audit'])/'technical/sources'/str(source['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        for filename, digest in receipt['hashes'].items():
            if file_hash(folder/filename) != digest:
                raise ValueError(f'Changed parent observation: {folder/filename}')
        a = dict(np.load(folder/'observations.npz'))
        x = np.load(folder/'positions.npy', mmap_mode='r')
        ids = np.load(folder/'atom_ids.npy', mmap_mode='r')
        positions = [x[lo:hi] for lo, hi in zip(a['offsets'][:-1], a['offsets'][1:], strict=True)]
        started = time.monotonic()
        z, mv = encode_rows(model, positions, a['center_indices'], config['batch_size'])
        prior = np.load(folder/'gatr.npy')
        np.testing.assert_allclose(z, prior, rtol=2e-5, atol=2e-6,
            err_msg=f'Hooked A100 inference disagrees with frozen native GATr states, source {source["id"]}')
        if not audited:
            numerical_audit(model, positions, a['center_indices'], root, config['seed'])
            audited = True
        vectors = vector_parts(torch.from_numpy(mv)).numpy()
        baseline = [geometry_baselines(p, model.scale) for p in positions]
        extra = cage_rotations(a, x, ids) if source['split'] == 'test' else {}
        np.savez(dest/'features.npz', z=z, multivectors=mv, vectors=vectors,
            baseline=np.stack([v[0] for v in baseline]), shape_gap=np.array([v[1] for v in baseline]),
            frames=a['frames'], centers=a['centers'], times_ps=a['times_ps'], labels=a['labels'], order=a['order'], **extra)
        save_json(dest/'complete.json', dict(source=source['id'], rows=len(z), split=source['split'],
            native_z_max_abs=float(np.max(np.abs(z-prior))), sha256=file_hash(dest/'features.npz'),
            elapsed_seconds=time.monotonic()-started))
        print(f'Temporal {source["id"]}: {len(z)} observations, {time.monotonic()-started:.1f}s', flush=True)


def extract_spatial(model, config, parent):
    from src.analysis.liquid_structure import bond_order
    from src.research.smooth_temporal_encoder.prepare import ptm_labels
    root = Path(config['output'])/'technical/spatial'
    root.mkdir(exist_ok=True)
    radius = OUTER_RADIUS*model.scale/REFERENCE_RADIUS
    for source in [s for s in parent['sources'] if s['split'] == 'test']:
        raw = source_arrays(dict(source, kind='dynamic'))
        for frame in config['spatial_frames']:
            dest = root/f'{source["id"]}-{frame:04d}.npz'
            if dest.exists():
                if not dest.with_suffix('.json').exists():
                    raise RuntimeError(f'Unverified spatial extraction: {dest}')
                continue
            started = time.monotonic()
            points, tree, box = chart(raw, frame, False)
            seed_ids = np.array(source['selected_centers'])[config['patch_centers']]
            seed_rows = np.searchsorted(raw['atom_ids'], seed_ids)
            np.testing.assert_array_equal(raw['atom_ids'][seed_rows], seed_ids)
            rng = np.random.default_rng(np.random.SeedSequence([config['seed'], source['id'], frame]))
            rows, patches = [], []
            for index, seed_row in enumerate(seed_rows):
                current = tree.query(points[seed_row], k=config['atoms_per_patch'])[1]
                rows.extend(current.tolist()); patches.extend([index]*len(current))
            current = rng.choice(len(points), config['uniform_centers'], replace=False)
            rows.extend(current.tolist()); patches.extend([len(seed_rows)]*len(current))
            _, first = np.unique(rows, return_index=True)
            first.sort(); rows = np.asarray(rows)[first]; patches = np.asarray(patches)[first]
            groups = tree.query_ball_point(points[rows], radius, return_sorted=True)
            positions = [offsets(points, center, np.asarray(neighbors), box) for center, neighbors in zip(rows, groups, strict=True)]
            centers = [int(np.flatnonzero(np.asarray(neighbors) == center).item()) for center, neighbors in zip(rows, groups, strict=True)]
            z, mv = encode_rows(model, positions, centers, config['batch_size'])
            baseline = [geometry_baselines(p, model.scale) for p in positions]
            _, nearest = tree.query(points[rows], k=80)
            local = points[nearest[:, 1:]]-points[rows, None]
            local -= box*np.round(local/box)
            labels = ptm_labels(local/10., .1)
            first13 = nearest[:, :13]
            _, neighbors = tree.query(points[first13], k=13)
            bonds = points[neighbors[:, :, 1:]]-points[first13][:, :, None, :]
            bonds -= box*np.round(bonds/box)
            order, _ = bond_order(bonds, 3.5)
            np.savez(dest, source=source['id'], frame=frame, time_ps=raw['timesteps'][frame]*source['timestep_fs']/1000,
                temperature_K=source['temperature_K'], atom_ids=raw['atom_ids'][rows], positions=points[rows],
                box=box, patch=patches, patch_anchor_ids=seed_ids, patch_anchors=points[seed_rows],
                nearest80_ids=raw['atom_ids'][nearest], z=z, multivectors=mv,
                vectors=vector_parts(torch.from_numpy(mv)).numpy(), labels=labels, order=order,
                baseline=np.stack([v[0] for v in baseline]), shape_gap=np.array([v[1] for v in baseline]))
            save_json(dest.with_suffix('.json'), dict(source=source['id'], frame=frame, rows=len(z),
                sha256=file_hash(dest), source_manifest_sha256=source['manifest_sha256'], seconds=time.monotonic()-started))
            print(f'Spatial {source["id"]} frame {frame}: {len(z)} atoms, {time.monotonic()-started:.1f}s', flush=True)


def run(config, stage):
    if socket.gethostname().split('.')[0] != config['required_hostname']:
        raise RuntimeError(f'This requested computation must run on {config["required_hostname"]}, got {socket.gethostname()}')
    if not torch.cuda.is_available() or config['required_gpu'] not in torch.cuda.get_device_name(0):
        raise RuntimeError('The requested A100 GPU is not visible')
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    root = Path(config['output'])/'technical'; root.mkdir(parents=True, exist_ok=True)
    parent = json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != config:
        raise ValueError('Existing equivariant audit has a different configuration')
    save_json(root/'config.json', config)
    save_json(root/'environment.json', dict(hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__, stages=STAGES, sectors=SECTORS, checkpoint_sha256=config['checkpoint_sha256'],
        parent_plan_sha256=file_hash(Path(config['parent_audit'])/'technical/plan.json')))
    if not (root/'encoder.pt').exists():
        shutil.copy2(config['checkpoint'], root/'encoder.pt')
    model = Capture(root/'encoder.pt', config['checkpoint_sha256']).to(config['device']).eval()
    if stage in ('all', 'temporal'):
        extract_temporal(model, config, parent)
    if stage in ('all', 'spatial'):
        extract_spatial(model, config, parent)
    if stage == 'all':
        from .controls import controls
        controls(config)
        from .report import report
        report(config, parent)
