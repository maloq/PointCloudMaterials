"""Independent full-cell PTM and sampled bond-order reference, with explicit gaps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y

from src.analysis.liquid_structure import wigner_terms, persistence_image
from src.data.static_sources import load_points

ORDER_NAMES = ['q4', 'q6', 'hat_w4', 'hat_w6', 'qbar6', 'q6_coherence', 'density_knn']
CONTEXT_NAMES = ['unclassified_liquid', 'crystal_interior', 'solid_liquid_boundary',
                 'Al_planar_fault', 'non_template_crystal_interior',
                 'fivefold_liquid_candidate', 'ordered_liquid_candidate']


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def bond_descriptors(vectors):
    """[batch,k+1,k,3] bonds; center plus its k neighbors. Equal bond weights."""
    b, core, k, xyz = vectors.shape
    if core != k+1 or xyz != 3 or k not in (12, 14):
        raise ValueError(f'Wrong neighbor-of-neighbor geometry: {vectors.shape}')
    r = np.linalg.norm(vectors, axis=-1)
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError('Bond order requires finite, nonzero bond lengths.')
    theta = np.arccos(np.clip(vectors[..., 2]/r, -1, 1))
    phi = np.arctan2(vectors[..., 1], vectors[..., 0])
    q, result = {}, []
    for ell in (4, 6):
        q[ell] = np.stack([sph_harm_y(ell, m, theta, phi).mean(-1)
                           for m in range(-ell, ell+1)], -1)
        result.append(np.sqrt(4*np.pi/(2*ell+1))*np.linalg.norm(q[ell][:, 0], axis=-1))
    for ell in (4, 6):
        ids, weights = wigner_terms(ell)
        values = q[ell][:, 0]
        cubic = (values[:, ids].prod(-1)*weights).sum(-1).real
        norm = np.linalg.norm(values, axis=-1)
        result.append(np.divide(cubic, norm**3, out=np.zeros_like(cubic), where=norm>1e-14))
    norm = np.linalg.norm(q[6], axis=-1)
    unit = np.divide(q[6], norm[..., None], out=np.zeros_like(q[6]), where=norm[..., None]>1e-14)
    coherence = np.einsum('bm,bnm->bn', unit[:, 0].conj(), unit[:, 1:]).real
    result.extend([np.sqrt(4*np.pi/13)*np.linalg.norm(q[6].mean(1), axis=-1),
                   coherence.mean(1), k/(4*np.pi*r[:, 0, -1]**3/3)])
    counts = np.stack([(coherence>c).sum(1) for c in (.65, .7, .75)], -1)
    return np.stack(result, -1).astype(np.float32), counts.astype(np.int8)


def contexts(ptm, solid_fraction, fault, order, threshold, material, w6_cutoff):
    """Operational overlapping axes projected to a declared priority class."""
    solid = np.isin(ptm, [1, 2, 3])
    out = np.zeros(len(ptm), dtype=np.int8)
    interior = solid_fraction >= .8
    out[solid & interior] = 1
    out[(solid & ~interior) | ((solid_fraction > .1) & ~interior)] = 2
    out[~solid & interior] = 4
    liquid = ~solid & (solid_fraction <= .1)
    out[liquid & (order[:, 4] >= threshold)] = 6
    out[liquid & ((ptm == 4) | (order[:, 3] < w6_cutoff))] = 5
    if material == 'Al':
        out[np.isin(fault, [2, 3, 4])] = 3
    return out


def full_ptm(points, material):
    from ovito.data import DataCollection, Particles
    from ovito.pipeline import Pipeline, StaticSource
    from ovito.modifiers import PolyhedralTemplateMatchingModifier, IdentifyFCCPlanarFaultsModifier
    particles = Particles(count=len(points))
    particles.create_property('Position', data=points)
    data = DataCollection(); data.objects.append(particles)
    pipeline = Pipeline(source=StaticSource(data=data))
    modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0, output_rmsd=True,
                        output_orientation=True, output_interatomic_distance=True)
    modifier.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
    pipeline.modifiers.append(modifier)
    result = pipeline.compute()
    best = np.asarray(result.particles['Structure Type']).astype(np.int8)
    rmsd = np.asarray(result.particles['RMSD']).astype(np.float32)
    fault = np.zeros(len(points), dtype=np.int8)
    if material == 'Al':
        modifier.rmsd_cutoff = .1
        pipeline.modifiers.append(IdentifyFCCPlanarFaultsModifier())
        result = pipeline.compute()
        fault = np.asarray(result.particles['Planar Fault Type']).astype(np.int8)
    return best, rmsd, fault


def prepare_frame(cfg, frame, index, destination):
    started = time.monotonic()
    points = load_points(frame['file'])
    best, rmsd, fault = full_ptm(points, frame['material'])
    print(f'{frame["file"]}: full PTM complete ({len(points)} atoms)', flush=True)
    np.savez_compressed(destination/f'full-reference-{index:02d}.npz', best_ptm=best,
                        rmsd=rmsd, planar_fault=fault)
    tree = cKDTree(points, balanced_tree=False)
    # No box inferred from extrema. All scored centers have full local support.
    lo, hi = points.min(0), points.max(0)
    radius = frame['radius_A']
    eligible = np.flatnonzero(((points > lo+2*radius) & (points < hi-2*radius)).all(1))
    rng = np.random.default_rng(cfg['seed']+index)
    anchors = rng.choice(eligible, cfg['centers_per_snapshot'], replace=False)
    _, pair = tree.query(points[anchors], k=2, workers=2)
    if not np.array_equal(pair[:, 0], anchors):
        raise ValueError(f'Duplicate positions or wrong center identity in {frame["file"]}')
    rows = np.r_[anchors, pair[:, 1]]
    distances, near = tree.query(points[rows], k=81, workers=2)
    if not np.array_equal(near[:, 0], rows):
        raise ValueError('Input center identity mismatch.')
    if (distances[:, 79] > radius).any():
        raise ValueError(f'80 atoms extend beyond training normalization radius: {frame["file"]}')
    cloud = points[near[:, :80]] - points[rows, None]
    split = np.full(len(rows), -1, dtype=np.int8)
    mid = (lo[0]+hi[0])/2
    split[points[rows, 0] < mid-1.1*radius] = 0
    split[points[rows, 0] > mid+1.1*radius] = 1
    order, connected = [], []
    for k in cfg['neighbors']:
        blocks, counts = [], []
        for start in range(0, len(rows), 256):
            core = near[start:start+256, :k+1]
            _, neighbors = tree.query(points[core], k=k+1, workers=2)
            vectors = points[neighbors[:, :, 1:]].astype(np.float64)-points[core][:, :, None]
            values, connected_count = bond_descriptors(vectors)
            blocks.append(values); counts.append(connected_count)
        order.append(np.concatenate(blocks)); connected.append(np.concatenate(counts))
    order, connected = np.stack(order, 1), np.stack(connected, 1)
    types, fractions = [], []
    for cutoff in cfg['ptm_cutoffs']:
        types.append(np.where(rmsd[rows] <= cutoff, best[rows], 0))
        all_solid = np.isin(best, [1, 2, 3]) & (rmsd <= cutoff)
        fractions.append(all_solid[near[:, 1:15]].mean(1))
    types, fractions = np.stack(types, 1), np.stack(fractions, 1)
    primary = cfg['ptm_cutoffs'].index(cfg['primary_ptm_cutoff'])
    liquid = ~np.isin(types[:, primary], [1, 2, 3]) & (fractions[:, primary] <= .1)
    fit = liquid & (split == 0) & (np.arange(len(rows)) < len(anchors))
    if fit.sum() < 50:
        raise ValueError(f'Fewer than 50 fitting liquid centers in {frame["file"]}; threshold undefined.')
    threshold = float(np.quantile(order[fit, 0, 4], cfg['liquid_order_quantile']))
    labels = np.stack([contexts(types[:, j], fractions[:, j], fault[rows], order[:, 0],
            threshold, frame['material'], cfg['competing_hat_w6_threshold'])
            for j in range(len(cfg['ptm_cutoffs']))], 1)
    topology_rows = rng.choice(len(anchors), cfg['topology_centers_per_snapshot'], replace=False)
    topology = np.stack([persistence_image(cloud[j, :65]) for j in topology_rows])
    np.savez(destination/f'frame-{index:02d}.npz', clouds=cloud/radius,
        rows=rows, coords=points[rows], split=split, order=order, connections=connected,
        ptm=types, solid_fraction=fractions, fault=fault[rows], context=labels,
        rmsd=rmsd[rows], pair_distance=distances[:len(anchors), 1],
        topology_rows=topology_rows, topology=topology)
    record = dict(**frame, frame_index=index, count=len(rows), anchor_count=len(anchors),
        input_sha256=hashlib.sha256(Path(frame['file']).read_bytes()).hexdigest(),
        points=len(points), potential='unknown', periodic_boundary='unavailable; free boundaries with margin',
        split_gap_A=2.2*radius, order_threshold_qbar6=threshold, liquid_calibration_count=int(fit.sum()),
        class_counts={CONTEXT_NAMES[j]: int((labels[:len(anchors), primary]==j).sum()) for j in range(7)},
        seconds=time.monotonic()-started)
    write_json(destination/f'frame-{index:02d}.json', record)
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    destination = Path(args.output)/'technical/reference'; destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'config.json', cfg)
    records = []
    for i, frame in enumerate(cfg['frames']):
        record = destination/f'frame-{i:02d}.json'
        if record.exists():
            raise FileExistsError(f'Reference already exists: {record}; use a new output for a changed assay.')
        records.append(prepare_frame(cfg, frame, i, destination))
    # Freeze one threshold per material across its entire checkpoint/time assay.
    # Only fitting-side anchors contribute; neighbor copies never upweight calibration.
    for material in ('Al', 'Ta', 'Zr'):
        selected = [r for r in records if r['material'] == material]
        ki = 0 if material == 'Al' else 1
        calibration = []
        for r in selected:
            a = np.load(destination/f'frame-{r["frame_index"]:02d}.npz')
            n = r['anchor_count']
            fit = (a['split'][:n] == 0) & ~np.isin(a['ptm'][:n, 1], [1, 2, 3]) & (a['solid_fraction'][:n, 1] <= .1)
            calibration.append(a['order'][:n, ki, 4][fit])
        calibration = np.concatenate(calibration)
        threshold = float(np.quantile(calibration, cfg['liquid_order_quantile']))
        for r in selected:
            path = destination/f'frame-{r["frame_index"]:02d}.npz'
            with np.load(path) as loaded:
                a = dict(loaded)
            a['context'] = np.stack([contexts(a['ptm'][:, j], a['solid_fraction'][:, j], a['fault'],
                a['order'][:, ki], threshold, material, cfg['competing_hat_w6_threshold']) for j in range(3)], 1)
            np.savez(path, **a)
            r.update(order_threshold_qbar6=threshold, liquid_calibration_count=len(calibration),
                primary_order_neighbors=cfg['neighbors'][ki],
                class_counts={CONTEXT_NAMES[j]: int((a['context'][:r['anchor_count'], 1]==j).sum()) for j in range(7)})
            write_json(destination/f'frame-{r["frame_index"]:02d}.json', r)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob('frame-*.npz'))}
    write_json(destination/'manifest.json', dict(frames=records, order_names=ORDER_NAMES, files=files,
        context_names=CONTEXT_NAMES, cfg_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
