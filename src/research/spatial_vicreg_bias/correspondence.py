"""Independent rich-descriptor clusters versus frozen neural clusters at interfaces."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src.experiment_runner.metric_docs import write_metric_table, check_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin.extract import raw_source, frame_geometry
from src.research.crystal_vector.interface import interface_mask
from src.research.structural_state.common import sha, write_json
from .data import load


def configuration(path):
    c = json.loads(Path(path).read_text())
    if c['protocol'] != 'interface_cluster_correspondence_v1':
        raise ValueError('Wrong interface correspondence protocol')
    for key in ('output', 'publication'):
        value = Path(c[key])
        c[key] = str(value.resolve() if value.is_absolute() else resolve_path(c[key]).resolve())
    parent = load(c['parent'])
    manifest = json.loads((Path(parent['cache'])/'manifest.json').read_text())
    if manifest['state'] != 'complete' or manifest['identity'] != c['cache_identity']:
        raise ValueError('Changed parent descriptor cache')
    return c, parent


def observations(parent):
    base = Path(parent['cache'])/'assay'
    ids = np.flatnonzero(np.load(base/'uniform.npy'))
    keys = ('source', 'frame', 'atom', 'role', 'ptm', 'support_fraction', 'visible_A',
            'coords', 'box', 'targets')
    a = {key: np.asarray(np.load(base/(key+'.npy'), mmap_mode='r')[ids]) for key in keys}
    a['original_row'] = ids
    a['names'] = json.loads((base/'descriptors.json').read_text())
    return a


def prepare_source(config, sid):
    c, parent = configuration(config)
    out = Path(c['output'])/'data/interface-sources'/str(sid)
    receipt = out/'complete.json'
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r['config_sha256'] != sha(config) or sha(out/'rows.npz') != r['rows_sha256']:
            raise ValueError(f'Changed interface source {sid}')
        return r
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((Path(parent['cache'])/'plan.json').read_text())
    item = next(s for s in plan['sources'] if s['id'] == sid)
    raw = raw_source(item)
    source = Path(parent['cache'])/'sources'/str(sid)
    record = json.loads((source/'complete.json').read_text())
    if sha(source/'assay.npz') != record['files']['assay.npz']:
        raise ValueError(f'Changed parent assay {sid}')
    with np.load(source/'assay.npz') as z:
        uniform = z['uniform']
        a = {k: z[k][uniform] for k in ('source', 'frame', 'atom', 'ptm', 'coords')}
    audit = Path(parent['ptm_audit'])/'technical/sources'/str(sid)
    ptm = json.loads((audit/'ptm-complete.json').read_text())
    fields = {k: [] for k in ('source', 'frame', 'atom', 'distance', 'solid', 'large_crystal',
                              'accepted_disorder', 'interface_member')}
    chunks = {}; frame_records = []
    for frame in np.unique(a['frame']):
        frame = int(frame); take = a['frame'] == frame
        points, box = frame_geometry(raw, frame)
        atoms = np.searchsorted(raw.atom_ids, a['atom'][take])
        if not np.array_equal(raw.atom_ids[atoms], a['atom'][take]):
            raise ValueError(f'Atom mismatch {sid}/{frame}')
        if not np.array_equal(points[atoms].astype(np.float32), a['coords'][take]):
            raise ValueError(f'Coordinate mismatch {sid}/{frame}')
        begin = frame//32*32; stop = min(item['frame_count'], begin+32)
        name = f'ptm-{begin:04d}-{stop:04d}.npz'
        if sha(audit/name) != ptm['files'][name]:
            raise ValueError(f'Changed PTM {sid}/{name}')
        chunks[name] = ptm['files'][name]
        with np.load(audit/name) as z: labels = z['labels'][frame-begin]
        if not np.array_equal(labels[atoms], a['ptm'][take]):
            raise ValueError(f'PTM center mismatch {sid}/{frame}')
        solid = np.isin(labels, [1, 2, 3]); solid_ids = np.flatnonzero(solid)
        large = np.zeros(len(points), bool)
        if len(solid_ids):
            pairs = cKDTree(points[solid_ids], boxsize=box).query_pairs(
                c['interface']['cutoff_A'], output_type='ndarray')
            graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(solid_ids), len(solid_ids))).tocsr()
            _, comp = connected_components(graph, directed=False)
            large[solid_ids] = np.bincount(comp)[comp] >= c['interface']['minimum_crystal_component']
        boundary, accepted = interface_mask(points, box, solid, c['interface']['cutoff_A'],
                                            c['interface']['minimum_disordered_component'])
        boundary &= large
        distance = np.full(len(atoms), np.inf, np.float32)
        if boundary.any():
            distance = cKDTree(points[boundary], boxsize=box).query(points[atoms])[0].astype(np.float32)
        row = dict(source=a['source'][take], frame=a['frame'][take], atom=a['atom'][take],
                   distance=distance, solid=solid[atoms], large_crystal=large[atoms],
                   accepted_disorder=accepted[atoms], interface_member=boundary[atoms])
        for key in fields: fields[key].append(row[key])
        frame_records.append(dict(frame=frame, crystal_atoms=int(solid.sum()), large_crystal_atoms=int(large.sum()),
                                  interface_atoms=int(boundary.sum()), accepted_disordered_atoms=int(accepted.sum())))
    np.savez_compressed(out/'rows.npz', **{k: np.concatenate(v) for k, v in fields.items()})
    write_json(out/'frames.json', frame_records)
    r = dict(source=sid, config_sha256=sha(config), rows_sha256=sha(out/'rows.npz'),
             parent_assay_sha256=record['files']['assay.npz'], ptm_chunks=chunks,
             rows=sum(len(v) for v in fields['source']))
    write_json(receipt, r)
    print(f'Interface reference complete: source {sid}', flush=True)
    return r


def prepare(config):
    c, parent = configuration(config)
    plan = json.loads((Path(parent['cache'])/'plan.json').read_text())
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        futures = [pool.submit(prepare_source, config, s['id']) for s in plan['sources']]
        for future in as_completed(futures): future.result()


def references(c, a):
    files = []; bindings = {}
    for sid in np.unique(a['source']):
        p = Path(c['output'])/'data/interface-sources'/str(sid)
        r = json.loads((p/'complete.json').read_text())
        if sha(p/'rows.npz') != r['rows_sha256']: raise ValueError(f'Changed source {sid}')
        bindings[str(p/'rows.npz')] = r['rows_sha256']
        with np.load(p/'rows.npz') as z: files.append({k: z[k] for k in z.files})
    ref = {k: np.concatenate([f[k] for f in files]) for k in files[0]}
    # Sources and frames are stored sorted in both producers; no approximate matching.
    for key in ('source', 'frame', 'atom'):
        if not np.array_equal(a[key], ref[key]): raise ValueError(f'Reference identity mismatch: {key}')
    return ref, bindings


def populations(c, ref, a):
    d = ref['distance']; solid = ref['solid']; large = ref['large_crystal']
    liquid = ref['accepted_disorder']; finite = np.isfinite(d)
    masks = dict(all=np.ones(len(d), bool), interface12=finite & (d <= c['interface']['fit_band_A']),
                 crystal_interface_layer=ref['interface_member'],
                 mixed_patch=finite & (d <= 12) & (a['support_fraction'] > 0) & (a['support_fraction'] < 1),
                 small_disordered_pocket=finite & ~solid & ~liquid,
                 crystal_free_near_liquid=finite & liquid & ~a['visible_A'] & (d <= 20),
                 no_interface=~finite)
    for lo, hi in zip(c['interface']['shell_edges_A'][:-1], c['interface']['shell_edges_A'][1:]):
        shell = finite & (d > lo) & (d <= hi)
        masks[f'crystal_{lo:g}_{hi:g}A'] = shell & large & ~ref['interface_member']
        masks[f'liquid_{lo:g}_{hi:g}A'] = shell & liquid
    masks['crystal_beyond20A'] = finite & large & (d > 20)
    masks['liquid_beyond20A'] = finite & liquid & (d > 20)
    return masks


def table(x, y, k):
    return np.bincount(x.astype(int)*k+y.astype(int), minlength=k*k).reshape(k, k)


def information(t):
    t = np.asarray(t, float); n = t.sum(axis=(-2, -1)); safe = np.maximum(n, 1)
    p = t/safe[..., None, None]; r = p.sum(-1); col = p.sum(-2)
    def entropy(v): return -(v*np.log(np.maximum(v, 1e-300))).sum(-1)
    hx = entropy(r); hy = entropy(col)
    mi = (p*np.log(np.maximum(p, 1e-300)/np.maximum(r[..., :, None]*col[..., None, :], 1e-300))).sum((-2, -1))
    both = (t*(t-1)/2).sum((-2, -1)); rows = t.sum(-1); cols = t.sum(-2)
    aa = (rows*(rows-1)/2).sum(-1); bb = (cols*(cols-1)/2).sum(-1)
    expected = aa*bb/np.maximum(n*(n-1)/2, 1); denom = (aa+bb)/2-expected
    ari = np.divide(both-expected, denom, out=np.ones_like(n), where=np.abs(denom)>1e-12)
    return dict(ari=ari, mutual_information_nats=mi, descriptor_entropy=hx, neural_entropy=hy,
                descriptor_given_neural_entropy=np.maximum(hx-mi, 0),
                neural_given_descriptor_entropy=np.maximum(hy-mi, 0))


def agreement(x, y, source, k, bootstrap=0):
    t = table(x, y, k); out = {key: float(v) for key, v in information(t).items()}
    out.update(rows=len(x), sources=len(np.unique(source)), ami=float(adjusted_mutual_info_score(x, y)),
               descriptor_counts=t.sum(1).tolist(), neural_counts=t.sum(0).tolist(), contingency=t.tolist())
    if bootstrap:
        ids = np.unique(source); tt = np.stack([table(x[source==s], y[source==s], k) for s in ids])
        rng = np.random.default_rng(20260929)
        sampled = tt[rng.integers(len(ids), size=(bootstrap, len(ids)))].sum(1)
        out['source_bootstrap_ari'] = dict(ci95=np.quantile(information(sampled)['ari'], [.025, .975]).tolist())
    return out


def matched_null(x, y, a, ref, rows, k, repetitions):
    # Preserve the cell, physical side, 3.6-A distance shell and crystal fraction quartile.
    d = ref['distance'][rows]
    keys = np.column_stack([a['source'][rows], a['frame'][rows], ref['solid'][rows],
                            np.floor(d/3.6).astype(int), np.minimum((a['support_fraction'][rows]*4).astype(int), 3)])
    _, group = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable'); cuts = np.r_[0, np.flatnonzero(np.diff(group[order]))+1, len(order)]
    groups = [order[lo:hi] for lo, hi in zip(cuts[:-1], cuts[1:]) if hi-lo > 1]
    movable = sum(len(g) for g in groups)
    rng = np.random.default_rng(20260929); values = []
    for _ in range(repetitions):
        shuffled = y.copy()
        for g in groups: shuffled[g] = y[rng.permutation(g)]
        values.append(float(information(table(x, shuffled, k))['mutual_information_nats']))
    observed = float(information(table(x, y, k))['mutual_information_nats'])
    return dict(observed_mi=observed, mean_shuffled_mi=float(np.mean(values)),
                excess_mi=observed-float(np.mean(values)), shuffle_range95=np.quantile(values, [.025, .975]).tolist(),
                movable_fraction=movable/len(x), repetitions=repetitions,
                interpretation='descriptive matched-shuffle contrast; not an iid atom-level significance test')


def descriptor_clusters(c, a, masks, out):
    result = {}; labels = {}; fit_models = {}; train = a['role'] == 'train'
    for scope in c['fit_populations']:
        fit = train & masks[scope]
        if fit.sum() < 500: raise ValueError(f'Insufficient fitting population {scope}: {fit.sum()}')
        sources, inv, counts = np.unique(a['source'][fit], return_inverse=True, return_counts=True)
        weight = len(inv)/(len(sources)*counts[inv])
        for family in c['families']:
            families = ['tda', 'bond_order', 'cna'] if family == 'joint' else [family]
            columns = np.array([i for i, n in enumerate(a['names']) if n.split('/')[0] in families])
            x = a['targets'][:, columns].astype(float)
            mean = np.average(x[fit], axis=0, weights=weight)
            sd = np.sqrt(np.average((x[fit]-mean)**2, axis=0, weights=weight))
            active = sd > 1e-8*np.maximum(1, np.abs(mean))
            columns = columns[active]; mean = mean[active]; sd = sd[active]
            names = [a['names'][i] for i in columns]
            z = ((a['targets'][:, columns]-mean)/sd).astype(np.float32)
            balance = np.ones(len(columns))
            for f in families:
                ix = np.array([n.startswith(f+'/') for n in names]); balance[ix] = np.sqrt(ix.sum())
            z /= balance
            for k in c['ks']:
                tag = f'{scope}-{family}-k{k}'; seed_labels = []
                for seed in c['cluster_seeds']:
                    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, max_iter=200,
                                        random_state=seed, reassignment_ratio=0)
                    km.fit(z[fit], sample_weight=weight)
                    lab = km.predict(z).astype(np.uint8); seed_labels.append(lab)
                    if seed == c['primary_cluster_seed']:
                        labels[tag] = lab
                        fit_models[tag] = dict(columns=columns, mean=mean, sd=sd, balance=balance, centers=km.cluster_centers_)
                held = (a['role']=='test') & masks['interface12']
                stability = [float(adjusted_rand_score(seed_labels[i][held], seed_labels[j][held]))
                             for i in range(len(seed_labels)) for j in range(i+1, len(seed_labels))]
                result[tag] = dict(training_rows=int(fit.sum()), training_sources=len(sources),
                                   active_features=len(names), names=names, interface_seed_pair_ari=stability)
                np.savez_compressed(out/'data'/f'{tag}-descriptor-model.npz', **fit_models[tag],
                                    assignments=np.stack(seed_labels), cluster_seeds=c['cluster_seeds'])
                print(f'Fitted descriptor clusters: {tag}', flush=True)
    write_json(out/'technical/descriptor-clustering.json', result)
    return labels, result, fit_models


def load_neural(path, a, bindings):
    bindings[str(path)] = sha(path)
    with np.load(path) as z:
        for key in ('source', 'frame', 'atom'):
            if not np.array_equal(z[key][a['original_row']], a[key]):
                raise ValueError(f'Neural assignment identity mismatch: {path}, {key}')
        return z['cluster'][a['original_row']]


def heatmap(t, path, title):
    t = np.asarray(t, float)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, den, label in zip(axs, [t.sum(1)[:, None], t.sum(0)[None, :]],
                             ['Neural membership within descriptor cluster', 'Descriptor membership within neural cluster']):
        values = np.divide(t, den, out=np.zeros_like(t), where=den>0)
        im = ax.imshow(values, vmin=0, vmax=1, cmap='magma')
        ax.set(xlabel='Neural cluster', ylabel='Descriptor cluster', title=label,
               xticks=np.arange(len(t)), yticks=np.arange(len(t)))
        fig.colorbar(im, ax=ax)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def profiles(a, ref, labels, name, mask, model, out):
    """Rich feature signatures; retain every feature numerically and show all columns."""
    columns = model['columns']; values = (a['targets'][:, columns]-model['mean'])/model['sd']
    k = len(model['centers']); means = np.full((k, len(columns)), np.nan); count = []
    for j in range(k):
        take = mask & (labels == j); count.append(int(take.sum()))
        if take.any(): means[j] = values[take].mean(0)
    np.savez_compressed(out/'data'/f'{name}-signatures.npz', standardized_means=means, counts=count,
                        columns=columns, names=np.array([a['names'][i] for i in columns]))
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(np.ma.masked_invalid(means), aspect='auto', vmin=-3, vmax=3, cmap='coolwarm')
    ax.set(xlabel='Rich descriptor coordinate (complete ordered feature bank)', ylabel='Cluster',
           title=name+'; held-out interface ±12 Å; training-standardized means',
           yticks=np.arange(k), yticklabels=[f'{j} (n={n})' for j, n in enumerate(count)])
    fig.colorbar(im, ax=ax, label='Training standard deviations; display clipped at ±3')
    fig.tight_layout(); fig.savefig(out/'plots/signatures'/f'{name}.png', dpi=150); plt.close(fig)
    # Choose readable coordinates from training descriptor centroids, not held-out contrasts.
    contrast = np.var(model['centers']*model['balance'], axis=0)
    selected = []
    for family in ('tda', 'bond_order', 'cna'):
        ix = np.array([j for j,i in enumerate(columns) if a['names'][i].startswith(family+'/')], int)
        selected.extend(ix[np.argsort(-contrast[ix])[:10]].tolist())
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(np.ma.masked_invalid(means[:, selected]), aspect='auto', vmin=-3, vmax=3, cmap='coolwarm')
    ax.set(xticks=np.arange(len(selected)), xticklabels=[a['names'][columns[j]] for j in selected],
           yticks=np.arange(k), ylabel='Cluster', title=name+'; readable training-selected signature coordinates')
    plt.setp(ax.get_xticklabels(), rotation=70, ha='right', fontsize=7)
    fig.colorbar(im, ax=ax, label='Training standard deviations')
    fig.tight_layout(); fig.savefig(out/'plots/signatures'/f'{name}-named-features.png', dpi=150); plt.close(fig)


def layer_plot(a, ref, x, y, path, title):
    d=ref['distance']; solid=ref['solid']; held=a['role']=='test'
    strata=[('crystal 8–12 Å', solid & (d>8) & (d<=12)),
            ('crystal 3.6–8 Å', solid & (d>3.6) & (d<=8)),
            ('crystal 0–3.6 Å', solid & (d>0) & (d<=3.6)),
            ('boundary layer', ref['interface_member']),
            ('disorder 0–3.6 Å', ref['accepted_disorder'] & (d<=3.6)),
            ('disorder 3.6–8 Å', ref['accepted_disorder'] & (d>3.6) & (d<=8)),
            ('disorder 8–12 Å', ref['accepted_disorder'] & (d>8) & (d<=12)),
            ('disorder 12–20 Å', ref['accepted_disorder'] & (d>12) & (d<=20))]
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for ax,labels,label in zip(axs,(x,y),('Rich descriptor clusters','Neural clusters')):
        counts=np.stack([np.bincount(labels[held & mask],minlength=7) for _,mask in strata])
        total=counts.sum(1); values=np.divide(counts,total[:,None],out=np.zeros_like(counts,dtype=float),where=total[:,None]>0)
        im=ax.imshow(values,aspect='auto',vmin=0,vmax=1,cmap='magma')
        ax.set(title=label,xlabel='Cluster number (independent labels)',yticks=np.arange(len(strata)),
               yticklabels=[f'{label} (n={n})' for (label,_),n in zip(strata,total)])
        fig.colorbar(im,ax=ax,label='Fraction within physical layer')
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def compare(config):
    c, parent = configuration(config); out = Path(c['output'])
    if (out/'technical/complete.json').exists(): raise FileExistsError('Completed correspondence revision exists')
    for part in ('data', 'tables', 'technical', 'plots/correspondence', 'plots/signatures', 'plots/spatial'):
        (out/part).mkdir(parents=True, exist_ok=True)
    a = observations(parent); ref, bindings = references(c, a); masks = populations(c, ref, a)
    test = a['role']=='test'; primary = test & masks['interface12']
    if primary.sum() < 500: raise ValueError(f'Insufficient interface observations: {primary.sum()}')
    physical, fits, models = descriptor_clusters(c, a, masks, out)
    coverage = {n: {role: int((m & (a['role']==role)).sum()) for role in np.unique(a['role'])}
                for n, m in masks.items()}
    np.savez_compressed(out/'data/observation-identities.npz', **{k: a[k] for k in ('source', 'frame', 'atom', 'role', 'original_row')},
                        **{k: ref[k] for k in ('distance', 'solid', 'large_crystal', 'accepted_disorder', 'interface_member')})
    for tag, lab in physical.items():
        if tag.endswith('k7'): profiles(a, ref, lab, tag, primary, models[tag], out)
    metrics = {}; curves = []; plot_labels = {}
    targets = []
    for seed in parent['training']['seed_values']:
        for alpha in parent['training']['alphas']:
            run = f'S{alpha:g}-seed{seed}'
            for epoch in parent['assay']['epochs']:
                for rep in ('encoder', 'projector'):
                    for k in c['ks']:
                        p = Path(parent['output'])/run/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k{k}-assignments.npz'
                        targets.append((f'{run}/epoch{epoch}/{rep}/k{k}', p, alpha, seed, epoch, rep, k))
    for name in ('diffusion-0', 'diffusion-1', 'diffusion-2', 'diffusion-4', 'local-q6', 'averaged-q6'):
        for k in c['ks']:
            p = Path(parent['output'])/'nulls/analyses'/name/'data'/f'{name}-k{k}-assignments.npz'
            targets.append((f'{name}/k{k}', p, None, None, None, 'classical_control', k))
    for name, path, alpha, seed, epoch, rep, k in targets:
        y = load_neural(path, a, bindings); result = {}
        for scope in c['fit_populations']:
            for family in c['families']:
                tag = f'{scope}-{family}-k{k}'; x = physical[tag]; blocks = {}
                for cohort, m in masks.items():
                    rows = np.flatnonzero(test & m)
                    if len(rows)<30:
                        blocks[cohort] = dict(rows=len(rows), state='insufficient_rows'); continue
                    detailed = k==7 and cohort=='interface12' and (epoch in (12,24) or epoch is None)
                    v = agreement(x[rows], y[rows], a['source'][rows], k,
                                  c['bootstrap_sources'] if detailed else 0)
                    if detailed:
                        v['matched_shuffle'] = matched_null(x[rows], y[rows], a, ref, rows, k, c['matched_permutations'])
                    blocks[cohort] = v
                result[tag] = blocks
                if k==7 and epoch is not None:
                    v = blocks['interface12']
                    curves.append(dict(run=name, alpha=alpha, seed=seed, epoch=epoch, representation=rep,
                                       descriptor=tag, ami=v['ami'], ari=v['ari']))
                if k==7 and seed==17 and epoch in (4,12,24) and family=='joint':
                    heatmap(blocks['interface12']['contingency'], out/'plots/correspondence'/f'{name.replace("/", "-")}-{scope}.png',
                            name+' versus '+tag+'; interface ±12 Å')
                    if scope=='interface12':
                        layer_plot(a,ref,x,y,out/'plots/correspondence'/f'{name.replace("/", "-")}-layers.png',name)
                        for shell in ('crystal_interface_layer','crystal_0_3.6A','liquid_0_3.6A','liquid_3.6_8A'):
                            if 'contingency' in blocks[shell]:
                                heatmap(blocks[shell]['contingency'],out/'plots/correspondence'/f'{name.replace("/", "-")}-{shell}.png',
                                        name+' versus '+tag+'; '+shell)
        metrics[name] = result
        if k==7 and seed==17 and epoch in (4,24) and rep=='encoder':
            tag='interface12-joint-k7'; plot_labels[name]=y
            profiles(a, ref, y, name.replace('/', '-'), primary, models[tag], out)
        write_json(out/'technical/progress.json', dict(completed=len(metrics), expected=len(targets), last=name))
        if k==10: print(f'Compared {name}', flush=True)
    write_json(out/'technical/correspondence.json', metrics)
    write_json(out/'technical/input-hashes.json', bindings)
    write_metric_table(dict(coverage=coverage, descriptor_fits=fits, comparisons=metrics), out,
                       family='interface_cluster_correspondence', name='cluster-correspondence')
    fig, axs = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    for row, rep in enumerate(('encoder', 'projector')):
        for col, family in enumerate(c['families']):
            ax=axs[row,col]
            for alpha in parent['training']['alphas']:
                epochs=parent['assay']['epochs']; v=np.array([[next(r['ami'] for r in curves
                    if r['alpha']==alpha and r['seed']==s and r['epoch']==e and r['representation']==rep
                    and r['descriptor']==f'interface12-{family}-k7') for e in epochs] for s in parent['training']['seed_values']])
                ax.plot(epochs, v.mean(0), marker='.', label=f'α={alpha:g}')
                ax.fill_between(epochs, v.min(0), v.max(0), alpha=.15)
            ax.set(title=f'{rep}: {family}', xlabel='Full epochs', ylabel='Adjusted mutual information'); ax.grid(alpha=.2)
    axs[0,0].legend(); fig.suptitle('Interface ±12 Å: independently fitted rich-descriptor clusters versus global neural K=7')
    fig.tight_layout(); fig.savefig(out/'plots/interface-correspondence-trajectories.png', dpi=170); plt.close(fig)
    # Sparse observed centers, explicitly no interpolation or inferred surfaces.
    candidates=[]
    for sid in np.unique(a['source'][test]):
        for f in np.unique(a['frame'][test & (a['source']==sid)]):
            mask=test & (a['source']==sid) & (a['frame']==f)
            candidates.append((int((mask & masks['interface12']).sum()), int(sid), int(f)))
    chosen=[]
    for count,sid,frame in sorted(candidates, reverse=True):
        if sid in [s for _,s,_ in chosen]: continue
        chosen.append((count,sid,frame))
        if len(chosen)==3: break
    for count,sid,frame in chosen:
        mask=test & (a['source']==sid) & (a['frame']==frame)
        xyz=a['coords'][mask]; sign=np.where(ref['solid'][mask], -1, 1)
        panels=[('Distance to crystal-side layer (Å)', np.clip(sign*ref['distance'][mask], -20,20), 'coolwarm'),
                ('Joint rich descriptors K=7', physical['interface12-joint-k7'][mask], 'tab10')]
        panels += [(name, lab[mask], 'tab10') for name,lab in plot_labels.items() if '/epoch24/' in name]
        fig,axs=plt.subplots(1,len(panels),figsize=(4*len(panels),4))
        for ax,(title,values,cmap) in zip(axs,panels):
            ax.scatter(xyz[:,0],xyz[:,1],c=values,cmap=cmap,s=9, vmin=-20 if cmap=='coolwarm' else 0,
                       vmax=20 if cmap=='coolwarm' else 9)
            ax.set(title=title,xlabel='x (Å)',ylabel='y (Å)',aspect='equal')
        fig.suptitle(f'Source {sid}, frame {frame}: same {mask.sum()} sampled centers, xy projection (not a dense slice)')
        fig.tight_layout(); fig.savefig(out/'plots/spatial'/f'source{sid}-frame{frame}.png',dpi=150);plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete', comparisons=len(targets), descriptor_clusterings=len(fits),
                                                coverage=coverage, job=os.environ.get('SLURM_JOB_ID')))
    (out/'README.md').write_text('# Rich-descriptor / neural cluster correspondence around interfaces\n\n'
        'Independent clusters of TDA, bond order, CNA and the family-balanced joint vector are fitted on training sources. '
        'Neural cluster assignments are the frozen original global clusterings. Descriptor fitting uses both all phases '
        'and the declared interface ±12 Å population. No neural encoder or predictor is trained.\n\n'
        '![Interface correspondence trajectories](plots/interface-correspondence-trajectories.png)\n\n'
        '[Metrics](tables/METRICS.md) · [Numerical results](tables/cluster-correspondence.csv)\n\n'
        'Plots are grouped into correspondence matrices, rich feature signatures and sparse matched spatial projections. '
        'Cluster numbers/colors from independent fits have no shared meaning; use the correspondence matrices. '
        'Spatial projections contain 64 sampled centers per snapshot and must not be interpreted as dense interface maps. '
        'Classical descriptors are reference partitions, not unique physical ground truth.\n')
    dest=Path(c['publication']); dest.mkdir(parents=True,exist_ok=True)
    for part in ('plots','tables'):
        shutil.copytree(out/part,dest/part,dirs_exist_ok=True)
    (dest/'technical').mkdir(exist_ok=True)
    for file in ('complete.json','metric-contract.json','descriptor-clustering.json'):
        shutil.copy2(out/'technical'/file,dest/'technical'/file)
    for part in ('metric-contracts','table-contracts'):
        shutil.copytree(out/'technical'/part,dest/'technical'/part,dirs_exist_ok=True)
    shutil.copy2(out/'README.md',dest/'README.md')
    print(f'Completed and published {len(targets)} comparisons: {dest}',flush=True)


def submit(config):
    c,parent=configuration(config); out=Path(c['output']); tech=out/'technical/queue'
    if (tech/'launch.json').exists(): raise FileExistsError('Queue already submitted')
    check_metric_docs(family='interface_cluster_correspondence')
    code=tech/'code'; repo=Path(__file__).resolve().parents[3]
    for directory in ('src','configs','docs/metrics'):
        shutil.copytree(repo/directory,code/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(repo/'machine.local.yaml',code/'machine.local.yaml')
    c['parent']=str(tech/'parent.json');write_json(c['parent'],parent)
    cfg=tech/'config.json';write_json(cfg,c)
    receipt=dict(state='submitting', config=str(cfg), jobs=[], code=str(code))
    write_json(tech/'launch.json',receipt)
    dependency=None
    for stage,hours in (('prepare',4),('compare',6)):
        script=tech/(stage+'.sbatch')
        cmd=[sys.executable,'-u','-m','src.research.spatial_vicreg_bias.correspondence',stage,'--config',str(cfg)]
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=SVB-interface-{stage}',
            '#SBATCH --partition=CPU',f'#SBATCH --cpus-per-task={c["workers"] if stage=="prepare" else 2}',
            '#SBATCH --mem=24G',f'#SBATCH --time={hours:02d}:00:00',f'#SBATCH --output={tech}/{stage}-%j.log',
            'set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OVITO_THREAD_COUNT=1 QT_QPA_PLATFORM=offscreen',
            'cd '+shlex.quote(str(code)),shlex.join(cmd)])+'\n')
        args=['sbatch','--parsable']
        if dependency: args+=['--dependency=afterok:'+dependency]
        job=subprocess.check_output(args+[str(script)],text=True).strip().split(';')[0]
        receipt['jobs'].append(dict(stage=stage,job=job,script=str(script)));write_json(tech/'launch.json',receipt)
        dependency=job
    receipt['state']='submitted';write_json(tech/'launch.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['submit','prepare','compare'])
    parser.add_argument('--config',required=True);args=parser.parse_args()
    globals()[args.stage](args.config)
