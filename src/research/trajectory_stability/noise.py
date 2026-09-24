"""Controlled coordinate-noise response on matched dense-trajectory observations."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from src.project_runtime.paths import resolve_path
from src.research.encoder_screen.common import checked, sha, write
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .noise_metrics import noise_response
from .spectrum import source_weights


def perturb(patches, atom_ids, centers, noise, sigma):
    values, nearest, mse, turnover = [], [], [], []
    for x, ids, center, epsilon in zip(patches, atom_ids, centers, noise, strict=True):
        if np.any(x[center]) or np.any(epsilon[center]):
            raise ValueError('Noise assay requires an exactly fixed tracked center')
        y = (x.astype(float)+sigma*epsilon).astype(np.float32)
        before = np.lexsort((ids, np.square(x.astype(float)).sum(1)))[:80]
        after = np.lexsort((ids, np.square(y.astype(float)).sum(1)))[:80]
        if len(after) != 80 or np.linalg.norm(y[after[-1]]) >= 10:
            raise ValueError('Perturbed nearest-80 context lacks a safe candidate halo')
        noncenter = np.arange(len(x)) != center
        mse.append(float(np.square(y.astype(float)-x).sum(1)[noncenter].mean()))
        turnover.append(1-len(np.intersect1d(before, after))/80)
        values.append(y); nearest.append(y[after])
    return dict(positions=np.concatenate(values), offsets=np.r_[0,np.cumsum([len(x) for x in values])],
                centers=np.asarray(centers), nearest80=np.stack(nearest),
                input_mse_A2=np.asarray(mse), nearest80_replacement=np.asarray(turnover))


def prepare(config, parent):
    root = resolve_path(config['output']); folder = root/'technical/inputs'
    dense = resolve_path(parent['dense']['root'])
    plan = json.loads((dense/'technical/plan.json').read_text())
    signature = dict(config=config, parent_config_sha256=sha(resolve_path(config['parent_config'])),
                     source_plan_sha256=sha(dense/'technical/plan.json'), producer_sha256=sha(__file__))
    if (folder/'manifest.json').exists():
        prior = json.loads((folder/'manifest.json').read_text())
        if prior['signature'] != signature:
            raise ValueError('Noise input identity changed; use a fresh output')
        for name,h in prior['files'].items():checked(folder/name,h)
        return prior
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config['seed'])
    patches, ids, centers, noise, records, source_inputs = [], [], [], [], [], {}
    for index, source in enumerate(plan['sources']):
        if source['split'] != 'test':continue
        old = dense/'technical/sources'/str(source['id'])
        receipt = json.loads((old/'complete.json').read_text())
        for name in ('positions.npy','atom_ids.npy','observations.npz'):
            checked(old/name, receipt['hashes'][name]); source_inputs[str(old/name)] = receipt['hashes'][name]
        positions, atom = np.load(old/'positions.npy', mmap_mode='r'), np.load(old/'atom_ids.npy', mmap_mode='r')
        with np.load(old/'observations.npz') as obs:
            np.testing.assert_allclose(np.diff(obs['times_ps']), .75, atol=1e-8, rtol=0)
            for ci, center_id in enumerate(obs['centers']):
                frames = np.sort(rng.choice(len(obs['frames'])-1, config['origins_per_atom'], replace=False))
                for frame in frames:
                    row = int(frame*len(obs['centers'])+ci)
                    a,b = obs['offsets'][row:row+2]; x = np.array(positions[a:b]); atom_ids = np.array(atom[a:b])
                    for draw in range(config['draws']):
                        epsilon = rng.normal(size=x.shape); center = int(obs['center_indices'][row]); epsilon[center] = 0
                        patches.append(x); ids.append(atom_ids); centers.append(center); noise.append(epsilon)
                        records.append(dict(source=source['id'], source_index=index, row=row,
                            next_row=row+len(obs['centers']), atom=int(center_id), frame=int(obs['frames'][frame]),
                            time_ps=float(obs['times_ps'][frame]), draw=draw, temperature_K=source['temperature_K'],
                            noncrystalline=bool(obs['labels'][row] not in (1,2,3))))
    write(folder/'records.json', records)
    manifest = dict(signature=signature, source_inputs=source_inputs, frames=[], files={'records.json':sha(folder/'records.json')})
    for i, sigma in enumerate([0.,0.]+config['sigma_A']):
        arrays = perturb(patches, ids, centers, noise, sigma)
        name = f'frame-{i:02d}.npz'; np.savez(folder/name, **arrays)
        manifest['frames'].append(dict(frame_index=i, material='Al', sigma_A=sigma,
            role='clean' if i==0 else 'repeat' if i==1 else 'noisy', count=len(records)))
        manifest['files'][name] = sha(folder/name)
    write(folder/'manifest.json', manifest)
    return manifest


def v6_producer(config, parent):
    """Reconstruct exact inference dependencies, checked against saved training hashes."""
    root = resolve_path(config['output'])/'technical/code-v6'
    dense = resolve_path(parent['dense']['root']); frozen = dense/'technical/code'
    plan = json.loads((dense/'technical/plan.json').read_text())
    if not root.exists():
        root.mkdir(parents=True)
        archive = subprocess.check_output(['git','archive',config['v6_commit'],'src'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter='data')
        for name in ('src/data/structural_pretraining/batches.py','src/research/trajectory_stability/encode.py'):
            shutil.copy2(frozen/name, root/name)
    result = {}
    for record in plan['checkpoints'].values():
        for name,h in record['identity']['implementation']['files'].items():
            if name.startswith('src/models/') or name.endswith(('/batches.py','/compilation.py')):
                checked(root/name,h); result[str(root/name)] = h
    path = root/'src/research/trajectory_stability/encode.py'
    checked(path,sha(frozen/'src/research/trajectory_stability/encode.py')); result[str(path)] = sha(path)
    return root, result


def tasks(config, parent):
    root = resolve_path(config['output']); inputs = root/'technical/inputs'
    for entry in parent['dense']['extra_exports']:
        receipt = json.loads((resolve_path(entry['root'])/'complete.json').read_text())
        task = dict(receipt['task'], inputs=str(inputs), destination=str(root/'technical/exports'/entry['name']))
        task['model_keys'] = {rep:'dense-current--'+entry['name']+'--'+rep for rep in receipt['extraction']['representations']}
        yield entry['name'], task, Path(__file__).resolve().parents[1]/'encoder_screen/native.py'
    producer, files = v6_producer(config,parent)
    dense = resolve_path(parent['dense']['root']); plan = json.loads((dense/'technical/plan.json').read_text())
    for name in ('mace','gatr'):
        record = plan['checkpoints'][name]
        yield 'v6-'+name, dict(kind='v6', producer=str(producer), producer_files=files,
            checkpoint=str(Path(record['path']).resolve()), checkpoint_sha256=record['sha256'],
            architecture=name, scale=plan['scale'], batch_size=32, inputs=str(inputs),
            destination=str(root/'technical/exports'/('v6-'+name)), model_keys={name:'dense-v6--'+name}), Path(__file__).with_name('noise_worker.py')
    paths = ['src/analysis/liquid_structure.py','src/data/structural_pretraining/prepare.py','src/data/predictive_memory/targets.py']
    yield 'descriptors', dict(kind='descriptors',producer=str(Path.cwd()),producer_files={str(Path(p).resolve()):sha(p) for p in paths},
        inputs=str(inputs),destination=str(root/'technical/exports/descriptors'),
        model_keys={name:'dense-v6--'+name for name in ['tda','soap','bond_order','radial','angular']}), Path(__file__).with_name('noise_worker.py')


def infer(config, parent):
    root = resolve_path(config['output'])
    for name, task, driver in tasks(config,parent):
        driver = driver.resolve(); task['inference_driver_sha256'] = sha(driver)
        path = root/'technical/tasks'/f'{name}.json'; destination = Path(task['destination'])
        if path.exists() and json.loads(path.read_text()) != task:
            raise ValueError(f'Noise extraction task changed: {name}')
        write(path, task)
        receipt = destination/'complete.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['task_sha256'] != sha(path):raise ValueError('Changed completed noise task')
            for f,h in saved['files'].items():checked(destination/f,h)
            continue
        command = [sys.executable,'-u',str(driver),'--record',str(path)]
        if task['kind'] in ('geometry','geoframe'):command.append('--static-only')
        destination.mkdir(parents=True, exist_ok=True)
        with (destination/'inference.log').open('w') as log:
            subprocess.run(command,cwd=task['producer'],stdout=log,stderr=subprocess.STDOUT,check=True)
        expected = len(config['sigma_A'])+2
        outputs = list(destination.glob('frame-*.npz'))
        if len(outputs) != expected:raise ValueError(f'Incomplete noise extraction: {name}')
        write(receipt, dict(state='complete',task_sha256=sha(path),files={p.name:sha(p) for p in outputs}))
        print('noise export complete',name,flush=True)


def clean_history(parent, task, records, key):
    """Join cached temporal states using the exact selected source and row IDs."""
    clean, future = [], []
    sources = np.array([r['source_index'] for r in records])
    for index in np.unique(sources):
        rows = np.flatnonzero(sources == index)
        if task['kind'] in ('geoframe','geometry'):
            entry = next(e for e in parent['dense']['extra_exports'] if task['name'] == e['name'])
            folder = resolve_path(entry['root']); path = folder/'embeddings'/f'frame-{index:02d}.npz'
            checked(path,json.loads((folder/'complete.json').read_text())['feature_files'][path.name])
            with np.load(path) as f: values = f[key]
        else:
            folder = resolve_path(parent['dense']['root'])/'technical/sources'/str(records[rows[0]]['source'])
            if key in ('mace','gatr'):
                path = folder/f'{key}.npy'; checked(path,json.loads((folder/f'{key}-verification.json').read_text())['features_sha256']); values = np.load(path)
            else:
                path = folder/'observations.npz'; checked(path,json.loads((folder/'complete.json').read_text())['hashes']['observations.npz'])
                with np.load(path) as f:
                    values = f['geometry'][:,:32] if key=='radial' else f['geometry'][:,64:80] if key=='angular' else f[key]
        clean.extend(values[[records[i]['row'] for i in rows]])
        future.extend(values[[records[i]['next_row'] for i in rows]])
    # Preparation records are in source-index order, with repeated noise draws adjacent.
    if np.any(np.diff(sources)<0):raise ValueError('Noise records lost source-major order')
    return np.asarray(clean),np.asarray(future)


def report(config,parent):
    root = resolve_path(config['output']); base = resolve_path(parent['output'])
    records = json.loads((root/'technical/inputs/records.json').read_text())
    source = np.array([r['source'] for r in records]); domains={'all':np.ones(len(source),bool),
        'noncrystalline':np.array([r['noncrystalline'] for r in records])}
    rows = []
    for name, task, _ in tasks(config,parent):
        directory = Path(task['destination'])
        clean, repeat = np.load(directory/'frame-00.npz'), np.load(directory/'frame-01.npz')
        for key, model in task['model_keys'].items():
            reference = json.loads((base/'technical'/f'{model}.json').read_text())['metrics']['reference']['total_energy']
            before, after = clean_history(parent,task,records,key)
            for j,sigma in enumerate(config['sigma_A'],2):
                with np.load(directory/f'frame-{j:02d}.npz') as noisy, np.load(root/'technical/inputs'/f'frame-{j:02d}.npz') as inputs:
                    for domain, mask in domains.items():
                        metric = noise_response(clean[key][mask],noisy[key][mask],repeat[key][mask],
                            (after-before)[mask],reference,source[mask],inputs['input_mse_A2'][mask],sigma)
                        weights=source_weights(source[mask])
                        metric.update(replay_rms=float(np.sqrt(weights@np.square(clean[key][mask]-before[mask]).sum(1)/(2*reference))),
                            nearest80_replacement_fraction=float(weights@inputs['nearest80_replacement'][mask]))
                        rows.append(dict(model=model,domain=domain,**metric))
        clean.close();repeat.close()
    for part in ('tables','plots'):(root/part).mkdir(exist_ok=True)
    snapshot_metric_docs(root,'embedding_noise')
    with (root/'tables/noise-response.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    old=list(csv.DictReader((base/'tables/comparison.csv').open()))
    for entry in old:
        metric=next(r for r in rows if r['model']==entry['model'] and r['domain']=='all' and r['sigma_A']==config['primary_sigma_A'])
        entry.update({k:metric[k] for k in ['sigma_A','response_rms','response_p95','sensitivity_per_A','repeat_rms',
            'temporal_rms_matched_075','noise_to_temporal_ratio','replay_rms']})
    with (root/'tables/comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(old[0]));writer.writeheader();writer.writerows(old)
    write(root/'technical/metrics.json',dict(rows=rows,parent_report_sha256=sha(base/'technical/complete.json')))
    plot(root,rows,old)
    text=['# Embedding response to controlled input noise','',
        'All temporal comparisons retain **0.75 ps**. Coordinate-noise response uses the',
        'same 320 original observations (32 per evaluation source), four Gaussian draws',
        'per observation and identical per-atom perturbations across representations.',
        'The tracked center stays fixed. The finite candidate neighborhood stays fixed;',
        'nearest-80 membership, native crops, graph edges and spatial weights are rebuilt.',
        'This measures the response of the complete input-to-embedding pipeline.', '',
        f'Primary noise: **sigma = {config["primary_sigma_A"]:g} Angstrom per Cartesian coordinate**.',
        'Expected 3D RMS displacement is sqrt(3)*sigma. Noise RMS is normalized by',
        'the same clean fitting-reference spread as the temporal results. Noise / motion',
        'compares its RMS with natural 0.75 ps motion at the **same sampled origins**.',
        'It is a ratio of response magnitudes, not a fraction of natural motion caused by noise.', '',
        '## Encoder and descriptor comparison','',
        '| Representation | Dataset rank | Movement rank | d95 | 0.75 ps RMS | Noise RMS | Noise / matched motion |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in old:
        if 'projector' in r['model']:continue
        ratio=f'{r["noise_to_temporal_ratio"]:.3f}' if r['noise_to_temporal_ratio'] is not None else 'undefined'
        text.append(f'| {r["label"]} | {float(r["dataset_rank"]):.3f} | {float(r["movement_rank"]):.3f} | {r["movement_d95"]} | {float(r["rms_jump"]):.3f} | {r["response_rms"]:.4f} | {ratio} |')
    text+=['','The 0.75 ps RMS column retains the full trajectory population; the ratio uses',
        'the paired sampled-origin denominator exported as `temporal_rms_matched_075`.',
        'Projector rows, repeated-input floors, clean-cache replay differences and all',
        'noise levels are retained in the CSVs. MACE on observed inputs retains the',
        'input-domain qualification in the parent report. Low noise response alone does',
        'not demonstrate information retention or useful physical responsiveness.','',
        '- [Combined table, including projectors](tables/comparison.csv)',
        '- [All noise levels and noncrystalline results](tables/noise-response.csv)',
        '- [Frozen definitions and limits](tables/METRICS.md)','',
        '![Noise response curves](plots/noise.png)','']
    (root/'RESULTS.md').write_text('\n'.join(text))
    write(root/'technical/complete.json',dict(state='complete',models=len(old),
        config_sha256=sha(root/'technical/config.json'),files={p.name:sha(p) for p in (root/'tables').iterdir()}))


def plot(root,rows,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for entry in comparison:
        if 'projector' in entry['model']:continue
        selected=[r for r in rows if r['model']==entry['model'] and r['domain']=='all']
        x=[r['sigma_A'] for r in selected]
        axes[0].plot(x,[r['response_rms'] for r in selected],'o-',label=entry['label'],markersize=3)
        axes[1].plot(x,[r['noise_to_temporal_ratio'] for r in selected],'o-',markersize=3)
    for ax in axes:ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Gaussian sigma per coordinate (Angstrom)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Reference-normalized noise RMS');axes[0].legend(fontsize=6,ncol=2)
    axes[1].set_ylabel('Noise RMS / matched natural 0.75 ps RMS');axes[1].axhline(1,color='gray',linestyle='--')
    fig.savefig(root/'plots/noise.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['all','prepare','infer','report'],default='all')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());parent=json.loads(resolve_path(config['parent_config']).read_text())
    root=resolve_path(config['output']);root.mkdir(parents=True,exist_ok=True)
    path=root/'technical/config.json'
    if path.exists() and json.loads(path.read_text()) != config:raise ValueError('Noise config changed; use a fresh output')
    if (root/'technical/complete.json').exists():raise FileExistsError('Preserve completed noise result; use a fresh output')
    write(path,config)
    if args.stage in ('all','prepare'):prepare(config,parent)
    if args.stage in ('all','infer'):infer(config,parent)
    if args.stage in ('all','report'):report(config,parent)
