"""Export real held-out forecasts and recorded birth probabilities for inspection.

Execute this file directly with --bundle: scientific imports use the frozen
training implementation. No fitting, model selection or W&B runs are performed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def render(args):
    serialized = (args.output/'predictions.json').read_text()
    template = Path(__file__).with_name('viewer.html').read_text()
    fragment = template.replace('"__PREDICTION_DATA__"', serialized)
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError('Inline visualization exceeds1MB; reduce retained example paths')
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(fragment)
    print('Visualization:', args.fragment, len(fragment.encode()), 'bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fragment', type=Path, required=True)
    parser.add_argument('--stage', choices=['export', 'render', 'all'], default='all')
    args = parser.parse_args()
    if args.stage == 'render':
        render(args)
        return
    sys.path.insert(0, str(args.bundle))
    import numpy as np
    import torch
    from src.research.ni_latent_sim.data import load, read, root, sha
    from src.research.ni_latent_sim.model import make_model

    torch.set_num_threads(2)
    c = read(args.bundle/'config.json'); campaign = root(c)
    plan = read(campaign/'technical/plan.json'); data = load(c, 'data')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(plan_identity=plan['identity'], bundle=str(args.bundle),
                    producer_sha256=sha(Path(__file__)), online_runs_created=0,
                    selection='Four fixed-hash rows per existing regime, distinct ancestors within each regime; no model-error selection.',
                    forecast='Selected validation-NLL checkpoints; 32 stochastic latent paths; no future resets; 1ps-trained models rolled to6ps.',
                    uncertainty='Pointwise5–95% empirical quantiles of32 samples including diagonal Gaussian decoder noise; not physical MD continuations or confidence intervals.',
                    time='MD observed every0.1ps, with history[-0.3,-0.2,-0.1,0]; neural5fs outputs in first0.1ps and0.1ps thereafter. Decoder output at0 is retained.',
                    feature_error='Mean squared error of particle-mean decoded first128 features against the saved MD features, using frozen training mean/scale.',
                    birth_predictions='Unchanged stored CatBoost frozen-embedding probe probabilities; original held-out ancestry; four models aligned by exact row identities.',
                    checkpoints={}, birth_files={})
    chosen = []
    for regime in ['growing_front', 'liquid_liquid', 'original']:
        candidates = [i for i, r in enumerate(data['rows']) if r['role'] == 'test' and r['regime'] == regime]
        candidates.sort(key=lambda i: hashlib.sha256((data['rows'][i]['id']+'viewer-v1').encode()).hexdigest())
        seen = set(); selected = []
        for i in candidates:
            ancestor = data['rows'][i]['ancestor']
            if ancestor not in seen:
                selected.append(i); seen.add(ancestor)
            if len(selected) == 4:
                break
        if len(selected) != 4:
            raise ValueError('Need four held-out ancestors per recorded regime')
        chosen.extend(selected)
    mean, scale = data['mean'], data['scale']
    z = (data['x'][chosen]-mean)/scale
    tr = np.flatnonzero([r['role'] == 'train' for r in data['rows']])
    training = ((data['x'][tr, 3, :128]-mean[:128])/scale[:128]).double()
    pc_center = training.mean(0); centered = training-pc_center
    values, vectors = torch.linalg.eigh(centered.T@centered/(len(centered)-1))
    pc = vectors[:, -1].float(); pc *= torch.sign(pc[pc.abs().argmax()])
    pc_center = pc_center.float()
    metadata['pca'] = dict(fit_rows=len(tr), scope='Training-row present structures only',
                           variance_fraction=float(values[-1]/values.sum()),
                           center=pc_center.tolist(), component=pc.tolist())

    def project(a):
        return torch.stack((a[..., 273]*scale[273]+mean[273],
                            a[..., 260]*scale[260]+mean[260],
                            (a[..., :128]-pc_center)@pc), -1)

    def compact(a):
        return np.asarray(a, dtype=np.float64).round(4).tolist()

    times = np.r_[np.arange(21)*.005, np.arange(2, 61)*.1]
    truth = project(z).numpy()
    payload = dict(seeds=c['seeds'], particles=32,
                   features=[dict(key='angular', title='Inner-shell angular order, l = 6', unit='Angular power', index=273),
                             dict(key='radial', title='Radial density, bin 5', unit='Weighted neighbours', index=260),
                             dict(key='pc1', title='Structural embedding, PC1', unit='PC1 score', index=None)],
                   times=compact(times), observed_times=compact(np.arange(-3, 61)*.1),
                   native_times=compact(np.arange(61)*.1), cases=[], births={},
                   pc_variance=round(float(values[-1]/values.sum()), 4))
    for n, i in enumerate(chosen):
        row = data['rows'][i]
        payload['cases'].append(dict(id=row['id'], source=row['source'], ancestor=row['ancestor'],
                                     center=int(row['center']), frame=int(row['frame']), regime=row['regime'],
                                     truth=compact(truth[n]), predictions={}))
    metadata['rows'] = [dict(index=i, **data['rows'][i]) for i in chosen]
    h = z[:, :4].cuda()
    arms = [a for a in c['arms'] if a['train_steps'] == 10]
    for arm in arms:
        tech = campaign/'analyses'/arm['name']/'technical'
        checkpoint = tech/'best.pt'; done = read(tech/'complete.json')
        if sha(checkpoint) != done['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = make_model(arm).cuda().eval(); model.load_state_dict(saved['model'])
        key = ('response' if arm['response'] else 'values')+'-'+str(arm['seed'])
        with torch.no_grad():
            g = torch.Generator(device='cuda').manual_seed(742301)
            mu, lv = model.rollout(h, 60, 32, g)
            g = torch.Generator(device='cuda').manual_seed(742301)
            dense_mu, dense_lv = model.dense_rollout(h, 32, g)
            discrepancy = float((dense_mu[:, :, -1]-mu[:, :, 0]).abs().max())
            if discrepancy > 1e-6:
                raise ValueError('Dense and native outputs disagree for the same initial random stream')
            now, now_lv = model.emit(model.encode(h[:, -1]))
            start_mu = now[:, None, None].expand(-1, 32, 1, -1)
            start_lv = now_lv[:, None, None].expand(-1, 32, 1, -1)
            error = ((torch.cat((start_mu, mu), 2).mean(1)[..., :128].cpu()-z[:, 3:, :128])**2).mean(-1)
            mu = torch.cat((start_mu, dense_mu, mu[:, :, 1:]), 2).cpu()
            lv = torch.cat((start_lv, dense_lv, lv[:, :, 1:]), 2).cpu()
            rng = torch.Generator().manual_seed(921471)
            samples = mu+(.5*lv).exp()*torch.randn(mu.shape, generator=rng)
            observed = project(samples)
            conditional = project(mu)
            means = conditional.mean(1)
            quantiles = observed.quantile(torch.tensor([.05, .95]), dim=1)
        for n, row in enumerate(payload['cases']):
            row['predictions'][key] = dict(mean=compact(means[n]), lo=compact(quantiles[0, n]),
                                           hi=compact(quantiles[1, n]), paths=compact(conditional[n, :2]),
                                           error=compact(error[n]))
        metadata['checkpoints'][key] = dict(path=str(checkpoint), sha256=sha(checkpoint),
                                           selected_update=saved['update'], completed_updates=done['updates'],
                                           dense_native_maximum_difference=discrepancy)
        print('Forecasts exported:', key, flush=True)
        del model, mu, lv, samples, dense_mu, dense_lv
        torch.cuda.empty_cache()

    for threshold in [32, 64]:
        for lead in [1, 3, 6]:
            arrays = []
            for arm in arms:
                folder = campaign/'analyses'/f'nucleus{threshold}'/'analyses/fits-v1'/arm['name']/'compact_history'/f'lead-{lead}ps/catboost/technical'
                f = folder/'predictions.npz'
                if sha(f) != read(folder/'complete.json')['predictions_sha256']:
                    raise ValueError('Changed stored birth predictions')
                with np.load(f, allow_pickle=False) as packet:
                    packet = {k: packet[k] for k in packet.files}
                mask = np.isin(packet['ancestor'], plan['partition']['test'])
                arrays.append({k: v[mask] for k, v in packet.items()})
                metadata['birth_files'][f'{threshold}-{lead}-{arm["name"]}'] = dict(path=str(f), sha256=sha(f))
            first = arrays[0]
            for a in arrays[1:]:
                for field in ['id', 'ancestor', 'event', 'pair', 'label', 'weight']:
                    if not np.array_equal(first[field], a[field]):
                        raise ValueError('Birth comparison rows or weights differ: '+field)
            ancestors = sorted(set(first['ancestor'].tolist()))
            order = np.argsort(first['id']); records = []
            for i in order:
                records.append([str(first['id'][i])[:10], int(first['label'][i]),
                                round(float(first['weight'][i]), 7), ancestors.index(first['ancestor'][i]),
                                *[round(float(a['probability'][i]), 5) for a in arrays]])
            payload['births'][f'{threshold}-{lead}'] = dict(ancestors=ancestors, rows=records,
                columns=['id_prefix', 'label', 'weight', 'ancestor_index']+[
                    ('response' if a['response'] else 'values')+'-'+str(a['seed']) for a in arms],
                events=len(set(first['event'].tolist())), raw_counts=dict(Counter(map(int, first['label']))))
    serialized = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    (args.output/'predictions.json').write_text(serialized)
    metadata['payload_sha256'] = sha(args.output/'predictions.json')
    metadata['payload_bytes'] = len(serialized.encode())
    (args.output/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.stage != 'export':
        render(args)


if __name__ == '__main__':
    main()
