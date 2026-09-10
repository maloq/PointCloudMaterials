"""Frozen-checkpoint diagnosis of the user-stopped September 9 temporal pilot."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from src.analysis.mace_temporal import analyze, ridge, scores, effective_rank
from src.data_utils.mace_history import Histories
from src.data_utils.temporal_campaign import write_json
from src.experiment_runner.tracking import tracked_run
from src.training_methods.mace_objective import variance_covariance
from src.training_methods.mace_temporal import TemporalLearner, encode


@torch.no_grad()
def frame_features(model, data, rows, size):
    result = []
    for start in range(0, len(rows), size):
        x, _, material = data.get(rows[start:start+size])
        z = model.encoder.mace(x.flatten(0, 1), material.repeat_interleave(x.shape[1]))
        result.append(z.reshape(len(x), x.shape[1], 256).cpu().numpy())
    return np.concatenate(result)


def gradient_audit(model, data, rows, cfg):
    model.train()
    x, y, m = data.get(rows)
    with torch.no_grad():
        z = encode(model, x, m, cfg['microbatch_size'])
    z.requires_grad_(True)
    variance, covariance = variance_covariance(z)
    terms = dict(tda=(model.tda(z)-y).square().mean(), variance=25*variance, covariance=covariance)
    terms['regularization'] = terms['variance'] + terms['covariance']
    dz = {name: torch.autograd.grad(value, z, retain_graph=True)[0] for name, value in terms.items()}
    def stats(vectors):
        norms = {name: float(vector.norm()) for name, vector in vectors.items()}
        cosine = float(torch.nn.functional.cosine_similarity(vectors['tda'].flatten(), vectors['regularization'].flatten(), dim=0))
        return dict(norms=norms, regularization_to_tda=norms['regularization']/norms['tda'], tda_regularization_cosine=cosine)
    result = dict(losses={name: float(value.detach()) for name, value in terms.items()}, embedding_gradients=stats(dz))
    # Apply full-batch latent derivatives to one microbatch without optimizer steps.
    size = cfg['microbatch_size']
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    parameter_gradients = {}
    for name in ('tda', 'regularization'):
        # Compiled radial backward donates its buffers: use a fresh forward for
        # each measured derivative, exactly as in training's single-use replay.
        actual = model.encoder(x[:size], m[:size])
        gradients = torch.autograd.grad(actual, parameters, grad_outputs=dz[name][:size])
        parameter_gradients[name] = torch.cat([g.flatten() for g in gradients])
    result['encoder_parameter_gradients_microbatch'] = stats(parameter_gradients)
    model.eval()
    return result


def diagnose(cfg, reuse_probes):
    out = Path(cfg['output'])
    directory = out/'diagnosis'
    directory.mkdir(exist_ok=True)
    if not (out/'stop_request.json').exists():
        raise RuntimeError('This diagnostic is for the explicitly stopped temporal pilot.')
    if not reuse_probes:
        analyze(cfg)  # Reuse maintained frozen probes and history interventions unchanged.
    torch.cuda.empty_cache()
    with np.load(out/'scaling.npz') as archive:
        scaling = dict(archive)
    data = Histories(cfg, scaling)
    train = np.random.default_rng(cfg['seed']+1).choice(data.indices['train'], cfg['analysis']['probe_train_anchors'], replace=False)
    val = data.indices['val']
    train_y = data.targets[torch.as_tensor(train, device='cuda'), 0].cpu().numpy()
    val_y = data.targets[torch.as_tensor(val, device='cuda'), 0].cpu().numpy()
    materials = data.materials[val]
    model = TemporalLearner(cfg).cuda().eval()
    audit_rows = np.random.default_rng(cfg['seed']+2).choice(data.indices['train'], cfg['batch_size'], replace=False)
    gradients, controls, spectra, attention = {}, {}, {}, {}
    for name, filename in (('initial', 'initial.pt'), ('trained', 'best.pt')):
        saved = torch.load(out/filename, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        gradients[name] = gradient_audit(model, data, audit_rows, cfg)
        train_f = frame_features(model, data, train, cfg['microbatch_size'])
        val_f = frame_features(model, data, val, cfg['microbatch_size'])
        for mode, a, b in (('anchor', train_f[:, -1], val_f[:, -1]), ('mean', train_f.mean(1), val_f.mean(1))):
            predicted = ridge(a, train_y, b, cfg['analysis']['ridge_alpha'])
            controls[name+'_mace_'+mode] = scores(val_y, predicted, materials)
        spectra[name] = {material: dict(
            frame_variation_over_between_patch_variation=float(np.square(val_f[materials == i]-val_f[materials == i, -1:]).mean()/val_f[materials == i, -1].var(0).mean()),
            anchor_effective_rank=effective_rank(val_f[materials == i, -1]))
            for i, material in enumerate(('Al', 'Mg', 'Ta'))}
        with torch.no_grad():
            f = torch.from_numpy(val_f[:128]).cuda()
            times = (model.encoder.frame_offsets_ps/model.encoder.time_scale_ps)[:, None]
            tokens = model.encoder.frame_projection(f)+model.encoder.time_embedding(times)[None]
            weights = []
            for block in model.encoder.blocks:
                normalized = block.norm1(tokens)
                _, weight = block.self_attn(normalized, normalized, normalized, need_weights=True, average_attn_weights=False)
                weights.append(weight[:, :, -1].mean((0, 1)).cpu().tolist())
                tokens = block(tokens)
            attention[name] = weights
        print('DIAGNOSIS', name, json.dumps(gradients[name]), flush=True)
    predictions = np.load(out/'analysis/embeddings.npz')['head_predictions']
    train_all = data.targets[torch.as_tensor(data.indices['train'], device='cuda'), 0].cpu().numpy()
    raw = data.raw_targets[data.indices['train'], 0].astype(np.float64)
    eigen = np.linalg.eigvalsh(np.cov(raw.T))[::-1].clip(0)
    cumulative = np.cumsum(eigen)/eigen.sum()
    pca = dict(cumulative_raw_variance={str(k): float(cumulative[k-1]) for k in (1, 2, 4, 8, 16, 32)},
        component_std=scaling['tda_std'].tolist(), train_whitened_variance=train_all.var(0).tolist(),
        val_component_mse=np.square(predictions-val_y).mean(0).tolist(),
        train_component_mean=train_all.mean(0).tolist(), val_component_mean=val_y.mean(0).tolist(),
        val_component_variance=val_y.var(0).tolist())
    baseline = np.stack([train_all[data.materials[data.indices['train']] == m].mean(0) for m in materials])
    by_material = {}
    for i, name in enumerate(('Al', 'Mg', 'Ta')):
        mask = materials == i
        target, pred = val_y[mask], predictions[mask]
        by_material[name] = dict(
            mean_baseline_mse=float(np.square(baseline[mask]-target).mean()),
            head_mse=float(np.square(pred-target).mean()),
            validation_variance=float(target.var(0).mean()),
            train_to_validation_mean_shift_mse=float(np.square(baseline[mask][0]-target.mean(0)).mean()),
            component_mse=np.square(pred-target).mean(0).tolist(),
            first_four_mse=float(np.square(pred[:, :4]-target[:, :4]).mean()),
            last_sixteen_mse=float(np.square(pred[:, 16:]-target[:, 16:]).mean()))
    rows = [json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    first, last = rows[0]['train'], rows[-1]['train']
    progress = dict(first_epoch=first, last_epoch=last,
        tda_loss_reduction=first['tda_mse']-last['tda_mse'],
        regularization_loss_reduction=(first['loss']-first['tda_mse'])-(last['loss']-last['tda_mse']))
    result = dict(gradients=gradients, frozen_mace_controls=controls, feature_spectra=spectra,
                  anchor_attention_weights_first128_Al=attention, target_scaling=pca, by_material=by_material,
                  training_progress=progress, training_source_trajectories=sum(r['split']=='train' for r in data.records))
    write_json(directory/'diagnostics.json', result)
    paired = json.loads(Path(cfg['paired_manifest']).read_text())['shards']
    displacements = {}
    for i, name in enumerate(('Al', 'Mg', 'Ta')):
        groups = []
        for record in paired:
            if record['material'] == i and record['split'] == 'train':
                clouds = np.load(Path(record['directory'])/'clouds.npy', mmap_mode='r')
                groups.append(np.linalg.norm(clouds[:, 4, 1:].astype(np.float32)-clouds[:, 0, 1:].astype(np.float32), axis=-1).ravel())
        values = np.concatenate(groups)
        displacements[name] = dict(mean_A=float(values.mean()), median_A=float(np.median(values)), p90_A=float(np.quantile(values, .9)))
    write_json(directory/'hot_relaxed_displacements.json', displacements)
    print('DIAGNOSIS_COMPLETE', json.dumps(result), flush=True)


def figures(cfg):
    import matplotlib.pyplot as plt
    out = Path(cfg['output'])
    directory = out/'diagnosis'
    d = json.loads((directory/'diagnostics.json').read_text())
    a = json.loads((out/'analysis/metrics.json').read_text())
    rows = [json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = [r['epoch'] for r in rows]
    axes[0].plot(epochs, [r['train']['tda_mse'] for r in rows], label='TDA prediction')
    axes[0].plot(epochs, [r['train']['loss']-r['train']['tda_mse'] for r in rows], label='Weighted regularization')
    axes[0].set(xlabel='Completed epoch', ylabel='Loss contribution (log scale)', yscale='log', title='Most optimization went to regularization')
    axes[0].legend()
    names = ['MACE anchor', 'MACE mean of 5', 'Initial transformer', 'Trained transformer']
    values = [d['frozen_mace_controls']['initial_mace_anchor']['mse'],
              d['frozen_mace_controls']['initial_mace_mean']['mse'],
              a['scores']['initial_ridge']['mse'], a['scores']['trained_ridge']['mse']]
    axes[1].barh(names, values, color=['#64748b', '#15803d', '#2563eb', '#b91c1c'])
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Validation TDA MSE (lower is better)', title='Identical frozen ridge probes')
    for i, value in enumerate(values):
        axes[1].text(value+.01, i, f'{value:.3f}', va='center')
    axes[1].set_xlim(0, 1.25)
    fig.tight_layout()
    fig.savefig(directory/'failure_diagnosis.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--reuse-probes', action='store_true', help='Read already completed analysis/ probe artifacts.')
    parser.add_argument('--figures-only', action='store_true', help='Render plots from completed diagnostic JSON without model inference.')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    with tracked_run(Path(cfg['output'])/'diagnosis', kind='analysis', configs=[Path(args.config)],
                     command=[sys.executable, *sys.argv], question='Why did the stopped temporal MACE pilot learn little relaxed topology?'):
        if not args.figures_only:
            diagnose(cfg, args.reuse_probes)
        figures(cfg)


if __name__ == '__main__':
    main()
