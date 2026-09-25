"""Separate onset, physical retention, 0.75 ps dynamics and noise diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.structural_state.data import Corpus, graph_arrays
from src.research.structural_state.evaluation import hazard_probe
from src.research.structural_state.model import block_error
from src.research.local_predictability.metrics import cumulative_risk, weighted_scores, threshold_at_fpr, hazard_loss
from src.research.trajectory_stability.spectrum import spectrum,source_weights,analyze
from .common import write_json,sha,remaining
from .data import targets,patches
from .model import Model,GraphBank
from .metrics import perturb_patch, horizon_index, HORIZONS_PS
from .train import encode


@torch.no_grad()
def infer_patches(model,patch_list,chunk,device,deadline=None):
    parts=[]
    for start in range(0,len(patch_list),chunk):
        remaining(deadline,60)
        bank=model.make_bank(model.inference_arrays(patch_list[start:start+chunk]),device)
        parts.append(model(bank.batch(np.arange(len(patch_list[start:start+chunk])))).cpu().numpy())
    return np.concatenate(parts)


def dense_dynamics(study,model,device,deadline):
    """Historical dense chart: descriptive transfer diagnostic, not model selection."""
    config=study.config;root=resolve_path(config['dense_root']);inputs=resolve_path(config['dense_inputs'])
    plan=json.loads((root/'technical/plan.json').read_text())
    manifest=json.loads((inputs/'manifest.json').read_text())
    if sha(root/'technical/plan.json')!=manifest['source_plan_sha256']:raise ValueError('Dense plan mismatch')
    values=[];source=[];atom=[];time=[];phase=[];split=[]
    for i,s in enumerate(plan['sources']):
        remaining(deadline,60);p=inputs/f'frame-{i:02d}.npz'
        if sha(p)!=manifest['files'][p.name]:raise ValueError(f'Dense input changed: {p}')
        with np.load(p) as data:
            a=dict(data);clean=[]
            for x,center in zip(patches(a),a['centers'],strict=True):
                keep=np.flatnonzero(np.linalg.norm(x,axis=1)<model.input_radius)
                keep=np.r_[center,keep[keep!=center]]
                clean.append(x[keep])
        values.append(infer_patches(model,clean,config['training']['microbatch'],device,deadline))
        folder=root/'technical/sources'/str(s['id'])
        receipt=json.loads((folder/'complete.json').read_text())
        if sha(folder/'observations.npz')!=receipt['hashes']['observations.npz']:raise ValueError('Dense metadata changed')
        with np.load(folder/'observations.npz') as data:
            n=len(clean);source.extend([s['id']]*n)
            atom.extend(np.tile(data['centers'],len(data['times_ps'])))
            time.extend(np.repeat(data['times_ps'],len(data['centers'])))
            phase.extend(data['labels'].ravel());split.extend([s['split']]*n)
    z=np.concatenate(values);source,atom,time,phase,split=map(np.asarray,[source,atom,time,phase,split])
    result=analyze(z,source,atom,time,np.flatnonzero(split=='train'),np.flatnonzero(split=='test'),
        lags_ps=[.75],domains={'noncrystalline':~np.isin(phase,[1,2,3])})
    result['scope']='Historical observed-coordinate cohort; may overlap fitting roots; descriptive, never used for selection. Relaxed-input arm is domain transfer.'
    return result,z


def run(study,name,device='cuda',deadline=None):
    folder=study.technical/'fits'/name;out=study.technical/'evaluations'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['identity']!=study.identity:raise ValueError('Evaluation changed')
        return
    corpus=Corpus(study);arm=study.arm(name);c=study.config;chunk=c['training']['microbatch']
    target,scalers,conditions,risk,sources=targets(corpus)
    saved=torch.load(folder/'best.pt',map_location=device,weights_only=False)
    if saved['identity']!=study.identity:raise ValueError('Checkpoint identity mismatch')
    model=study.make_model(saved['encoder_config'],arm,conditions.shape[1]).to(device)
    model.load_state_dict(saved['model']);model.eval()
    arrays=study.graph_arrays(arm)
    bank=model.make_bank(arrays,device)
    z=encode(model,bank,np.arange(len(corpus.records)),chunk).cpu().numpy()
    np.savez(out/'features.npz',exported=z)
    result=dict(identity=study.identity,arm=name,selected_step=saved['step'],checkpoint_sha256=sha(folder/'best.pt'))
    with torch.no_grad():
        all_logits=model.logits(torch.as_tensor(z,device=device),torch.as_tensor(conditions,device=device))
        probabilities=cumulative_risk(all_logits).cpu().numpy()
        tune,dev=risk['tune'],risk['development'];bins=corpus.targets['event_bin']
        result['primary_horizon_ps']=c['primary_horizon_ps']
        result['onset']={f'{t:g}':weighted_scores(bins[dev]<=k,probabilities[dev,k],sources[dev],
                            threshold_at_fpr(bins[tune]<=k,probabilities[tune,k],sources[tune]))
                         for k,t in enumerate(HORIZONS_PS)}
        result['hazard_nll']=float(source_weights(sources[dev])@hazard_loss(all_logits[dev],torch.as_tensor(bins[dev],device=device)).cpu().numpy())
        result['physical']={}
        for role,ix in corpus.split.items():
            zz=torch.as_tensor(z[ix],device=device);w=source_weights(sources[ix]);m={}
            for head in model.heads:
                pred=model.heads[head](zz).cpu().numpy()
                m[head]=float(w@np.square(pred-target[head][ix]).mean(1))
            result['physical'][role]=m
        np.savez(out/'predictions.npz',indices=dev,source=sources[dev],event_bin=bins[dev],risk=probabilities[dev],
            temperature=np.array([r['temperature_K'] for r in corpus.records])[dev],
            frame=np.array([r['frame'] for r in corpus.records])[dev],
            atom=np.array([r['center_atom_id'] for r in corpus.records])[dev])
    # Retrained probes test exported-state accessibility; their selection is
    # NLL, kept separate from the joint head selected on constrained tuning AP.
    result['probes']={}
    for kind in ('linear','mlp'):
        remaining(deadline,120)
        metrics,_,prediction=hazard_probe(z,conditions,corpus,c,kind,device,deadline)
        result['probes'][kind]=metrics
        np.savez(out/f'probe-{kind}.npz',**prediction)
    reference=spectrum(z[corpus.split['fit']],source_weights(sources[corpus.split['fit']]))
    result['dataset_spectrum']=spectrum(z,source_weights(sources))
    # One shared deterministic development subset and random fields per model.
    rng=np.random.default_rng(c['seed']+512);chosen=[]
    for source in np.unique(sources[corpus.split['development']]):
        ix=corpus.split['development'][sources[corpus.split['development']]==source]
        chosen.extend(rng.choice(ix,8,replace=False))
    chosen=np.asarray(chosen);all_patches=study.noise_patches(arm,arrays);clean=[all_patches[i] for i in chosen]
    clean_z=z[chosen];weights=source_weights(sources[chosen]);noise=[]
    for fraction in c['noise']['evaluation_rms_fractions']:
        # Reset across amplitudes; precisely the same displacement directions.
        rng=np.random.default_rng(c['seed']+513)
        changed,receipts=zip(*(perturb_patch(x,fraction,rng) for x in clean),strict=True)
        zz=infer_patches(model,list(changed),chunk,device,deadline)
        d2=np.square(zz.astype(float)-clean_z).sum(1)
        noise.append(dict(expected_relative_rms=fraction,
            input_rms_A=float(np.sqrt(weights@np.array([r['input_mse_A2'] for r in receipts]))),
            mean_spacing_A=float(weights@np.array([r['spacing_A'] for r in receipts])),
            input_rms_percent_of_spacing=float(100*np.sqrt(weights@np.array([r['input_relative_mse'] for r in receipts]))),
            embedding_noise_rms=float(np.sqrt(weights@d2/(2*reference['total_energy'])))))
    result['noise']=noise
    result['context_diagnostics']=study.additional_diagnostics(model,corpus,z,device,deadline)
    # Free resident fitting bank before the full dense inference.
    del bank;torch.cuda.empty_cache() if str(device).startswith('cuda') else None
    result['dynamics'],dense_z=dense_dynamics(study,model,device,deadline)
    np.save(out/'dense-embeddings.npy',dense_z)
    write_json(out/'metrics.json',result)
    write_json(out/'complete.json',dict(state='complete',identity=study.identity,metrics_sha256=sha(out/'metrics.json')))


def collect(study):
    rows=[];missing=[];predictions={}
    horizon=study.config['primary_horizon_ps'];index=horizon_index(horizon);key=f'{horizon:g}'
    for arm in study.config['arms']:
        name=arm['name'];p=study.technical/'evaluations'/name
        if not (p/'complete.json').exists():missing.append(name);continue
        receipt=json.loads((p/'complete.json').read_text())
        if receipt['identity']!=study.identity or sha(p/'metrics.json')!=receipt['metrics_sha256']:raise ValueError('Collector identity mismatch')
        m=json.loads((p/'metrics.json').read_text());onset=m['onset'][key];d=m['dynamics']['domains']['all']['lags']['0.75']
        primary=next(n for n in m['noise'] if n['expected_relative_rms']==.005)
        rows.append(dict(arm=name,selected_step=m['selected_step'],primary_horizon_ps=horizon,
            **{f'AP{t}':m['onset'][str(t)]['average_precision'] for t in (3,6,12)},
            prevalence=onset['prevalence'],primary_Brier=onset['brier'],recall_at_tuning_5pct_FPR=onset['recall'],
            development_FPR=onset['false_positive_rate'],physical_mse=m['physical']['development'][arm['input']],
            future_mse=m['physical']['development']['future'],noise_rms=primary['embedding_noise_rms'],
            input_noise_percent=primary['input_rms_percent_of_spacing'],jump_rms_075=d['rms_jump'],
            dataset_participation_rank=m['dataset_spectrum']['participation_rank'],
            movement_participation_rank=d['movement']['participation_rank'],movement_d95=d['movement']['d95'],
            outer_context_swap_rms=m['context_diagnostics'].get('outer_context_swap_rms')))
        with np.load(p/'predictions.npz') as a:predictions[name]=dict(a)
    snapshot_metric_docs(study.root,study.metric_family)
    if rows:
        with (study.root/'tables/comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Paired, temperature-stratified whole-root bootstrap. Multiplicity is
    # applied to existing source-balanced weights; duplicated roots stay duplicated.
    paired=[];reference=study.config.get('reference_arm','B-onset');base=predictions.get(reference)
    if base is not None:
        roots=np.unique(base['source']);temperature={s:base['temperature'][base['source']==s][0] for s in roots}
        rng=np.random.default_rng(study.config['seed']+901);draws=[]
        for _ in range(study.config['bootstrap']):
            sample=np.concatenate([rng.choice([s for s in roots if temperature[s]==t],
                size=sum(temperature[s]==t for s in roots),replace=True) for t in sorted(set(temperature.values()))])
            draws.append(source_weights(base['source'])*np.array([sum(sample==s) for s in base['source']]))
        y=base['event_bin']<=index
        for name,p in predictions.items():
            np.testing.assert_array_equal(p['indices'],base['indices'])
            for field in ('source','event_bin','temperature'):
                np.testing.assert_array_equal(p[field],base[field])
            values=[]
            for w in draws:
                if w[y].sum()==0:continue
                values.append(average_precision_score(y,p['risk'][:,index],sample_weight=w)-
                              average_precision_score(y,base['risk'][:,index],sample_weight=w))
            if not values:
                raise ValueError(f'No valid {horizon:g} ps bootstrap draws for {name}')
            lo,hi=np.quantile(values,[.025,.975])
            paired.append(dict(arm=name,reference=reference,horizon_ps=horizon,
                delta_AP_low=float(lo),delta_AP_high=float(hi),valid_draws=len(values)))
        with (study.root/'tables/paired-AP.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    baseline_path=study.technical/'descriptor-baselines.json'
    if baseline_path.exists():
        baselines=json.loads(baseline_path.read_text())
        if baselines['identity']!=study.identity:raise ValueError('Baseline identity changed')
    else:
        corpus=Corpus(study);_,_,conditions,_,_=targets(corpus);scores={}
        for domain in ('observed','relaxed','temperature_only'):
            x=(np.pad(corpus.geometry[domain],((0,0),(0,39))) if domain!='temperature_only'
               else np.zeros((len(corpus.records),128),np.float32))
            for kind in ('linear','mlp'):
                metrics,_,_=hazard_probe(x,conditions,corpus,study.config,kind,'cpu',None)
                scores[f'{domain}-{kind}']=metrics
        baselines=dict(identity=study.identity,scores=scores);write_json(baseline_path,baselines)
    baseline_rows=[dict(model=name,primary_horizon_ps=horizon,
        **{f'AP{t}':m['horizons'][str(float(t))]['average_precision'] for t in (3,6,12)},
        primary_Brier=m['horizons'][str(float(horizon))]['brier'],selected_step=m['best_step']) for name,m in baselines['scores'].items()]
    with (study.root/'tables/descriptor-baselines.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(baseline_rows[0]));writer.writeheader();writer.writerows(baseline_rows)
    write_json(study.technical/'collection.json',dict(complete=[r['arm'] for r in rows],missing=missing,paired=paired))
    first=next(iter(predictions.values()),None)
    coverage=(f'Development has {int((first["event_bin"]<=index).sum())} onset windows by {horizon:g} ps '
              f'among {len(first["indices"])} at-risk windows.' if first is not None else 'No completed development predictions.')
    lines=['# '+study.config.get('title','Robust onset encoder screen'),'',f'{len(rows)}/{len(study.config["arms"])} evaluations complete.',
        '', 'One seed; 25 fitting / 5 tuning / 15 reused development roots. '+coverage+' This is screening, not an untouched test.',
        '', f'Primary AP is source-weighted {horizon:g} ps sustained local-onset AP. Checkpoint selection uses tuning AP subject to present/current/future retention versus initialization. Frozen probes use NLL selection separately.',
        '', f'| Arm | AP3 | AP6 | AP12 | Brier at {horizon:g} ps | Noise RMS | Input noise / spacing | Jump RMS, 0.75 ps |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f'| {r["arm"]} | {r["AP3"]:.4f} | {r["AP6"]:.4f} | {r["AP12"]:.4f} | {r["primary_Brier"]:.4f} | {r["noise_rms"]:.4f} | {r["input_noise_percent"]:.3f}% | {r["jump_rms_075"]:.4f} |' for r in rows]
    lines += ['', 'Separate matched descriptor/temperature probes (tuning-NLL selected):', '',
        f'| Input/readout | AP3 | AP6 | AP12 | Brier at {horizon:g} ps |', '|---|---:|---:|---:|---:|']
    lines += [f'| {r["model"]} | {r["AP3"]:.4f} | {r["AP6"]:.4f} | {r["AP12"]:.4f} | {r["primary_Brier"]:.4f} |' for r in baseline_rows]
    lines += ['', 'Input spacing is the mean center-to-12-nearest-neighbor distance. Noise RMS is a 3D displacement, not per-coordinate sigma. Dense dynamics are descriptive observed-input transfer diagnostics, never selection data.',
        '', 'Incomplete: '+(', '.join(missing) if missing else 'none')]
    (study.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
