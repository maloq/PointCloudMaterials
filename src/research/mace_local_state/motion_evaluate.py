"""Source-held-out motion, information and original-pair stability measurements."""
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from .motion import specifications, time_differences, projection_residual, projector_distance
from .motion_data import load, atomic_npz, check_deadline
from .motion_train import load_model
from .smooth import FAMILIES, jump_metrics


def source_mean(values, source, mask=None):
    if mask is None: mask = np.ones(len(values),dtype=bool)
    ids = np.unique(source[mask])
    if len(ids)<1: raise ValueError('Empty conditional source population')
    return ids,np.stack([values[mask & (source==i)].mean(0) for i in ids])


def interval(values, config, *, rms=False):
    rng = np.random.default_rng(config['seed'])
    values = np.asarray(values)
    samples = values[rng.integers(len(values),size=(config['bootstrap_draws'],len(values)))].mean(1)
    if rms: samples = np.sqrt(samples)
    return np.quantile(samples,[.025,.975]).tolist()


def physical_metrics(pred, target, low, source):
    error = (pred-target)**2
    result = {}
    for subset,mask in [('all',np.ones(len(source),bool)),('low_order',low)]:
        _,by_source = source_mean(np.stack([error[:,s].mean(1) for s in FAMILIES.values()],-1),source,mask)
        result[subset] = {k:float(v) for k,v in zip(FAMILIES,by_source.mean(0),strict=True)}
        result[subset+'_per_source'] = by_source.tolist()
    return result


def independent_probes(z, data, config):
    """Train-only scaling, validation ridge selection, additional unsupervised observables."""
    nt = z.shape[1];flat = z.reshape(-1,z.shape[-1]);split = np.repeat(data['split'],nt)
    y = data['extra_observables'].reshape(-1,4).astype(float)
    train = split==0;val = split==1;test = split==2
    weights = np.repeat(data['weights'],nt)[train].astype(float);weights /= weights.sum()
    xm = weights@flat[train];xs = np.sqrt(weights@((flat[train]-xm)**2))
    if np.any(xs<=1e-10): raise FloatingPointError('Collapsed channel in independent physical probe')
    ym = weights@y[train];ys = np.sqrt(weights@((y[train]-ym)**2))
    if np.any(ys<=1e-10): raise FloatingPointError('Degenerate additional physical observable')
    xx = (flat-xm)/xs;yy = (y-ym)/ys
    scores = [];models = []
    source = np.repeat(data['source_id'],nt)
    for alpha in config['probe_alphas']:
        model = Ridge(alpha=alpha).fit(xx[train],yy[train],sample_weight=weights*train.sum())
        _,errors = source_mean((model.predict(xx[val])-yy[val])**2,source[val])
        scores.append(float(errors.mean()));models.append(model)
    chosen = int(np.argmin(scores));model = models[chosen]
    result = dict(alpha=config['probe_alphas'][chosen],names=['nearest_distance','mean_radius79','std_radius79','shape_anisotropy'])
    for name,mask in [('validation',val),('development_test',test)]:
        _,errors = source_mean((model.predict(xx[mask])-yy[mask])**2,source[mask])
        result[name] = dict(normalized_mse=errors.mean(0).tolist(),mean=float(errors.mean()))
    return result


def local_direction_test(z, data, split, config):
    """Local bases fitted ONLY to training increments from other source groups."""
    nt = z.shape[1]
    delta = np.diff(z,axis=1).reshape(-1,z.shape[-1])
    current_physics = data['target'][:,:-1,:16].reshape(-1,16)
    ss = np.repeat(data['split'],nt-1);group = np.repeat(data['lineage_id'],nt-1)
    train = np.flatnonzero(ss==0);held = np.flatnonzero(ss==split)
    rng = np.random.default_rng(config['seed'])
    train = np.concatenate([rng.choice(train[group[train]==g],min(16,int(np.sum(group[train]==g))),replace=False)
        for g in np.unique(group[train])])
    held = np.sort(rng.choice(held,min(len(held),config['local_basis_samples']),replace=False))
    count = min(len(train),1024)
    _,neighbors = cKDTree(current_physics[train]).query(current_physics[held],k=count,workers=1)
    curves = {4:[],8:[]};energies = [];counts = []
    stability = []
    for row,query in enumerate(held):
        candidates = train[neighbors[row]];chosen = [];seen = set()
        for candidate in candidates:
            g = int(group[candidate])
            if g != group[query] and g not in seen:
                chosen.append(candidate);seen.add(g)
                if len(chosen)==config['local_basis_neighbors']: break
        if len(chosen)<config['local_basis_minimum_groups']:
            raise ValueError(f'Only {len(chosen)} independent training groups for local direction test')
        increments = delta[chosen].astype(float)
        _,_,basis = np.linalg.svd(increments,full_matrices=False)
        e = float(delta[query]@delta[query]);energies.append(e);counts.append(len(chosen))
        for rank in curves:
            q = basis[:rank]
            curves[rank].append(float(np.sum((q@delta[query])**2)))
        # Train-neighborhood perturbation: two disjoint sets of source groups.
        _,_,a = np.linalg.svd(increments[::2],full_matrices=False)
        _,_,b = np.linalg.svd(increments[1::2],full_matrices=False)
        rank = min(4,len(a),len(b))
        stability.append(float(np.sum((a[:rank]@b[:rank].T)**2)/rank))
    if sum(energies)<=1e-12: raise FloatingPointError('No energy in local-direction test')
    return dict(samples=len(held),minimum_training_groups=min(counts),
        explained_energy={str(k):float(sum(v)/sum(energies)) for k,v in curves.items()},
        train_neighborhood_subspace_overlap=float(np.mean(stability)),
        caveat='Physical-label neighborhoods are an evaluation diagnostic, not deployed encoder inputs.')


def sequence_metrics(z, q, data, split, reference_z, paired, config):
    ids = np.flatnonzero(data['split']==split);zz = z[ids]
    nt = zz.shape[1];low = data['raw_target'][ids,:,4]<config['low_order_threshold']
    source = np.repeat(data['source_id'][ids],nt-1)
    times = torch.as_tensor(data['time_ps'][ids]);inc,vel,acc,bend = time_differences(torch.as_tensor(zz),times)
    inc = inc.numpy();vel = vel.numpy();acc = acc.numpy();bend = bend.numpy()
    basis = q[ids]
    residual = projection_residual(torch.as_tensor(inc.reshape(-1,zz.shape[-1])),
        torch.as_tensor(basis[:,:-1].reshape(-1,zz.shape[-1],basis.shape[-1]))).numpy()
    drift = projector_distance(torch.as_tensor(basis[:,:-1].reshape(-1,zz.shape[-1],basis.shape[-1])),
        torch.as_tensor(basis[:,1:].reshape(-1,zz.shape[-1],basis.shape[-1]))).numpy()
    dt = np.diff(data['time_ps'][ids],axis=1).ravel();energy = np.sum(inc**2,axis=-1).ravel()
    result = [];reference = paired['reference_pair_ids']
    for subset in ('all','low_order'):
        anchors = 2*reference
        mask = np.ones(len(energy),bool)
        triple_mask = np.ones((len(ids),nt-2),bool)
        if subset=='low_order':
            anchors = anchors[paired['raw_target'][anchors,4]<config['low_order_threshold']]
            mask = (low[:,:-1]&low[:,1:]).ravel()
            triple_mask = low[:,:-2]&low[:,1:-1]&low[:,2:]
        scale = float(2*np.var(reference_z[anchors].astype(float),axis=0).sum())
        if scale<=1e-12: raise FloatingPointError('Collapsed original training-reference scale')
        for lag in np.unique(np.round(dt[mask],9)):
            valid = mask & np.isclose(dt,lag,atol=1e-9,rtol=0)
            _,per_source = source_mean(energy/scale,source,valid)
            _,residual_source = source_mean(np.stack([residual,energy],-1),source,valid)
            triple = triple_mask & np.isclose(np.diff(data['time_ps'][ids],axis=1)[:,:-1],lag,atol=1e-9,rtol=0)
            triple &= np.isclose(np.diff(data['time_ps'][ids],axis=1)[:,1:],lag,atol=1e-9,rtol=0)
            if not triple.any(): raise ValueError('No equal-cadence triples for reported sequence lag')
            bend_energy = np.sum(bend**2,axis=-1)
            bend_sources = np.repeat(data['source_id'][ids],nt-2)
            _,per_bend = source_mean(bend_energy.ravel()/scale,bend_sources,triple.ravel())
            velocity_change = np.sum(np.diff(vel,axis=1)**2,axis=-1)
            velocity_energy = .5*(np.sum(vel[:,1:]**2,axis=-1)+np.sum(vel[:,:-1]**2,axis=-1))
            _,vstats = source_mean(np.stack([velocity_change,velocity_energy],-1).reshape(-1,2),bend_sources,triple.ravel())
            jumps = np.sqrt(energy[valid]/scale)
            same_membership = valid & (data['neighbor_retention'][ids].ravel()>=1.-1e-7)
            changed_membership = valid & ~same_membership
            row = dict(subset=subset,lag_ps=float(lag),pairs=int(valid.sum()),sources=len(per_source),
                rms_jump=float(np.sqrt(per_source.mean())),rms_ci95=interval(per_source,config,rms=True),
                p95_jump=float(np.quantile(jumps,.95)),max_jump=float(jumps.max()),
                absolute_rms_increment=float(np.sqrt(energy[valid].mean())),reference_squared_distance=scale,
                bend_rms=float(np.sqrt(per_bend.mean())),
                acceleration_rms_per_ps2=float(np.sqrt(np.sum(acc[triple]**2,axis=-1).mean())),
                velocity_change_ratio=float(vstats[:,0].mean()/vstats[:,1].mean()),
                learned_direction_explained=float(1-residual_source[:,0].mean()/residual_source[:,1].mean()),
                direction_projector_change=float(drift[valid].mean()),
                mean_neighbor_retention=float(data['neighbor_retention'][ids].ravel()[valid].mean()),
                atom_matched_rms_A=float(np.sqrt(np.mean(data['atom_matched_rms_A'][ids].ravel()[valid]**2))),
                retained_membership_pairs=int(same_membership.sum()),changed_membership_pairs=int(changed_membership.sum()),
                retained_membership_rms=float(np.sqrt(np.mean(energy[same_membership]/scale))) if same_membership.any() else None,
                changed_membership_rms=float(np.sqrt(np.mean(energy[changed_membership]/scale))) if changed_membership.any() else None)
            result.append(row)
    return result


def evaluate(config, root):
    torch.set_num_threads(config['cpu_threads']);device = config['devices'][0]
    data = load(config);paired = dict(np.load(Path(config['paired_features_cache'])/'features.npz'))
    x = torch.as_tensor(data['embedding'][:,:,:256],device=device)
    xp = torch.as_tensor(paired['embedding'][:,:256],device=device)
    rows = [];motion_rows = [];reference_errors = {}
    for spec in specifications(config):
        check_deadline(config)
        model,epoch = load_model(config,root,spec,device)
        with torch.no_grad():
            z,pred = model(x);zp,pp = model(xp)
            q = torch.cat([model.basis(v) for v in z.reshape(-1,z.shape[-1]).split(config['direction_samples'])])
        z = z.cpu().numpy();pred = pred.cpu().numpy();zp = zp.cpu().numpy();pp = pp.cpu().numpy()
        q = q.cpu().numpy().reshape(*z.shape,spec['rank'])
        directory = root/'technical'/spec['name']
        atomic_npz(directory/'evaluation-arrays.npz',embedding=z,prediction=pred,paired_embedding=zp,paired_prediction=pp)
        report = dict(spec=spec,selected_epoch=epoch,splits={},independent_probes=independent_probes(z,data,config))
        for split_name,split in [('validation',1),('development_test',2)]:
            ids = np.flatnonzero(data['split']==split);nt=z.shape[1]
            physical = physical_metrics(pred[ids].reshape(-1,160),data['target'][ids,:,:160].reshape(-1,160),
                (data['raw_target'][ids,:,4]<config['low_order_threshold']).ravel(),np.repeat(data['source_id'][ids],nt))
            pairs = np.flatnonzero(paired['split']==split);pr = np.ravel(np.c_[2*pairs,2*pairs+1])
            plow = (paired['raw_target'][2*pairs,4]<config['low_order_threshold']) & (paired['raw_target'][2*pairs+1,4]<config['low_order_threshold'])
            pair_physical = physical_metrics(pp[pr],paired['target'][pr,:160],np.repeat(plow,2),np.repeat(paired['source_id'][pairs],2))
            jm = jump_metrics(zp,paired,pairs,paired['reference_pair_ids'],config['low_order_threshold'])
            jl = jump_metrics(zp,paired,pairs,paired['reference_pair_ids'],config['low_order_threshold'],True)
            seq = sequence_metrics(z,q,data,split,zp,paired,config)
            local = local_direction_test(z,data,split,config)
            report['splits'][split_name] = dict(physical=physical,original_pair_physical=pair_physical,
                original_pair_jump=jm,original_pair_low_order_jump=jl,sequence_motion=seq,local_directions=local)
            key = spec['seed'],spec['rank'],split
            if spec['kind']=='reference': reference_errors[key] = physical,pair_physical
            base,base_pair = reference_errors[key]
            ratios = [p[sub][k]/b[sub][k] for p,b in [(physical,base),(pair_physical,base_pair)]
                for sub in ('all','low_order') for k in FAMILIES]
            row = dict(variant=spec['name'],split=split_name,kind=spec['kind'],dimension=spec['dimension'] or 256,
                rank=spec['rank'],seed=spec['seed'],selected_epoch=epoch,
                paired_rms_jump=jm['rms_jump'],paired_p95_jump=jm['p95_jump'],paired_max_jump=jm['max_jump'],
                paired_low_order_rms_jump=jl['rms_jump'],worst_physical_error_ratio=max(ratios),
                information_pass=int(max(ratios)<=1+config['retention_allowance']),
                jump_pass=int(max(jm['rms_jump'],jl['rms_jump'])<=config['target_jump']),
                extra_observable_probe_mse=report['independent_probes'][split_name]['mean'],
                local_4_direction_energy=local['explained_energy']['4'],local_8_direction_energy=local['explained_energy']['8'])
            for value in seq:
                if np.isclose(value['lag_ps'],config['evaluation_lag_ps']):
                    prefix = value['subset']+'_'
                    row.update({prefix+k:value[k] for k in ('rms_jump','bend_rms','velocity_change_ratio','learned_direction_explained')})
                motion_rows.append(dict(variant=spec['name'],split=split_name,**value))
            rows.append(row)
        write_json(directory/'evaluation.json',report)
        write_json(root/'technical/evaluation-status.json',dict(state='evaluating',variant=spec['name'],completed=len(rows)//2))
        print('MOTION EVALUATED',spec['name'],flush=True)
    snapshot_metric_docs(root,'mace_local_motion')
    for name,values in [('comparison',rows),('sequence_motion',motion_rows)]:
        keys = list(dict.fromkeys(k for r in values for k in r))
        with (root/f'tables/{name}.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(values)
    eligible = [r for r in rows if r['split']=='validation' and r['kind']!='reference' and r['information_pass']]
    winner = min(eligible,key=lambda r:r['paired_low_order_rms_jump'])['variant'] if eligible else None
    write_json(root/'technical/evaluation-status.json',dict(state='complete',rows=len(rows),validation_selected=winner,
        scope='Frozen snapshot stage B. Development test sources; no history or MACE fine-tuning.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axes,('validation','development_test'),strict=True):
        for kind in sorted(set(r['kind'] for r in rows)):
            subset = [r for r in rows if r['split']==split and r['kind']==kind]
            ax.scatter([r['paired_low_order_rms_jump'] for r in subset],[r['worst_physical_error_ratio'] for r in subset],label=kind)
        ax.axvline(config['target_jump'],ls='--',color='gray');ax.axhline(1+config['retention_allowance'],ls=':',color='gray')
        ax.set(xlabel='Within-low-order RMS jump, original 0.75 ps pairs',ylabel='Worst physical-error ratio',title=split)
        ax.legend()
    fig.tight_layout();fig.savefig(root/'plots/motion_information.png',dpi=180);plt.close(fig)
