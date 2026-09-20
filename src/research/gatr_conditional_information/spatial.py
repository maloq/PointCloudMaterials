"""Dense spatial extension: strict radial matches with frozen trajectory probes."""
from src.data.structural_pretraining.support import OUTER_RADIUS
import argparse
import json
from pathlib import Path
import time

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

from src.data.structural_pretraining.prepare import source_arrays,chart,offsets,file_hash,save_json,geometry_packet,REFERENCE_RADIUS
from src.research.gatr_equivariant.model import Capture
from src.research.gatr_equivariant.run import encode_rows
from .data import radial_control,load,require_hardware
from .probe import feature_sets,predict_path


def prepare(extension,config):
    require_hardware(config)
    root=Path(config['output'])/'technical/spatial';root.mkdir(exist_ok=True)
    if (root/'extension.json').exists() and json.loads((root/'extension.json').read_text())!=extension:
        raise ValueError('Spatial extension definition changed')
    save_json(root/'extension.json',extension)
    parent=json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
    audit=Path(extension['spatial_audit']);audit_config=json.loads((audit/'technical/config.json').read_text())
    if audit_config['checkpoint_sha256']!=config['checkpoint_sha256']:raise ValueError('Spatial audit used a different checkpoint')
    model=Capture(Path(config['parent_audit'])/'technical/gatr.pt',config['checkpoint_sha256']).cuda().eval()
    soap=SOAP(species=['Al'],periodic=False,r_cut=7.,n_max=8,l_max=6,sigma=.3,sparse=False,dtype='float64')
    radius=OUTER_RADIUS*model.scale/REFERENCE_RADIUS
    for source in parent['sources']:
        if source['split']!='test':continue
        raw=source_arrays(dict(source,kind='dynamic'))
        for frame in audit_config['spatial_frames']:
            path=root/f'{source["id"]}-{frame:04d}.npz'
            if path.with_suffix('.json').exists():continue
            started=time.monotonic();prior=audit/'technical/spatial'/path.name
            rec=json.loads(prior.with_suffix('.json').read_text())
            if file_hash(prior)!=rec['sha256']:raise ValueError(f'Changed original spatial panel {prior}')
            a=dict(np.load(prior));points,tree,box=chart(raw,frame,False)
            rows=np.searchsorted(raw['atom_ids'],a['atom_ids']);np.testing.assert_array_equal(raw['atom_ids'][rows],a['atom_ids'])
            np.testing.assert_array_equal(points[rows],a['positions'])
            neighbors=tree.query_ball_point(points[rows],radius,return_sorted=True)
            local=[offsets(points,center,np.array(ids),box) for center,ids in zip(rows,neighbors,strict=True)]
            replacements=[];near=[];quantiles=[];moments=[];errors=[];geometry=[];soaps=[]
            for p in local:
                r,n,q,m,e=radial_control(p,model.scale,config['reference_quantiles'])
                replacements.append(r);near.append(n);quantiles.append(q);moments.append(m);errors.append(e)
                geometry.append(geometry_packet(p))
                soaps.append(soap.create(Atoms('Al'*len(p),positions=p),centers=[[0.,0.,0.]])[0])
            zrad,_=encode_rows(model,replacements,[0]*len(replacements),config['batch_size'])
            centers=[int(np.flatnonzero(np.array(ids)==c).item()) for c,ids in zip(rows,neighbors,strict=True)]
            check,_=encode_rows(model,local[:16],centers[:16],config['batch_size'])
            np.testing.assert_allclose(check,a['z'][:16],atol=2e-6,rtol=2e-5)
            geometry=np.array(geometry)
            radial=np.concatenate((geometry[:,:32],near,quantiles,moments,a['order'][:,6:8]),axis=1)
            n=len(rows)
            np.savez(path,radial=radial,radii80=near,radial_quantiles=quantiles,
                context=np.column_stack((np.full(n,source['temperature_K']),np.full(n,float(a['time_ps'])))),
                gatr=a['z'],radial_gatr=zrad,soap=np.array(soaps),bond=a['order'][:,:6],angular=geometry[:,64:80],
                labels=a['labels'],order=a['order'],source=np.full(n,source['id']),frame=np.full(n,frame),
                atom=a['atom_ids'],temperature=np.full(n,source['temperature_K']))
            save_json(path.with_suffix('.json'),dict(source=source['id'],frame=frame,rows=n,sha256=file_hash(path),
                parent_spatial_sha256=rec['sha256'],radius_preservation_max_A=max(errors),seconds=time.monotonic()-started))
            print(f'Spatial conditional {source["id"]}/{frame}: {n} rows, {time.monotonic()-started:.1f}s',flush=True)


def load_spatial(config):
    root=Path(config['output'])/'technical/spatial'
    parts=[]
    for path in sorted(root.glob('*.npz')):
        rec=json.loads(path.with_suffix('.json').read_text())
        if file_hash(path)!=rec['sha256']:raise ValueError('Changed spatial conditional observations')
        parts.append(dict(np.load(path)))
    if len(parts)!=70:raise ValueError(f'Expected 70 spatial snapshots; found {len(parts)}')
    return {k:np.concatenate([a[k] for a in parts]) for k in parts[0]}


def predict(extension,config):
    require_hardware(config)
    original,sources,_=load(config);spatial=load_spatial(config)
    y=np.concatenate((original['bond'],original['angular']),axis=1).astype(float)
    inputs=feature_sets(original,'structure',extension['methods'])
    test_inputs=feature_sets(spatial,'structure',extension['methods'])
    root=Path(config['output'])/'technical'
    for family in ('linear','nonlinear'):
        for method,x in inputs.items():
            directory=root/'spatial-probes'/family/method;directory.mkdir(parents=True,exist_ok=True)
            for source in sources:
                sid=source['id'];dest=directory/f'{sid}.npz'
                if dest.with_suffix('.json').exists():continue
                training=np.flatnonzero((original['source']!=sid)&(original['frame']%config['probe_stride']==0))
                test=np.flatnonzero(spatial['source']==sid)
                receipt=root/'probes/structure'/family/method/f'{sid}.json'
                selected=json.loads(receipt.read_text())
                np.testing.assert_array_equal(np.unique(original['source'][training]),selected['training_sources'])
                predictions,mean,scale=predict_path(x[training],y[training],test_inputs[method][test],original['source'][training],
                    family,config['ridge_penalties'],selected['rff_seed'],config['rff_dimensions'],False)
                indices=[config['ridge_penalties'].index(p) for p in selected['penalty_per_target']]
                prediction=np.stack([predictions[k,:,j] for j,k in enumerate(indices)],axis=1)
                np.savez(dest,indices=test,prediction=prediction,target_mean=mean,target_scale=scale)
                save_json(dest.with_suffix('.json'),dict(source=sid,rows=len(test),sha256=file_hash(dest),
                    trajectory_probe_selection_sha256=file_hash(receipt),training_sources=selected['training_sources']))
            print(f'Spatial probe: {family}/{method}',flush=True)


def report_extension(extension,config):
    import pandas as pd
    import matplotlib.pyplot as plt
    from .report import source_scores,pair_scores,comparisons
    from .metrics import matched_pairs,pair_mask
    from .plots import bars,save
    from src.experiment_runner.metric_docs import snapshot_metric_docs
    a=load_spatial(config);_,sources,_=load(config)
    root=Path(config['output']);pairs=matched_pairs(a,config)
    np.savez(root/'technical/spatial-matched-pairs.npz',**pairs)
    source_rows=[];pair_rows=[]
    for family in ('linear','nonlinear'):
        for method in extension['methods']:
            result={k:np.full((len(a['source']),22),np.nan) for k in ('prediction','mean','scale')}
            for source in sources:
                path=root/'technical/spatial-probes'/family/method/f'{source["id"]}.npz'
                receipt=json.loads(path.with_suffix('.json').read_text())
                if file_hash(path)!=receipt['sha256'] or source['id'] in receipt['training_sources']:
                    raise ValueError('Invalid spatial held-out predictions')
                p=np.load(path);ix=p['indices']
                if not np.all(a['source'][ix]==source['id']):raise ValueError('Incorrect spatial source row map')
                result['prediction'][ix]=p['prediction'];result['mean'][ix]=p['target_mean'];result['scale'][ix]=p['target_scale']
            if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Incomplete spatial predictions')
            source_rows.extend(source_scores(a,sources,'structure',family,method,result,config))
            pair_rows.extend(pair_scores(a,sources,pairs,'structure',family,method,result,config))
    balance=[]
    for source in sources:
        belongs=a['source'][pairs['left']]==source['id']
        for caliper in config['match_calipers_A']:
            valid=belongs&pair_mask(pairs,caliper,config)
            balance.append(dict(source=source['id'],temperature_K=source['temperature_K'],caliper_A=caliper,
                candidate_pairs=int(belongs.sum()),matched_pairs=int(valid.sum()),
                mean_radial_rms_A=float(pairs['radial_rms_A'][valid].mean()) if valid.any() else None,
                mean_full_radial_rms_A=float(pairs['full_radial_rms_A'][valid].mean()) if valid.any() else None,
                mean_density_gap=float(pairs['density_relative'][valid].mean()) if valid.any() else None))
    tables=dict(spatial_source_scores=source_rows,spatial_matched_scores=pair_rows,spatial_matching_balance=balance,
        spatial_conditional_gains=comparisons(source_rows,['task','family','target'],config),
        spatial_matched_gains=comparisons(pair_rows,['task','family','caliper_A','target'],config))
    snapshot_metric_docs(root,'gatr_conditional_information')
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(root/'tables'/f'{name}.csv',index=False)
    gains=pd.DataFrame(tables['spatial_matched_gains'])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,family in zip(axes,('linear','nonlinear'),strict=True):
        sub=gains[(gains.family==family)&(gains.caliper_A==.05)&(gains.baseline=='radial_control')]
        bars(ax,sub,['q6','qbar6','angular_arrangement'],['plus_gatr','plus_angular_delta','plus_soap'],['q6','q̄6','Angular\narrangement'])
        ax.set_title(f'{family.title()} probe · entire test source excluded')
    primary=pd.DataFrame(balance).query('caliper_A == 0.05')
    total=int(primary.matched_pairs.sum());represented=int((primary.matched_pairs>0).sum());axes[0].legend(fontsize=8)
    fig.suptitle(f'Dense spatial extension: {total:,} radially matched pairs across 70 snapshots\n'
        'Both radial RMS gaps ≤0.05 Å; density gap ≤2%; frozen trajectory-probe settings',fontsize=14)
    save(fig,root,'spatial-matched-information')
    lines=['# Dense spatial extension of the radial-matching test','',
        'The four-track primary matching population yielded only three pairs at 0.05 Å, with no prospective pairs. '
        'That sample cannot support an inferential matched-pair conclusion. We therefore extended only the structural '
        'test to the 70 dense spatial snapshots already selected by the preceding equivariance audit. The threshold '
        'and probe settings were kept fixed; this is an explicitly added exploratory population.','',
        f'There are {len(a["source"]):,} environments and {len(pairs["left"]):,} same-source, same-frame candidate pairs. '
        f'**{total:,} pairs** pass 0.05 Å inner/full radial RMS and 2% density calipers. '
        f'They span **{represented} of {len(sources)} sources**. They are dependent pairs, not independent experimental replicates; '
        'intervals resample the represented sources, stratified by temperature. '
        'Dense patches are a local sample, not a uniform full-system pair distribution.','',
        'Readouts are fitted on the original trajectory observations from nine other sources. Their penalty selections '
        'come from the original nested training procedure; no spatial labels tune them. Original scalar GATr z128, '
        'its difference from the radius-only control, and SOAP are assessed with both probe families.','',
        '| Probe | Target contrast | Add GATr z128 | Add GATr angular difference | Add SOAP |','|---|---|---:|---:|---:|']
    for family in ('linear','nonlinear'):
        for target in ('q6','qbar6','angular_arrangement'):
            entries=[]
            for method in ('plus_gatr','plus_angular_delta','plus_soap'):
                row=gains[(gains.family==family)&(gains.caliper_A==.05)&(gains.target==target)&(gains.method==method)&(gains.baseline=='radial_control')].iloc[0]
                entries.append(f'{row.improvement_percent:+.1f}% [{row.low:+.1f}, {row.high:+.1f}]')
            lines.append(f'| {family} | {target} | '+ ' | '.join(entries)+' |')
    loose=pd.DataFrame(balance).query('caliper_A == 0.1')
    lines+=['',f'The predeclared 0.10 Å sensitivity retains **{int(loose.matched_pairs.sum()):,} pairs across all ten sources**. '
        'In the nonlinear probe, adding original GATr changes q6-contrast error by '
        f'**{gains.query("family == \'nonlinear\' and caliper_A == 0.1 and method == \'plus_gatr\' and target == \'q6\'").improvement_percent.iloc[0]:+.1f}%** '
        '(negative means worse), while the angular-contrast gain is '
        f'**{gains.query("family == \'nonlinear\' and caliper_A == 0.1 and method == \'plus_gatr\' and target == \'angular_arrangement\'").improvement_percent.iloc[0]:+.1f}%**. '
        'This larger, less strict matched population also shows little additional GATr information.']
    lines+=['','Values are reductions in held-out target-contrast MSE relative to the full radial control. '
        'Positive is better; brackets are paired source-bootstrap intervals. Future crystallization is evaluated '
        'only on the original identity-preserving timelines in the [main report](RESULTS.md).','',
        '![Dense matched comparison](plots/spatial-matched-information.png)','',
        'Reproduce after the main extraction and probe stage:','',
        '```bash','conda run -n pointnet-torch214 python -m src.research.gatr_conditional_information.spatial \\',
        '  --config configs/analysis/gatr_conditional_spatial.json','```','']
    (root/'RESULTS_spatial.md').write_text('\n'.join(lines))
    save_json(root/'technical/spatial-summary.json',dict(observations=len(a['source']),candidate_pairs=len(pairs['left']),
        matched_pairs_primary=total,sources=len(sources),matched_sources_primary=represented,
        matched_pairs_01_A=int(loose.matched_pairs.sum()),protocol='dense_spatial_extension',config=extension))
    print(f'Spatial extension complete: {total} primary-caliper matched pairs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    args=p.parse_args();extension=json.loads(Path(args.config).read_text());config=json.loads(Path(extension['parent_config']).read_text())
    prepare(extension,config);predict(extension,config);report_extension(extension,config)
