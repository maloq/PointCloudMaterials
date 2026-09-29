"""PaCMAP scientific plots from frozen encoders and independent rich descriptors."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import html
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_runner.metric_docs import write_metric_table, check_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.equivariant_context.cache import RetainedCache
from src.research.structural_state.common import sha, digest, write_json
from .correspondence import configuration, observations, references


def settings(config):
    c=json.loads(Path(config).read_text())
    if c['protocol']!='interface_pacmap_v1': raise ValueError('Wrong PaCMAP protocol')
    for key in ('output','publication'):
        value=Path(c[key])
        c[key]=str(value.resolve() if value.is_absolute() else resolve_path(c[key]).resolve())
    corr,parent=configuration(c['correspondence_config'])
    return c,corr,parent


def context(c,corr,parent):
    a=observations(parent); ref,_=references(corr,a)
    ids=np.flatnonzero(a['role']=='test')
    for key in tuple(a):
        if key!='names': a[key]=a[key][ids]
    ref={k:v[ids] for k,v in ref.items()}
    labels={}; models={}; bindings={}
    for family in corr['families']:
        p=Path(corr['output'])/'data'/f'interface12-{family}-k7-descriptor-model.npz'
        bindings[str(p)]=sha(p)
        with np.load(p) as z:
            seed=np.flatnonzero(z['cluster_seeds']==corr['primary_cluster_seed'])
            if len(seed)!=1: raise ValueError('Missing declared descriptor seed')
            labels[family]=z['assignments'][seed[0],ids]
            models[family]={k:z[k] for k in ('columns','mean','sd','balance','centers')}
    return a,ref,labels,models,bindings


def color_fields(a,ref,labels,own,own_title):
    d=ref['distance']; solid=ref['solid']; near=np.isfinite(d)&(d<=12)
    region=np.full(len(d),4,np.int32)
    region[near & solid]=1;region[near & ref['accepted_disorder']]=2
    region[near & ~solid & ~ref['accepted_disorder']]=3
    region[ref['interface_member']]=0;region[~np.isfinite(d)]=5
    distance=np.clip(np.where(solid,-1,1)*d,-20,20)
    # Missing interfaces remain missing, never an invented finite distance.
    distance[~np.isfinite(d)]=np.nan
    fields={own_title:own, 'TDA clusters':labels['tda'], 'Bond-order clusters':labels['bond_order'],
            'CNA clusters':labels['cna'],'Joint descriptor clusters':labels['joint'],
            'PTM type':a['ptm'], 'Interface distance (Å, clipped ±20)':distance,
            'Input crystal fraction':a['support_fraction'], 'Physical region':region}
    return fields


def list_values(value):
    array=np.asarray(value)
    if array.dtype.kind=='f': return [float(x) if np.isfinite(x) else None for x in array]
    return array.tolist()


def static_plot(y,fields,title,path):
    selected=[next(iter(fields)), 'Joint descriptor clusters','TDA clusters','Physical region',
              'Interface distance (Å, clipped ±20)','Input crystal fraction']
    fig,axs=plt.subplots(2,3,figsize=(14,9))
    for ax,key in zip(axs.flat,selected):
        val=fields[key]; distance=key.startswith('Interface distance'); fraction=key=='Input crystal fraction'
        options=dict(cmap='coolwarm',vmin=-20,vmax=20) if distance else (
            dict(cmap='viridis',vmin=0,vmax=1) if fraction else dict(cmap='tab10',vmin=-.5,vmax=9.5))
        im=ax.scatter(y[:,0],y[:,1],c=val,s=.7,alpha=.7,linewidths=0,rasterized=True,**options)
        ax.set(title=key,xlabel='PaCMAP 1',ylabel='PaCMAP 2'); fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(title+f' · {len(y):,} identical observations');fig.tight_layout()
    fig.savefig(path,dpi=170);plt.close(fig)


def interactive(y2,y3,a,ref,fields,title,path):
    payload=dict(title=title,y2=np.round(y2,6).tolist(),y3=np.round(y3,6).tolist(),
                 source=list_values(a['source']),frame=list_values(a['frame']),atom=list_values(a['atom']),
                 distance=list_values(ref['distance']),solid=list_values(ref['solid']),
                 region=list_values(fields['Physical region']),fields={k:list_values(v) for k,v in fields.items()})
    template=Path(__file__).with_name('pacmap_view.html').read_text()
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    path.write_text(template.replace('__TITLE__',html.escape(title)).replace('__DATA__',encoded))


def gallery(c):
    out=Path(c['output']);dest=Path(c['publication']);dest.mkdir(parents=True,exist_ok=True)
    with (out/'technical/gallery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        receipts=sorted((out/'technical/views').glob('*.json'))
        rows=[]
        for p in receipts:
            r=json.loads(p.read_text());stem=r['name']
            rows.append(f'<tr><td>{html.escape(r["title"])}</td><td>{r["rows"]:,}</td>'
                        f'<td><a href="plots/{stem}.png">2D panels</a></td>'
                        f'<td><a href="interactive/{stem}.html">Interactive 2D + 3D</a></td></tr>')
        page=('<!doctype html><meta charset="utf-8"><title>Interface PaCMAP gallery</title><style>'
             'body{font:16px system-ui;margin:32px;max-width:1300px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}'
             'a{color:#145bc0}table{border-collapse:collapse;width:100%}</style><h1>Interface PaCMAP gallery</h1>'
             '<p>Frozen neural embeddings and rich non-neural descriptors. No encoder training. '
             'Both projections use the same atom identities. Colors are high-dimensional cluster assignments; '
             'PaCMAP coordinates are visualization only.</p>'
             '<p>Interface-focused views contain atoms within 20 Å of the crystal-side boundary layer. '
             'Full-context views contain all 24,960 fixed held-out samples. Separate layouts have arbitrary axes.</p>'
             f'<p>{len(rows)} completed views. This page updates as each view completes.</p>'
             '<table><thead><tr><th>Feature space / population</th><th>Atoms</th><th>2D</th><th>Interactive</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
        for root in (out,dest):
            temporary=root/'index.html.building';temporary.write_text(page);temporary.replace(root/'index.html')
            (root/'README.md').write_text('# Interface PaCMAP views\n\n[Open gallery](index.html).\n\n'
                f'{len(rows)} views currently complete. Each includes 2D PNG panels and an interactive 2D/3D HTML view. '
                'Frozen neural embeddings, rich TDA, bond order, CNA and their balanced combination; no neural training. '
                'The same fixed held-out atoms are used in each space. PaCMAP layouts are transductive visualizations, '
                'not held-out predictive evaluations or evidence of distinct states by themselves.\n')


def projection(c,a,ref,labels,x,own,own_title,title,name,bindings):
    import pacmap
    import numba
    import faiss
    numba.set_num_threads(c['threads_per_worker']);faiss.omp_set_num_threads(c['threads_per_worker'])
    if version('pacmap')!=c['pacmap']['version']: raise ValueError('PaCMAP version changed')
    out=Path(c['output']);dest=Path(c['publication'])
    params={k:v for k,v in c['pacmap'].items() if k!='version'}
    for population in c['populations']:
        stem=name+'-'+population;receipt=out/'technical/views'/f'{stem}.json'
        if receipt.exists(): continue
        mask=np.ones(len(x),bool) if population=='all_test' else np.isfinite(ref['distance'])&(ref['distance']<=20)
        ids=np.flatnonzero(mask)
        if len(ids)<100: raise ValueError(f'Insufficient projection population: {population}')
        xx=np.asarray(x[ids],np.float32)
        if not np.isfinite(xx).all(): raise ValueError(f'Nonfinite projection features: {name}')
        fields=color_fields(a,ref,labels,own,own_title);fields={k:v[ids] for k,v in fields.items()}
        aa={k:a[k][ids] for k in ('source','frame','atom')};rr={k:ref[k][ids] for k in ref}
        coords={};started=time.monotonic()
        for dim in (2,3):
            print(f'PaCMAP {stem}, dimensions={dim}, rows={len(ids)}',flush=True)
            model=pacmap.PaCMAP(n_components=dim,**params)
            coords[dim]=model.fit_transform(xx,init='pca')
            if coords[dim].shape!=(len(ids),dim) or not np.isfinite(coords[dim]).all():
                raise ValueError(f'Invalid PaCMAP coordinates {stem}/{dim}')
        np.savez_compressed(out/'data'/f'{stem}.npz',pacmap2=coords[2],pacmap3=coords[3],
                            original_row=a['original_row'][ids],**aa)
        label=title+' / '+population
        static_plot(coords[2],fields,label,out/'plots'/f'{stem}.png')
        interactive(coords[2],coords[3],aa,rr,fields,label,out/'interactive'/f'{stem}.html')
        for part,suffix in (('plots','png'),('interactive','html')):
            shutil.copy2(out/part/f'{stem}.{suffix}',dest/part/f'{stem}.{suffix}')
        r=dict(name=stem,title=label,rows=len(ids),dimensions=[2,3],features=x.shape[1],
               pacmap_params=params,packages={p:version(p) for p in ('pacmap','faiss-cpu','numpy','numba','plotly','torch')},
               neural_training=False,inputs=bindings,seconds=time.monotonic()-started,
               coordinate_sha256=sha(out/'data'/f'{stem}.npz'))
        write_json(receipt,r);gallery(c)


def task(config,item):
    c,corr,parent=settings(config);a,ref,labels,models,bindings=context(c,corr,parent)
    if item['kind']=='descriptor':
        family=item['family'];model=models[family]
        x=((a['targets'][:,model['columns']]-model['mean'])/model['sd']/model['balance']).astype(np.float32)
        projection(c,a,ref,labels,x,labels[family],family+' clusters',family+' rich descriptors',
                   'descriptors-'+family,bindings)
        return item
    import torch
    from omegaconf import OmegaConf
    from .train import PairEncoder
    torch.set_num_threads(c['threads_per_worker'])
    device=c['inference_device']
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    run=f'S{item["alpha"]:g}-seed{item["seed"]}';epoch=item['epoch']
    root=Path(parent['output'])/run;checkpoint=root/'checkpoints'/f'epoch-{epoch:02d}.pt'
    saved_receipt=json.loads((root/'analyses'/f'epoch-{epoch:02d}'/'technical/complete.json').read_text())
    checkpoint_hash=sha(checkpoint)
    if checkpoint_hash!=saved_receipt['checkpoint_sha256']: raise ValueError('Checkpoint identity changed')
    bindings[str(checkpoint)]=checkpoint_hash
    metadata=dict(checkpoint_sha256=checkpoint_hash,rows=digest(a['original_row'].tolist()),
                  producer_sha256=sha(Path(__file__)),model_snapshot=c['model_snapshot_sha256'],
                  device=device+'-float32',original_ab_batches=(device=='cuda'))
    cache=RetainedCache(parent['feature_cache'],limit=6)
    with cache.lease(digest(metadata),deadline=time.time()+c['cpu_hours']*3600,metadata=metadata) as folder:
        path=folder/'heldout-features.npy';marker=folder/'features-complete.json'
        if marker.exists():
            if sha(path)!=json.loads(marker.read_text())['sha256']: raise ValueError('Changed generated features')
        else:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['epoch']!=epoch or saved['offset']!=0 or saved['data_identity']!=corr['cache_identity']:
                raise ValueError('Wrong checkpoint epoch or data identity')
            torch.manual_seed(saved['seed']);np.random.seed(saved['seed'])
            model=PairEncoder(OmegaConf.create(saved['recipe'])).to(device)
            model.load_state_dict(saved['model'],strict=True);model.eval();model.requires_grad_(False)
            del saved
            parents=np.load(Path(parent['cache'])/'assay/parents.npy',mmap_mode='r')
            values=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(a['source']),2,128))
            with torch.inference_mode():
                if device=='cuda':
                    # Replay the original A+B inference batches, including the original offsets.
                    indices=np.load(Path(parent['cache'])/'assay/view_indices.npy',mmap_mode='r')
                    choices=np.load(Path(parent['cache'])/'assay/pair_choice.npy',mmap_mode='r')
                    batch_start=(a['original_row']//256)*256
                    for first in np.unique(batch_start):
                        p=np.asarray(parents[first:first+256]);idx=np.asarray(indices[first:first+256])
                        choice=np.asarray(choices[first:first+256]);near=idx[np.arange(len(p)),choice]
                        b=np.take_along_axis(p,near[:,:,None],axis=1);b=b-b[:,:1]
                        x=torch.as_tensor(np.concatenate([p[:,:80],b])/parent['geometry']['length_scale_A'],device=device)
                        z,y=model(x);aa=torch.stack([z[:len(p)],y[:len(p)]],1).cpu().numpy()
                        take=np.flatnonzero(batch_start==first)
                        values[take]=aa[a['original_row'][take]-first]
                else:
                    for first in range(0,len(values),256):
                        ids=a['original_row'][first:first+256]
                        points=np.asarray(parents[ids,:80])/parent['geometry']['length_scale_A']
                        z,y=model(torch.from_numpy(points))
                        values[first:first+len(ids)]=torch.stack([z,y],1).numpy()
            if not np.isfinite(values).all(): raise ValueError('Nonfinite frozen inference')
            values.flush();values._mmap.close();del values,model
            if device=='cuda':torch.cuda.empty_cache()
            write_json(marker,dict(sha256=sha(path),neural_training=False))
        # Keep no NFS mmap alive after its lease, when another worker may evict it.
        values=np.load(path);inference_check={}
        for j,rep in enumerate(('encoder','projector')):
            assignment=root/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz'
            bindings[str(assignment)]=sha(assignment)
            with np.load(assignment) as z:
                for key in ('source','frame','atom'):
                    if not np.array_equal(z[key][a['original_row']],a[key]): raise ValueError('Assignment identity changed')
                own=z['cluster'][a['original_row']];centers=z['centers']
            x=np.asarray(values[:,j])
            from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
            xx=np.ascontiguousarray(x)
            replay=_labels_inertia_threadpool_limit(xx,np.ones(len(xx),dtype=xx.dtype),centers,n_threads=1,return_inertia=False)
            disagreement=int(np.count_nonzero(replay!=own));inference_check[rep]=disagreement
            # CPU/CUDA roundoff at exact boundaries is recorded; material disagreement fails.
            if disagreement>2: raise ValueError(f'Frozen inference changed assignments: {run}/{epoch}/{rep}: {disagreement}')
            projection(c,a,ref,labels,x,own,'Neural clusters',f'{run}, epoch {epoch}, {rep}',
                       f'{run}-epoch{epoch:02d}-{rep}',bindings)
        write_json(Path(c['output'])/'technical'/f'{run}-epoch{epoch:02d}-inference.json',
                   dict(neural_training=False,device=device+'-float32',original_ab_batches=(device=='cuda'),assignment_disagreements=inference_check,
                        checkpoint_sha256=checkpoint_hash,model_snapshot_sha256=c['model_snapshot_sha256']))
    return item


def setup(c):
    from plotly.offline import get_plotlyjs
    out=Path(c['output']);dest=Path(c['publication'])
    for root in (out,dest):
        for part in ('plots','interactive','assets','technical/views','data'):(root/part).mkdir(parents=True,exist_ok=True)
        (root/'assets/plotly.min.js').write_text(get_plotlyjs())
    gallery(c)


def preview(config):
    c,_,_=settings(config);setup(c)
    # First production views also exercise the full descriptor/projection/frozen-inference path.
    task(config,dict(kind='descriptor',family='joint'))
    task(config,dict(kind='neural',**c['preview_checkpoint']))
    write_json(Path(c['output'])/'technical/preview-complete.json',dict(state='complete',neural_training=False))


def run(config):
    c,corr,parent=settings(config);out=Path(c['output']);dest=Path(c['publication'])
    if not (Path(corr['output'])/'technical/complete.json').exists(): raise ValueError('Descriptor comparison not complete')
    setup(c)
    tasks=[dict(kind='descriptor',family=f) for f in corr['families']]
    tasks += [dict(kind='neural',epoch=e,alpha=a,seed=s) for e in c['epochs'] for s in c['seeds'] for a in c['alphas']]
    gallery(c)
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(task,config,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            done=future.result();print(f'Completed PaCMAP task {i}/{len(tasks)}: {done}',flush=True)
    receipts=[json.loads(p.read_text()) for p in (out/'technical/views').glob('*.json')]
    expected=(len(corr['families'])+2*len(c['epochs'])*len(c['seeds'])*len(c['alphas']))*len(c['populations'])
    if len(receipts)!=expected: raise ValueError('Incomplete projection coverage')
    result=dict(state='complete',views=len(receipts),projections=2*len(receipts),neural_training_runs=0,
                checkpoint_evaluations=len(tasks)-len(corr['families']),job=os.environ.get('SLURM_JOB_ID'))
    write_metric_table(dict(coverage=result,views={r['name']:dict(rows=r['rows'],features=r['features'],seconds=r['seconds'])
                                                for r in receipts}),out,family='interface_pacmap',name='projection-coverage')
    write_json(out/'technical/complete.json',result)
    shutil.copytree(out/'tables',dest/'tables',dirs_exist_ok=True)
    for part in ('metric-contracts','table-contracts'):
        shutil.copytree(out/'technical'/part,dest/'technical'/part,dirs_exist_ok=True)
    for file in ('complete.json','metric-contract.json'):shutil.copy2(out/'technical'/file,dest/'technical'/file)
    gallery(c)


def submit(config):
    c,corr,parent=settings(config);tech=Path(c['output'])/'technical/queue';code=tech/'code'
    if (tech/'launch.json').exists(): raise FileExistsError('PaCMAP already submitted')
    check_metric_docs(family='interface_pacmap');repo=Path(__file__).resolve().parents[3]
    for directory in ('src','configs','docs/metrics'):
        shutil.copytree(repo/directory,code/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(repo/'machine.local.yaml',code/'machine.local.yaml')
    # Use exactly the original encoder implementation, even if the live architecture has since changed.
    original=Path(parent['output'])/'technical/queue/code'
    shutil.rmtree(code/'src/models');shutil.copytree(original/'src/models',code/'src/models',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative in ('src/training_methods/shared/vicreg.py','src/research/spatial_vicreg_bias/train.py'):
        shutil.copy2(original/relative,code/relative)
    files={str(p.relative_to(code)):sha(p) for p in (code/'src/models').rglob('*.py')}
    for relative in ('src/training_methods/shared/vicreg.py','src/research/spatial_vicreg_bias/train.py'):
        files[relative]=sha(code/relative)
    c['model_snapshot_sha256']=digest(files);write_json(tech/'model-snapshot.json',files)
    corr['parent']=str(tech/'parent.json');write_json(corr['parent'],parent)
    c['correspondence_config']=str(tech/'correspondence.json');write_json(c['correspondence_config'],corr)
    cfg=tech/'config.json';write_json(cfg,c)
    if 'inherit_descriptors_from' in c:
        previous=Path(c['inherit_descriptors_from']);inheritance={}
        for receipt in sorted((previous/'technical/views').glob('descriptors-*.json')):
            r=json.loads(receipt.read_text());name=r['name']
            if sha(previous/'data'/f'{name}.npz')!=r['coordinate_sha256']:raise ValueError('Changed inherited descriptor projection')
            for relative in (f'technical/views/{name}.json',f'data/{name}.npz',f'plots/{name}.png',f'interactive/{name}.html'):
                dest=Path(c['output'])/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(previous/relative,dest)
                inheritance[relative]=dict(source=str(previous/relative),sha256=sha(dest))
        write_json(tech/'inherited-projections.json',inheritance)
    launch=json.loads((Path(corr['output'])/'technical/queue/launch.json').read_text())
    dependency=next(r['job'] for r in launch['jobs'] if r['stage']=='compare')
    correspondence_complete=(Path(corr['output'])/'technical/complete.json').exists()
    receipt=dict(state='submitting',neural_training=False,config=str(cfg),jobs=[])
    write_json(tech/'launch.json',receipt)
    for stage,hours in (('preview',2),('run',c['cpu_hours'])):
        script=tech/(stage+'.sbatch')
        command=[sys.executable,'-u','-m','src.research.spatial_vicreg_bias.pacmap_views',stage,'--config',str(cfg)]
        resource=['#SBATCH --partition=CPU'] if c['inference_device']=='cpu' else [
            '#SBATCH --partition='+c['gpu_partition'],'#SBATCH --gres=gpu:1']
        if c['inference_device'] != 'cpu' and c.get('gpu_node'):
            resource.append('#SBATCH --nodelist='+c['gpu_node'])
        script.write_text('\n'.join(['#!/bin/bash',f'#SBATCH --job-name=SVB-PaCMAP-{stage}',*resource,
            f'#SBATCH --cpus-per-task={c["workers"]*c["threads_per_worker"]+(2 if c["inference_device"]=="cpu" else 0)}', '#SBATCH --mem=24G',
            f'#SBATCH --time={hours:02d}:00:00',f'#SBATCH --output={tech}/{stage}-%j.log','set -euo pipefail',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            f'export NUMBA_NUM_THREADS={c["threads_per_worker"]}',
            'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OVITO_THREAD_COUNT=1 QT_QPA_PLATFORM=offscreen',
            'cd '+shlex.quote(str(code)),shlex.join(command)])+'\n')
        args=['sbatch','--parsable']
        if stage=='run':
            deps=[receipt['jobs'][0]['job']]+([] if correspondence_complete else [dependency])
            args+=['--dependency=afterok:'+':'.join(deps)]
        job=subprocess.check_output(args+[str(script)],text=True).strip().split(';')[0]
        receipt['jobs'].append(dict(job=job,stage=stage,script=str(script)));write_json(tech/'launch.json',receipt)
    receipt['state']='submitted';write_json(tech/'launch.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['submit','preview','run']);parser.add_argument('--config',required=True)
    args=parser.parse_args();globals()[args.stage](args.config)
