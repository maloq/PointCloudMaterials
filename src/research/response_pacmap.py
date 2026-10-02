"""Matched Al64 PaCMAP: response transfer, MM-TDA and GeoFormer encoders."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.metric_docs import write_metric_rows, check_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.equivariant_context.cache import RetainedCache


def read(path):
    return json.loads(Path(path).read_text())


def prepare(c, out):
    cache = resolve_path(c['assay_cache'])
    manifest = read(cache/'manifest.json')
    if manifest['identity'] != c['assay_identity'] or manifest['fixed_identity'] != c['fixed_identity']:
        raise ValueError('Al64 release changed')
    keys = ('uniform', 'role', 'source', 'frame', 'atom', 'ptm', 'support_fraction', 'parents')
    a = {}
    for key in keys:
        path = cache/'assay'/f'{key}.npy'
        if sha(path) != manifest['files'][f'assay/{key}.npy']:
            raise ValueError(f'Changed Al64 observation array: {key}')
        a[key] = np.load(path, mmap_mode='r')
    train = np.flatnonzero(a['uniform'] & (a['role'] == 'train'))
    test = np.flatnonzero(a['uniform'] & (a['role'] == 'test'))
    if len(test) != 24960 or len(np.unique(a['source'][test])) != 30:
        raise ValueError('Expected every all64 held-out structural-assay row')
    # Fixed label-free training landmarks; all test rows remain present.
    rng = np.random.default_rng(c['seed'])
    landmarks = []
    for source in np.unique(a['source'][train]):
        rows = train[a['source'][train] == source]
        for frame in np.unique(a['frame'][rows]):
            candidates = rows[a['frame'][rows] == frame]
            if len(candidates) != 64:
                raise ValueError('Expected64 frozen centers per source/frame')
            landmarks.extend(rng.choice(candidates, c['landmarks_per_source_frame'], replace=False))
    train = np.sort(landmarks)
    rows = np.concatenate((train, test))
    positions = np.asarray(a['parents'][rows, :80], dtype=np.float32)
    if np.max(np.linalg.norm(positions, axis=-1)) >= 8 or np.any(positions[:, 0]):
        raise ValueError('All three models must see the same centered80 atoms within8A')
    np.save(out/'data/positions.npy', positions)
    np.savez_compressed(out/'data/observations.npz', original_row=rows,
        **{k:np.asarray(a[k][rows]) for k in ('role', 'source', 'frame', 'atom', 'ptm', 'support_fraction')})
    return train, test, read(cache/'plan.json'), manifest


def render(c, out, entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    data = np.load(out/'data/observations.npz')
    mask = data['role'] == 'test'
    ptm, fraction = data['ptm'][mask], data['support_fraction'][mask]
    structure = np.where(np.isin(ptm, [1,2,3]), 2, np.where(fraction > 0, 1, 0))
    palette = ['#277da8', '#e5a335', '#ba4561']
    categories = ['Crystal-free neighborhood', 'Noncrystalline center / mixed neighborhood', 'Crystalline center']
    models = [(e, np.load(out/'data'/f'{e["name"]}-pacmap.npz')['test_xy']) for e in entries]
    order = np.random.default_rng(c['seed']).permutation(len(structure))
    for field, values in [('present-structure', structure), ('crystalline-fraction', fraction)]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.3))
        fig.subplots_adjust(left=.025,right=.965,top=.78,bottom=.19,wspace=.12)
        for ax, (entry, xy) in zip(axes, models, strict=True):
            opts = dict(cmap=ListedColormap(palette), norm=BoundaryNorm([-.5,.5,1.5,2.5],3)) if field=='present-structure' else dict(cmap='viridis',vmin=0,vmax=1)
            scatter = ax.scatter(xy[order,0],xy[order,1],c=values[order],s=2,alpha=.7,linewidths=0,rasterized=True,**opts)
            ax.set_title(entry['plot_title'],fontsize=11)
            center=(xy.min(0)+xy.max(0))/2
            half=float(np.ptp(xy,axis=0).max())*.55
            ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half)
            ax.set_aspect('equal',adjustable='box');ax.set_xticks([]);ax.set_yticks([])
        if field == 'present-structure':
            handles=[Line2D([],[],marker='o',linestyle='',color=color,label=label) for color,label in zip(palette,categories)]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,fontsize=9,frameon=False)
        else:
            fig.colorbar(scatter,ax=axes,shrink=.7,label='Present crystalline fraction among80 atoms')
        fig.suptitle('PaCMAP · 24,960 identical held-out Al neighborhoods\nResponse model transferred from periodic cells; frozen encoders, no retraining',fontsize=14)
        fig.supxlabel('Training-source landmarks fit normalization and PaCMAP; held-out points use transform. Independent map axes are arbitrary.',fontsize=9,y=.025)
        for ext in ('png','pdf'):fig.savefig(out/'plots'/f'{field}.{ext}',dpi=180)
        plt.close(fig)
    custom=np.column_stack([data[k][mask] for k in ('source','frame','atom')])
    figure=make_subplots(rows=1,cols=3,subplot_titles=[e['plot_title'].replace('\n','<br>') for e in entries])
    for column,(entry,xy) in enumerate(models,1):
        figure.add_trace(go.Scattergl(x=xy[:,0],y=xy[:,1],mode='markers',name=entry['label'],
            marker=dict(size=3,color=structure,coloraxis='coloraxis',opacity=.7),customdata=custom,
            hovertemplate='Source %{customdata[0]}<br>Frame %{customdata[1]}<br>Atom %{customdata[2]}<extra>%{fullData.name}</extra>'),row=1,col=column)
    scale=[[0,palette[0]],[1/3,palette[0]],[1/3,palette[1]],[2/3,palette[1]],[2/3,palette[2]],[1,palette[2]]]
    source_ids=np.unique(data['source'][mask]);source_index=np.searchsorted(source_ids,data['source'][mask])
    options=[('Present structure',structure,dict(colorscale=scale,cmin=-.5,cmax=2.5,colorbar=dict(title='Structure',tickvals=[0,1,2],ticktext=['Crystal-free','Mixed / noncrystalline center','Crystalline center']))),
        ('Crystalline fraction',fraction,dict(colorscale='Viridis',cmin=0,cmax=1,colorbar=dict(title='Fraction',tickvals=[0,.5,1],ticktext=['0','0.5','1']))),
        ('Source audit',source_index,dict(colorscale='Turbo',cmin=0,cmax=29,colorbar=dict(title='Source',tickvals=list(range(30)),ticktext=source_ids.astype(str).tolist())))]
    buttons=[dict(label=label,method='update',args=[{'marker.color':[values.tolist()]*3},{'coloraxis':axis}]) for label,values,axis in options]
    figure.update_layout(height=650,showlegend=False,template='plotly_white',margin=dict(t=120,b=40),
        title='Three frozen encoders · identical Al64 neighborhoods',coloraxis=options[0][2],
        updatemenus=[dict(buttons=buttons,x=0,y=1.16,direction='right',type='buttons')])
    figure.update_xaxes(showticklabels=False,zeroline=False);figure.update_yaxes(showticklabels=False,zeroline=False)
    for i in range(1,4):figure.update_yaxes(scaleanchor='x' if i==1 else f'x{i}',scaleratio=1,row=1,col=i)
    description=('<div style="font:15px system-ui;margin:24px;max-width:1200px"><h1>Al neighborhoods: response transfer, MM-TDA and GeoFormer</h1>'
        '<p>24,960 matched neighborhoods from30 held-out sources. Fit:9,360 label-free landmarks from90 training sources. '
        'PaCMAP and feature scaling fit training landmarks only; all test points use out-of-sample transform. Colors never enter the projection.</p>'
        '<p><b>Response transfer:</b> frozen mean/variance pooling and learned normalization from256-atom periodic cells, '
        'applied to80-atom open neighborhoods. No periodic patch wrapping, center indicator, retraining or new normalization inside the encoder. '
        'MM-TDA uses its native local encoder; GeoFormer is matched S1/seed17/epoch24, encoder output before its VICReg projector. '
        'No predictor heads, temperature, time, history or motion are inputs.</p>'
        '<p>Independent maps have arbitrary axes. Visual separation is not a predictive performance score. '
        'Hover to see the source/frame/atom identity; use Source audit to inspect source effects.</p></div>')
    html=figure.to_html(full_html=True,include_plotlyjs=True)
    (out/'index.html').write_text(html.replace('<body>','<body>'+description))


def run(config):
    check_metric_docs(family='response_pacmap')
    c=read(config);out=resolve_path(c['output'])
    for name in ('data','plots','technical/requests','technical/projections'):(out/name).mkdir(parents=True,exist_ok=True)
    train,test,plan,manifest=prepare(c,out)
    specs=c['encoders']
    for spec in specs:
        for key in ('checkpoint','producer'):
            spec[key]=str(resolve_path(spec[key]))
        if 'native_parents' in spec:spec['native_parents']=str(resolve_path(spec['native_parents']))
    binding=dict(config=c,assay_manifest_sha256=sha(resolve_path(c['assay_cache'])/'manifest.json'),
        observation_sha256=sha(out/'data/observations.npz'),positions_sha256=sha(out/'data/positions.npy'),
        implementation={name:sha(Path(__file__).with_name(name)) for name in ('response_pacmap.py','response_pacmap_infer.py')},
        mm_inference_source_sha256=sha(Path(__file__).parent/'birth_prediction/infer.py'),
        packages={k:version(k) for k in ('pacmap','numpy','torch','scipy','numba','faiss-cpu','matplotlib','plotly')})
    identity=digest(binding)
    existing=out/'technical/binding.json'
    if existing.exists() and read(existing)['identity'] != identity:
        raise ValueError('Analysis definition changed; use a new output revision')
    write_json(existing,dict(identity=identity,**binding))
    write_json(out/'technical/prediction-context.json',dict(
        encoder_inputs=dict(common='same nearest80 center-relative Al coordinates, all within8A',
            halo=None,history=0,motion=False,conditions=[],relaxation=False,
            response='open-boundary transfer, unit weights, zero center marker, original mean/variance pool and learned normalization',
            mm_tda='native local radial taper and center marker; original fixed Al length factor',
            geoformer='native local geometry divided by9.192189A'),
        predictor_inputs=None,heads_used=False,neural_training=False,track=c['evaluation_track'],
        checkpoint_selection=[dict(name=s['name'],selector=s['selector']) for s in specs]))
    shutil.copy2(config,out/'technical/config.json')
    for name in binding['implementation']:shutil.copy2(Path(__file__).with_name(name),out/'technical'/name)
    shutil.copy2(Path(__file__).parent/'birth_prediction/infer.py',out/'technical/mm_infer.py')
    import pacmap,numba,faiss
    if version('pacmap')!='0.9.1':raise ValueError('Expected PaCMAP0.9.1')
    numba.set_num_threads(2);faiss.omp_set_num_threads(2)
    entries=[];coverage=[]
    cache=RetainedCache(resolve_path(c['feature_cache']),6)
    for spec in specs:
        meta=dict(analysis=identity,spec=spec,positions_sha256=binding['positions_sha256'])
        with cache.lease(digest(meta),deadline=time.time()+14400,metadata=meta) as folder:
            output=folder/'states.npy';receipt=folder/'complete.json'
            if not receipt.exists():
                request=dict(model=spec,positions=str(out/'data/positions.npy'),destination=str(output),
                             batch_size=spec['batch_size'],seed=c['seed'])
                rp=out/'technical/requests'/f'{spec["name"]}.json';write_json(rp,request)
                implementation=Path(__file__).with_name('response_pacmap_infer.py')
                if spec['kind']=='rich':implementation=Path(__file__).parent/'birth_prediction/infer.py'
                with (out/'technical'/f'inference-{spec["name"]}.log').open('w') as log:
                    subprocess.run([sys.executable,'-u',str(implementation),str(rp)],cwd=spec['producer'],
                        env=dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
                write_json(receipt,dict(metadata=meta,sha256=sha(output),inference_source_sha256=sha(implementation)))
            record=read(receipt)
            if record['metadata']!=meta or sha(output)!=record['sha256']:raise ValueError('Changed embeddings')
            write_json(out/'technical'/f'features-{spec["name"]}.json',dict(cache=str(folder),**record))
            if spec['kind'] in ('response','geoformer'):
                shutil.copy2(output.with_suffix('.checks.json'),out/'technical'/f'inference-checks-{spec["name"]}.json')
            z=np.load(output)
            if len(z)!=len(train)+len(test) or not np.isfinite(z).all():raise ValueError('Invalid feature rows')
            path=out/'data'/f'{spec["name"]}-pacmap.npz';pr=out/'technical/projections'/f'{spec["name"]}.json'
            if not pr.exists():
                center=z[:len(train)].astype(np.float64).mean(0)
                scale=z[:len(train)].astype(np.float64).std(0).clip(1e-5)
                x=((z-center)/scale).astype(np.float32);xt=x[:len(train)]
                reducer=pacmap.PaCMAP(**c['pacmap'])
                print(f'PaCMAP {spec["name"]}: {len(train)} train landmarks / {len(test)} held-out',flush=True)
                train_xy=reducer.fit_transform(xt,init='pca')
                test_xy=reducer.transform(x[len(train):],basis=xt,init='pca')
                if not np.isfinite(test_xy).all():raise ValueError('Nonfinite PaCMAP')
                np.savez_compressed(path,train_xy=train_xy,test_xy=test_xy,center=center,scale=scale,train_rows=train,test_rows=test)
                with (out/'technical/projections'/f'{spec["name"]}.pkl').open('wb') as stream:pickle.dump(reducer,stream)
                write_json(pr,dict(identity=identity,embedding_sha256=record['sha256'],coordinates_sha256=sha(path)))
            elif read(pr)['coordinates_sha256']!=sha(path):raise ValueError('Changed projection')
            entries.append(spec)
            coverage.append(dict(encoder=spec['name'],dimensions=z.shape[1],train_landmarks=len(train),
                heldout_rows=len(test),train_sources=90,test_sources=30))
            print(f'Completed {spec["name"]}',flush=True)
    write_metric_rows(coverage,out,family='response_pacmap',name='coverage')
    render(c,out,entries)
    write_json(out/'technical/complete.json',dict(state='complete',identity=identity,encoders=3,
        test_rows=len(test),neural_training=False,page_sha256=sha(out/'index.html')))
    print(f'Published {out}/index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
