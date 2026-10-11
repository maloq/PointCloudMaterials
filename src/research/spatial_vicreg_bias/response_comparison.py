"""Add a selected native response encoder to the saved PaCMAP/MD comparison."""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.analysis.representative_style import sparse_geometry
from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from .cluster_matching import summarize
from .comparison_layout import comparison_template, population_controls, publish_run_entrypoint, model_navigation
from .mace_publication import rebase_assets, contingency, score, FAMILIES
from .sample_lattice import fit_batch
from .viewer_payload import read_payload, read_asset, write_asset, write_vector_asset, write_comparison


def load(config):
    c=json.loads(Path(config).read_text())
    # Resolve operational paths only; training configuration and archived receipts
    # are frozen evidence and retain their producer's original spelling.
    for field in ('output','viewer_root','viewer_archive','checkpoint','producer','assay','static_data'):
        c[field]=str(resolve_path(c[field]))
    for dataset in c['datasets'].values():
        for field in ('reference','publication'):
            dataset[field]=str(resolve_path(dataset[field]))
    if c['protocol']!='local_response_interactive_comparison_v1':raise ValueError('Wrong comparison protocol')
    out=Path(c['output'])
    if json.loads((out/'technical/run-binding.json').read_text())!=c:
        raise ValueError('Frozen comparison definition changed')
    return c,out


def template(c,kind):
    dest=Path(c['datasets'][kind]['publication'])
    options=''.join(f'<option value="../../{key}/interactive/comparison-{population}.html"'
        +(' selected' if key==kind else '')+f'>{label}</option>'
        for key,population,label in [('matched','all_test','Held-out Al MD'),('static','all_static','Al static · 166–240 ps')])
    nav='<nav class="controls"><label>Data <select onchange="location.href=this.value">'+options+'</select></label>'+model_navigation(kind,'response',dest/'interactive')+'</nav>'
    methods=('<details><summary>Methods and checkpoint</summary><p>Response MACE128: weight 1, seed 20261002, update 250. '
        'Selected by the lowest validation future-feature Gaussian NLL among the 12 response-weight/schedule fits (0.99494). '
        'Encoder input: centered nearest-80 geometry, constant atom channel and center indicator, radius 8 Å, edge cutoff 5 Å. '
        'The native normalized 128-dimensional embedding is exported before the prediction head. '
        'Training predicted 20/100-fs future Fourier features and their directional responses under a MACE teacher. '
        'This is a structural comparison; those short-time targets do not establish crystallization prediction.</p>'
        '<p>K7 clustering fits the original 74,880 uniform training-source observations. '
        'All saved descriptor clusters, descriptor projections and older neural results are reused. '
        'Neural PaCMAP uses the same 24,960 held-out or 24,000 static observations. '
        'Color assignment maximizes shared membership and stays fixed across frames. '
        'The six static snapshots are already relaxed; the response encoder trained on unrelaxed patches.</p>'
        '<p><a id="metricDefinitions" href="../tables/METRICS.md">Metric definitions</a> · '
        '<a id="metricTable" href="../tables/metrics.csv">Correspondence table</a></p></details>')
    controls=dict(FRAME_CONTROLS=population_controls()) if kind=='matched' else {}
    return comparison_template(nav,HEADLINE='Response encoder ↔ descriptors',METHODS=methods,**controls)


def samples(folder,dest,key,identity):
    with np.load(folder/'samples.npz') as z:a=dict(z)
    examples={};clusters=[]
    for cid in range(7):
        take=np.flatnonzero(a['clusters']==cid)
        clusters.append(dict(count=int(a['counts'][cid]),rows=a['rows'][take].tolist()))
        for i in take:
            xyz=a['patches'][i];r=np.sort(np.linalg.norm(xyz,axis=1))
            oriented,edges,_=sparse_geometry(xyz,float(1.2*np.median(r[1:13])),orientation='pca')
            examples[str(a['rows'][i])]=dict(atom=int(a['atoms'][i]),xyz=oriented.tolist(),edges=edges)
    name=key+'-'+identity
    write_asset(dest/'sample-data'/f'{name}.js',name,dict(clusters=clusters,patches=examples),'CLUSTER_SAMPLES')
    rows=list(examples);fits=fit_batch([examples[row]['xyz'] for row in rows])
    write_asset(dest/'lattice-data'/f'{name}.js',name,dict(zip(rows,fits)),'LATTICE_SAMPLES')
    return dict(key=name,asset='../sample-data/'+name+'.js'),dict(key=name,asset='../lattice-data/'+name+'.js')


def publish(config):
    c,out=load(config);identity=c['model']['id']
    inference=json.loads((out/'technical/inference.json').read_text())
    if inference['checkpoint_sha256']!=c['checkpoint_sha256']:raise ValueError('Different inference checkpoint')
    import faiss,numba,pacmap
    if version('pacmap')!=c['pacmap']['version']:raise ValueError('PaCMAP version changed')
    faiss.omp_set_num_threads(2);numba.set_num_threads(2)
    for kind,ds in c['datasets'].items():
        reference,dest=Path(ds['reference']),Path(ds['publication'])
        for name in ('data','assets','interactive','projection-data','md-data','travel-data','sample-data','lattice-data','technical/rendering'):
            (dest/name).mkdir(parents=True,exist_ok=True)
        base=read_payload(reference/'index.html')
        if base['descriptor_fit']['population']!='all training environments':raise ValueError('Wrong descriptor fit')
        d=rebase_assets(copy.deepcopy(base),reference,dest)
        with np.load(out/'data'/('heldout.npz' if kind=='matched' else 'static.npz')) as z:
            vectors,labels=z['z'],z['labels']
            for field in ('frame','atom'):
                if not np.array_equal(z[field],d[field]):raise ValueError('Changed display identities '+field)
        field='neural:'+identity;d['fields'][field]=labels.tolist()
        d['frozen_model']=c['model'];d['publication_recipe']=str(Path(config).resolve())
        d['descriptor_fit']['metrics']='../tables/METRICS.md'
        model=dict(id=identity,title=c['model']['title'],representation='encoder',snapshots={})
        dense_tables={f:np.zeros((7,7),np.int64) for f in FAMILIES};snapshots={}
        for snap in d['md']['snapshots']:
            key=snap['key'];folder=out/'data'/kind/key;name=key+'-'+identity
            with np.load(folder/'labels.npz') as z:assigned=z['encoder']
            physical=read_asset(dest/'interactive'/snap['asset'])
            descriptors={f:np.asarray(physical['fields'][df],np.uint8) for f,df in FAMILIES.items()}
            if len(assigned)!=snap['count']:raise ValueError('Incomplete dense assignments '+key)
            snapshots[key]=(assigned,descriptors)
            for family in FAMILIES:dense_tables[family]+=contingency(assigned,descriptors[family])
            write_asset(dest/'md-data'/f'{name}.js',name,dict(encoder=assigned.tolist()),'MD_NEURAL')
            model['snapshots'][key]=dict(key=name,asset='../md-data/'+name+'.js')
            sample,lattice=samples(folder,dest,key,identity)
            d['samples'][key]['neural'][identity]=sample;d['lattice'][key]['neural'][identity]=lattice
            with np.load(folder/'travel.npz') as z:
                write_vector_asset(dest/'travel-data'/f'{name}.js',name,z['z'],dict(neighbors=z['neighbors'].tolist(),clusters=z['labels'].tolist()))
            d['travel'][key]['models'][identity]=dict(key=name,asset='../travel-data/'+name+'.js')
        d['md']['models'].append(model);d['md']['default_model']=identity
        matching={};metrics={}
        for family,df in FAMILIES.items():
            right=np.asarray(d['fields'][df]);table=contingency(labels,right) if kind=='matched' else dense_tables[family]
            rows,mapping=linear_sum_assignment(table,maximize=True)
            if not np.array_equal(rows,np.arange(7)):raise ValueError('Incomplete optimal color assignment')
            matching[family]=dict(neural_to_descriptor=mapping.tolist(),contingency=table.tolist(),reference=summarize(table,mapping))
            pooled=summarize(table,mapping)
            pooled['adjusted_rand_index']=float(adjusted_rand_score(labels,right)) if kind=='matched' else float(adjusted_rand_score(
                np.concatenate([v[0] for v in snapshots.values()]),np.concatenate([v[1][family] for v in snapshots.values()])))
            metrics[family]=dict(reference=pooled,display=score(labels,right,mapping),
                snapshots={key:score(left,desc[family],mapping) for key,(left,desc) in snapshots.items()})
        d['matching'][identity]=matching
        path=dest/'data/pacmap.npz'
        if not path.exists():
            params={k:v for k,v in c['pacmap'].items() if k!='version'}
            y=pacmap.PaCMAP(n_components=3,**params).fit_transform(np.asarray(vectors,np.float32),init='pca')
            if not np.isfinite(y).all():raise ValueError('Nonfinite projection')
            np.savez_compressed(path,y3=y)
        with np.load(path) as z:y=z['y3']
        name=identity+'-'+kind
        write_asset(dest/'projection-data'/f'{name}.js',name,dict(y3=np.round(y,6).tolist(),clusters=labels.tolist()),'PACMAP_LAYOUTS')
        d['paired']['neural'].append(dict(id=identity,title=c['model']['title'],representation='encoder',field=field,key=name,asset='../projection-data/'+name+'.js'))
        d['paired']['default_neural']=identity
        for filename in ('cluster_comparison.js','viewer_extensions.js'):
            shutil.copy2(Path(__file__).with_name(filename),dest/'assets'/filename)
        shutil.copy2(reference/'assets/plotly.min.js',dest/'assets/plotly.min.js')
        write_metric_table(metrics,dest,family='response_cluster_comparison')
        write_json(dest/'technical/matches.json',matching)
        population='all_test' if kind=='matched' else 'all_static'
        write_comparison(dest/'interactive'/f'comparison-{population}.html',template(c,kind),d,title='Response encoder and descriptor clusters')
        write_comparison(dest/'index.html',template(c,kind),d,title='Response encoder and descriptor clusters',index=True)
        publish_run_entrypoint(dest,kind)
        write_json(dest/'technical/rendering/provenance.json',dict(checkpoint_sha256=c['checkpoint_sha256'],
            reference=str(reference),reference_payload_sha256=sha(reference/'index.json'),
            clusters_sha256=sha(out/'data/clusters.npz'),projection_sha256=sha(path),
            input_record=str(out/'technical/inference.json'),model=c['model'],neural_training=False,descriptor_refit=False,
            original_observations_preserved=True,old_neural_results_preserved=True))
        print('Published '+str(dest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=['publish','restore']);parser.add_argument('--config',required=True)
    args=parser.parse_args()
    if args.action=='restore':
        from .viewer_recovery import restore
        restore(args.config)
    else:publish(args.config)
