"""Compare frozen embedding geometry with topology geometry; fit no readout."""

import csv
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from .run import data_audit, read_json


def distances(values, metric):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite input to direct representation distances')
    if metric == 'cosine' and np.any(np.linalg.norm(values,axis=1)==0):
        raise ValueError('Cosine distance is undefined for a zero embedding')
    result = cdist(values, values, metric=metric)
    if not np.isfinite(result).all():
        raise ValueError(f'Undefined {metric} distances')
    np.fill_diagonal(result,0.)
    return result


def neighbors(matrix, k):
    """Exclude self; exact distance ties use the original row order."""
    if not 0 < k < len(matrix)-1:
        raise ValueError(f'Need 0 < k < N-1; got k={k}, N={len(matrix)}')
    masked = matrix.copy()
    np.fill_diagonal(masked,np.inf)
    ordered = np.argsort(masked,axis=1,kind='stable')[:,:k+1]
    boundary = np.take_along_axis(masked,ordered[:,-2:],axis=1)
    return ordered[:,:k], float(np.mean(boundary[:,0]==boundary[:,1]))


def overlap(first, second):
    return (first[:,:,None]==second[:,None,:]).any(axis=2).mean(axis=1)


def compare_geometry(embedding, topology, pairs, k, permutation):
    """Topology blocks retain raw units; no standardizer or linear mapping."""
    first, embedding_ties = neighbors(embedding,k)
    # D'[i,j] = D[p[i],p[j]]; remap neighbors without materializing D'.
    inverse = np.argsort(permutation)
    shuffled = inverse[first[permutation]]
    rows, arrays = [], dict(embedding_neighbors=first, pairs=pairs,
        embedding_pair_distances=embedding[pairs[:,0],pairs[:,1]])
    for name, matrix in topology.items():
        second, target_ties = neighbors(matrix,k)
        target_distances=matrix[pairs[:,0],pairs[:,1]]
        rho=float(spearmanr(arrays['embedding_pair_distances'],target_distances).statistic)
        null_rho=float(spearmanr(embedding[permutation[pairs[:,0]],permutation[pairs[:,1]]],target_distances).statistic)
        if not np.isfinite(rho) or not np.isfinite(null_rho):
            raise ValueError(f'Undefined rank correlation for {name}; inspect constant distances')
        shared=overlap(first,second)
        rows.append(dict(block=name, spearman=rho, neighbor_overlap=float(shared.mean()),
            shuffled_spearman=null_rho, shuffled_neighbor_overlap=float(overlap(shuffled,second).mean()),
            chance_neighbor_overlap=k/(len(matrix)-1), embedding_boundary_tie_fraction=embedding_ties,
            target_boundary_tie_fraction=target_ties, samples=len(matrix), pairs=len(pairs)))
        arrays[name+'_neighbors']=second
        arrays[name+'_pair_distances']=target_distances
        arrays[name+'_row_overlap']=shared
    return rows,arrays


def summarize(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=[]
    for method in ['random','mlip']:
        for metric in ['euclidean','cosine']:
            for scope in ['all_test','within_frame']:
                selected=[r for r in rows if r['method']==method and r['metric']==metric and r['scope']==scope]
                names=['spearman','neighbor_overlap','shuffled_spearman','shuffled_neighbor_overlap','chance_neighbor_overlap']
                summary.append(dict(method=method,metric=metric,scope=scope,
                                    **{name:float(np.mean([r[name] for r in selected])) for name in names}))
    snapshot_metric_docs(root,'mace_tda_ridge_audit')
    for filename,items in [('scores',rows),('summary',summary)]:
        with (root/'tables'/f'{filename}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    write_json(root/'technical/summary.json',summary)
    colors={'random':'#9b6a41','mlip':'#2276a8'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(scope,value) in zip(axes.flat,[('all_test','spearman'),('all_test','neighbor_overlap'),
                                        ('within_frame','spearman'),('within_frame','neighbor_overlap')]):
        factor=100 if value=='neighbor_overlap' else 1
        for method,offset in [('random',-.18),('mlip',.18)]:
            scores=[next(r[value] for r in summary if r['method']==method and r['metric']==metric and r['scope']==scope)*factor
                    for metric in ['euclidean','cosine']]
            ax.bar(np.arange(2)+offset,scores,width=.34,color=colors[method],label='Random (3 seeds)' if method=='random' else 'Original MLIP')
            for i,score in enumerate(scores):
                ax.annotate(f'{score:.3f}' if factor==1 else f'{score:.2f}%',
                            (i+offset,score),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks([0,1],['Raw Euclidean distance','Cosine distance'])
        ax.set_ylabel('Mean H0/H1/H2 rank correlation' if factor==1 else 'Mean top-10 neighbor overlap (%)')
        ax.set_title('All 4,608 test structures' if scope=='all_test' else 'Compare only within each simulation frame')
        chance=0. if factor==1 else next(r['chance_neighbor_overlap'] for r in summary if r['scope']==scope)*100
        ax.axhline(chance,color='gray',linestyle='--',linewidth=1,label='Random correspondence')
        ax.grid(axis='y',alpha=.2)
        maximum=max(r[value]*factor for r in summary if r['scope']==scope)
        ax.set_ylim(0,1 if factor==1 else maximum*1.30)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Frozen MACE: direct comparison with topology, without a trained readout',fontsize=13)
    fig.savefig(root/'plots/direct-topology-comparison.png',dpi=180)
    fig.savefig(root/'plots/direct-topology-comparison.pdf');plt.close(fig)
    lines=['# Direct frozen-embedding topology comparison','',
        'No encoder updates, trained readout, feature standardizer, fitted projection or hyperparameter search. All 4,608 held-out test neighborhoods are used. Original MLIP is one checkpoint; random MACE averages three seeds. Each summary equally averages H0/H1/H2; within-frame summaries additionally average 18 frames.',
        '', '| Encoder | Distance | Scope | Rank correlation | Top-10 overlap | Chance overlap |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['metric']} | {r['scope']} | {r['spearman']:.4f} | {100*r['neighbor_overlap']:.2f}% | {100*r['chance_neighbor_overlap']:.2f}% |")
    lines += ['', '![Direct comparison](plots/direct-topology-comparison.png)', '',
        'Rank correlation compares pairwise embedding distances with pairwise raw TDA-block distances. Higher means better agreement about which structures are similar. Neighbor overlap compares the two top-10 neighbor sets for every structure; self is excluded. These are representation-geometry metrics, not descriptor prediction errors or crystallization accuracies.', '',
        'The global rank score uses 100,000 distinct unordered pairs chosen once with a fixed seed. Within each frame it uses every unordered pair. Nearest-neighbor searches use all eligible structures. The per-frame restriction controls temperature/time/source context. A shuffled-row control is exported for each block, model, distance and scope. No independence-based p-values are reported for overlapping pairs.', '',
        'Targets are the original relaxed 80-atom, 144D alpha-complex descriptors: 16 H0 death-radius samples and 8x8 birth/lifetime surfaces each for H1 and H2. Details and exact definitions are in tables/METRICS.md.', '',
        'Reproduce: `python -m src.research.mace_tda_ridge_audit.run --config configs/analysis/mace_tda_direct.json --stage direct` with a fresh output.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def run_direct(config):
    root=Path(config['output'])
    for name in ['technical','tables','plots']:(root/name).mkdir(parents=True,exist_ok=True)
    if (root/'technical/summary.json').exists():
        raise FileExistsError(f'Use a fresh output for direct comparison: {root}')
    write_json(root/'technical/status.json',dict(state='running',stage='data-audit'))
    write_json(root/'technical/config.json',config)
    data=data_audit(config,root)
    reference=Path(load_json(config['initialization_recipe'])['output'])
    if read_json(root/'technical/data-audit.json') != read_json(reference/'technical/data-audit.json'):
        raise ValueError('Direct comparison and frozen feature data identities differ')
    manifest=read_json(Path(config['cache'])/'manifest.json')
    original=read_json(manifest['protocol']['source_manifest'])
    producer=Path('src/analysis/liquid_structure.py')
    if any(r['provenance']['producer'][str(producer)] != sha256(producer) for r in original['shards']):
        raise ValueError('Documented topology producer differs from the cached target producer')
    test=data['indices']['test'];target=data['targets'][test]
    contexts=data['contexts'][test];sources=data['sources'][test]
    np.savez(root/'technical/identities.npz',test_indices=test,contexts=contexts,sources=sources,targets=target)
    all_pairs=np.column_stack(np.triu_indices(len(test),k=1))
    chosen=np.random.default_rng(config['seed']).choice(len(all_pairs),config['global_pairs'],replace=False)
    global_pairs=all_pairs[chosen];del all_pairs
    scopes=[('all_test',-1,np.arange(len(test)),global_pairs)]
    for context in np.unique(contexts):
        ids=np.flatnonzero(contexts==context)
        scopes.append(('within_frame',int(context),ids,np.column_stack(np.triu_indices(len(ids),k=1))))
    rows=[];provenance=[]
    # Scope outside model loop avoids repeated TDA distance construction.
    for scope,context,ids,pairs in scopes:
        topology={f'H{d}':distances(target[ids,block],'euclidean')
                  for d,block in enumerate([slice(0,16),slice(16,80),slice(80,144)])}
        permutation=np.random.default_rng(np.random.SeedSequence([config['seed'],context+1])).permutation(len(ids))
        for item in config['encoders']:
            directory=reference/'technical'/item['name']
            inference=read_json(directory/'inference.json')
            if sha256(directory/'features.npz') != inference['features_sha256'] or not inference['frozen_state_unchanged']:
                raise ValueError(f'Frozen feature provenance failed: {directory}')
            features=np.load(directory/'features.npz')['encoder'][test[ids]]
            if features.shape != (len(ids),256):raise ValueError(f'Wrong raw encoder shape: {features.shape}')
            for metric in ['euclidean','cosine']:
                computed,arrays=compare_geometry(distances(features,metric),topology,pairs,config['neighbors'],permutation)
                for r in computed:
                    rows.append(dict(method=item['method'],seed=item['seed'],metric=metric,scope=scope,context=context,
                                     source=-1 if context==-1 else int(sources[ids[0]]),**r))
                name=f"{item['name']}-{metric}-{context}"
                np.savez(root/'technical'/f'{name}.npz',indices=test[ids],permutation=permutation,**arrays)
            if scope=='all_test':provenance.append(dict(**item,directory=str(directory),inference=inference))
        print('DIRECT',scope,context,'rows',len(ids),flush=True)
        write_json(root/'technical/status.json',dict(state='running',stage='direct-geometry',scope=scope,context=context))
    write_json(root/'technical/provenance.json',dict(encoders=provenance,topology_producer_sha256=sha256(producer),
        fitted_parameters=0,feature_standardization=False,tda_block_scaling=False,hyperparameter_search=False))
    summarize(root,rows)
    for relative in read_json(root/'technical/metric-contract.json')['files']:
        destination=root/'technical/source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(relative,destination)
    write_json(root/'technical/status.json',dict(state='complete',encoders=len(config['encoders']),test_samples=len(test)))
