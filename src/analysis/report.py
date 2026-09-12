"""Flat, portable galleries of artifacts produced by the standard analysis pipeline."""

import fcntl
import html
import json
import os
from pathlib import Path
import shutil

from omegaconf import OmegaConf


def report_directory(cfg, analysis_cfg):
    root = OmegaConf.select(analysis_cfg, 'report.root')
    if root is None:
        return None
    variant = OmegaConf.select(cfg, 'experiment_variant')
    name = (f'{variant}-seed{cfg.seed_everything}' if variant is not None
            else OmegaConf.select(analysis_cfg, 'report.model_name', default=cfg.experiment_name))
    return Path(root)/str(name).lower().replace('_', '-')


def _write(path, text):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def publish_report(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    metrics = json.loads((source/'analysis_metrics.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination/'source.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['checkpoint_sha256'] != metrics['topology']['checkpoint_sha256']:
            raise FileExistsError(f'{destination} already belongs to {previous["analysis_directory"]}. '
                                  'Choose a distinct report.root for another experiment.')
    k = int(metrics['clustering']['primary_k'])
    files = {
        'tsne.png': 'latent_tsne_clusters.png',
        'umap.png': 'latent_umap_clusters.png',
        'pca.png': 'latent_pca_analysis.png',
        'pca-3d.png': 'latent_pca_3d.png',
        'latent-statistics.png': 'latent_statistics.png',
        'representatives.png': f'real_md/representatives/04_cluster_representatives_k{k}_pca_reciprocal.png',
        'representatives-bonds.png': f'real_md/representatives/09_cluster_representatives_knn_edges_k{k}.png',
        'representatives.html': 'real_md/representatives/12_cluster_representatives_3d.html',
        'md-umap.png': 'real_md/latent/latent_projection_umap_clusters.png',
        'md-pca.png': 'real_md/latent/latent_projection_pca_clusters.png',
        'cluster-proportions.png': 'real_md/time_series/cluster_proportions_stacked_area.png',
        'transitions.png': 'real_md/transitions/transition_aggregate_flow.png',
        'metrics.json': 'analysis_metrics.json',
        'topology.json': 'topology/metrics.json',
    }
    for snapshot in sorted((source/'snapshots').glob('*')):
        for extension in ('png', 'html'):
            files[f'spatial-{snapshot.name}.{extension}'] = (
                f'snapshots/{snapshot.name}/md_space/md_space_clusters_k{k}.{extension}')
        files[f'representatives-{snapshot.name}.png'] = (
            f'snapshots/{snapshot.name}/figure_set_k{k}/04_cluster_representatives_k{k}_pca_reciprocal.png')
    for path in sorted((source/'connected_regimes').glob('*.png')):
        files[path.name.replace('_', '-')] = str(path.relative_to(source))
    for path in sorted((source/'real_md/representatives').glob('11_*.html')):
        files[path.name.removeprefix('11_').replace('_', '-')] = str(path.relative_to(source))
    published = {}
    for name, relative in files.items():
        original = source/relative
        if not original.exists():
            continue  # These plots are optional stages of the analysis pipeline.
        temporary = destination/f'.{name}.{os.getpid()}.tmp'
        shutil.copy2(original, temporary)
        temporary.replace(destination/name)
        published[name] = relative
    cards, links = [], []
    for name in published:
        label = name.rsplit('.', 1)[0].replace('-', ' ')
        if name.endswith('.png'):
            cards.append(f'<figure><a href="{html.escape(name)}"><img loading="lazy" '
                         f'src="{html.escape(name)}" alt="{html.escape(label)}"></a>'
                         f'<figcaption>{html.escape(label)}</figcaption></figure>')
        else:
            links.append(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>')
    umap_status = 'UMAP available.' if 'umap.png' in published else 'UMAP has not been generated for this report yet.'
    title = destination.name
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>{html.escape(title)}</title><style>body{{font:16px system-ui;margin:2rem;background:#f5f6f8;color:#202530}}'
        'a{color:#2454a6}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}'
        'figure{margin:0;background:white;padding:1rem;border-radius:8px}img{width:100%;height:auto}'
        'figcaption{padding-top:.6rem}li{margin:.4rem 0}</style>'
        f'<a href="../index.html">All runs</a><h1>{html.escape(title)}</h1><p>{umap_status}</p>'
        f'<ul>{"".join(links)}</ul><div class="plots">{"".join(cards)}</div>')
    _write(destination/'index.html', page)
    _write(destination/'README.md', f'# {title}\n\nOpen [the gallery](index.html).\n\n{umap_status}\n\n'
           + '\n'.join(f'- [{name}]({name})' for name in published)+'\n')
    _write(manifest_path, json.dumps(dict(analysis_directory=str(source), files=published,
        checkpoint_sha256=metrics['topology']['checkpoint_sha256']), indent=2)+'\n')
    with (destination.parent/'.index.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reports = sorted(p.parent for p in destination.parent.glob('*/source.json'))
        entries = ''.join(f'<li><a href="{html.escape(p.name)}/index.html">{html.escape(p.name)}</a></li>' for p in reports)
        _write(destination.parent/'index.html', '<!doctype html><meta charset="utf-8"><title>MACE results</title>'
            '<style>body{font:18px system-ui;margin:3rem}li{margin:.7rem 0}</style>'
            f'<h1>MACE results</h1><ul>{entries}</ul>')
        _write(destination.parent/'README.md', '# MACE results\n\n'
               + '\n'.join(f'- [{p.name}]({p.name}/index.html)' for p in reports)+'\n')
    print(f'[analysis] Gallery: {destination/"index.html"}', flush=True)
