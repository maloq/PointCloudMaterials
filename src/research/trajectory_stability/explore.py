"""Standalone, offline Plotly explorer for every audited atom trajectory."""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

from .report import METHODS, LABELS, COLORS, features
from .metrics import reference_statistics


def export(plan):
    root = Path(plan['config']['output'])
    training = {m: [] for m in METHODS}
    test = []
    for source in plan['sources']:
        folder = root/'technical/sources'/str(source['id'])
        a = dict(np.load(folder/'observations.npz'))
        f = features(folder, a)
        if source['split'] == 'train':
            for m in METHODS:
                training[m].append(f[m])
        else:
            test.append((source, a, f))
    references, axes = {}, {}
    for m in METHODS:
        x = np.concatenate(training[m]).astype(float)
        references[m] = reference_statistics(x)
        axes[m] = np.linalg.svd(x-references[m]['mean'], full_matrices=False)[2][0]
    tracks = []
    for source, a, f in test:
        nf, nc = len(a['frames']), len(a['centers'])
        f = {m: v.reshape(nf, nc, -1).astype(float) for m, v in f.items()}
        for c, atom in enumerate(a['centers']):
            pc, jumps = {}, {}
            for m in METHODS:
                pc[m] = np.round((f[m][:, c]-references[m]['mean'])@axes[m]/references[m]['scale'], 5).tolist()
                jumps[m] = np.round(np.linalg.norm(np.diff(f[m][:, c], axis=0), axis=1)/references[m]['scale'], 5).tolist()
            qbar = a['order'].reshape(nf, nc, 8)[:, c, 4]
            crystalline = np.isin(a['labels'].reshape(nf, nc)[:, c], [1, 2, 3])
            tracks.append(dict(label=f'{source["temperature_K"]:g} K · source {source["id"]} · atom {atom}',
                pc=pc, jumps=jumps, qbar=np.round(qbar, 5).tolist(), crystal=np.where(crystalline, .7, 0).tolist()))
    time = test[0][1]['times_ps'].tolist()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07,
        subplot_titles=['Training PC1 (one separate projection per method)', 'Adjacent-frame jump', 'Physical structural context'])
    initial = tracks[0]
    for m in METHODS:
        visible = True if m in ['mace', 'gatr', 'tda', 'soap'] else 'legendonly'
        fig.add_trace(go.Scatter(x=time, y=initial['pc'][m], name=LABELS[m], legendgroup=m,
            line=dict(color=COLORS[m], width=1), visible=visible), row=1, col=1)
    for m in METHODS:
        visible = True if m in ['mace', 'gatr', 'tda', 'soap'] else 'legendonly'
        fig.add_trace(go.Scatter(x=time[1:], y=initial['jumps'][m], name=LABELS[m], legendgroup=m,
            showlegend=False, line=dict(color=COLORS[m], width=1), visible=visible), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=initial['crystal'], name='PTM crystalline', line=dict(width=0),
        fill='tozeroy', fillcolor='rgba(0,158,115,.12)', hoverinfo='skip'), row=3, col=1)
    fig.add_trace(go.Scatter(x=time, y=initial['qbar'], name='q̄6 bond order', line=dict(color='#333333', width=1.1)), row=3, col=1)
    fig.update_layout(height=900, template='plotly_white', margin=dict(l=80, r=20, t=40, b=50),
        legend=dict(orientation='h', y=1.10, groupclick='togglegroup'), hovermode='x unified',
        dragmode='zoom', uirevision='stability')
    fig.update_yaxes(title_text='PC1 / reference distance', row=1, col=1)
    fig.update_yaxes(title_text='Normalized jump', rangemode='tozero', row=2, col=1)
    fig.update_yaxes(title_text='q̄6', range=[0, .7], row=3, col=1)
    fig.update_xaxes(title_text='Time (ps)', row=3, col=1)
    fragment = fig.to_html(full_html=False, include_plotlyjs=False, div_id='trajectory-plot', config={'responsive': True, 'displaylogo': False})
    options = ''.join(f'<option value="{i}">{t["label"]}</option>' for i, t in enumerate(tracks))
    script = '\nconst tracks='+json.dumps(tracks, separators=(',', ':'))+';\nconst methods='+json.dumps(METHODS)+';\n'+'''
document.getElementById('track').addEventListener('change', function () {
  const track = tracks[Number(this.value)];
  const y = methods.map(m => track.pc[m]).concat(methods.map(m => track.jumps[m]), [track.crystal, track.qbar]);
  Plotly.restyle('trajectory-plot', {y: y});
});
'''
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+\
        '<title>Explore Al trajectory stability</title><style>body{font:16px system-ui;margin:24px;color:#24313d}select{font:inherit;padding:6px;max-width:100%}p{max-width:1100px}a{color:#2066a8}</style>'+\
        '<script>'+get_plotlyjs()+'</script><h1>Explore the tracked trajectories</h1>'+\
        '<p>Choose an atom; toggle representations using the legend. Drag to zoom into any interval, double-click to reset. '+\
        'All curves are unsmoothed. Separate training data define the projections and distance scales.</p>'+\
        '<p><label for="track">Trajectory: </label><select id="track">'+options+'</select></p>'+fragment+\
        '<p>Jumps include physical motion, changing neighbors and stored-coordinate quantization. '+\
        'PC1 is a different projection for each method; its sign is arbitrary. PTM-unclassified is not necessarily liquid. '+\
        '<a href="index.html">Comparison gallery</a> · <a href="RESULTS.md">Interpretation</a></p><script>'+script+'</script></html>'
    (root/'explore.html').write_text(html)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    export(json.loads(Path(args.plan).read_text()))
