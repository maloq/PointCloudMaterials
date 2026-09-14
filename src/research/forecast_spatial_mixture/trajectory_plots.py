"""Static figures and offline 3D time sliders for measured/forecast local paths."""

import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .trajectory_projection import channel_contributions, time_blocks
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = ['#7c3aed', '#e58214', '#db2777', '#0891b2']
PAST, FUTURE, MEAN = '#2563a6', '#15803d', '#171717'
TITLES = dict(correct='Well-timed forecast', early='Early forecast', missed='Missed transition',
              **{'false-alarm': 'False alarm within 9 ps'})
PRIMARY = 'history12_spatial_mixture4'
METHODS = {'history12_deterministic': ('12 ps history', '#60758c'),
           'history12_spatial': ('+ spatial context', '#b65b21'),
           PRIMARY: ('Spatial mixture mean', MEAN)}
PTM = {0: 'Other', 1: 'FCC', 2: 'HCP', 3: 'BCC', 4: 'ICO', 5: 'SC',
       6: 'Cubic diamond', 7: 'Hex. diamond', 8: 'Graphene'}


def save(fig, output, name):
    fig.savefig(output/'plots'/f'{name}.png', dpi=175, facecolor='white')
    fig.savefig(output/'plots'/f'{name}.pdf', facecolor='white')
    plt.close(fig)


def title(e):
    return (f"{TITLES[e['category']]} · {e['source']['temperature_K']:g} K · "
            f"source {e['source']['source_index']}, atom {e['atom_id']}")


def cloud_colors(a, frame):
    retained = np.isin(a['neighbor_atom_ids'][frame], a['neighbor_atom_ids'][16])
    colors = np.where(retained, '#5192c5', '#e9a044').tolist()
    colors[0] = '#159447' if a['physical_crystal'][frame] else '#313949'
    return colors


def draw_cloud(ax, a, frame, limit):
    xyz = a['cloud_A'][frame]
    colors = cloud_colors(a, frame)
    ax.scatter(*xyz[13:].T, c=colors[13:], s=11, alpha=.3, depthshade=True)
    ax.scatter(*xyz[1:13].T, c=colors[1:13], s=32, alpha=.9, depthshade=True)
    ax.scatter(*xyz[:1].T, c=colors[:1], s=85, edgecolors='black', linewidths=.6, depthshade=False)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=34)
    ax.set_axis_off()


def montage(output, examples, arrays):
    fig = plt.figure(figsize=(17, 12), layout='constrained')
    rows = fig.subfigures(4, 1, hspace=.05)
    for subfig, e, a in zip(rows, examples, arrays):
        limit = np.max(abs(a['cloud_A']))*1.05
        axes = subfig.subplots(1, 7, subplot_kw=dict(projection='3d'))
        subfig.suptitle(title(e), fontsize=12, x=.02, ha='left')
        for col, frame in enumerate([0, 8, 12, 16, 20, 24, 28]):
            ax = axes[col]
            draw_cloud(ax, a, frame, limit)
            label = f"{a['time_ps'][frame]:+g} ps · {PTM[int(a['ptm_labels'][frame])]}"
            ax.set_title(label, fontsize=10, color=FUTURE if a['physical_crystal'][frame] else '#303a4a', pad=0)
    fig.suptitle('Measured local point clouds through time\n'
                 'The same center atom; 80 nearest atoms reselected each frame; fixed camera and scale within each row', fontsize=15)
    fig.supxlabel('0 ps = forecast origin. Blue: atoms present at origin. Amber: other neighbors. '
                  'Green center: PTM crystal. The nearest 12 neighbors are emphasized.\n'
                  'Coordinates are measured, not decoded predictions. Scale and camera stay fixed within each row.', fontsize=10)
    save(fig, output, 'point-cloud-evolution')


def path_legend():
    return [Line2D([], [], color=PAST, lw=2, label='Observed history'),
            Line2D([], [], color=FUTURE, lw=2.5, label='Actual future'),
            Line2D([], [], color=MEAN, lw=2, ls='--', label='Mixture mean'),
            Line2D([], [], color='#9ca3af', lw=1, label='Sampled futures')]+[
            Line2D([], [], color=c, lw=2, label=f'Component {i+1}') for i, c in enumerate(COLORS)]


def joined(a, path):
    return np.concatenate([a['true_map'][16:17], path])


def embedding_futures(output, examples, arrays, projection):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
    for ax, e, a in zip(axes.flat, examples, arrays):
        for i, path in enumerate(a['component_map']):
            path = joined(a, path)
            ax.plot(*path[:, :2].T, c=COLORS[i], lw=1.8, marker='.', ms=3)
            ax.text(*path[-1, :2], f' {i+1}', color=COLORS[i], fontsize=10)
        mean = joined(a, a[PRIMARY+'_mean_map'])
        ax.plot(*mean[:, :2].T, c=MEAN, lw=2, ls='--')
        ax.plot(*a['true_map'][:17, :2].T, c=PAST, lw=2, marker='.', ms=4)
        ax.plot(*a['true_map'][16:, :2].T, c=FUTURE, lw=2.6, marker='o', ms=3)
        ax.scatter(*a['true_map'][16, :2], c='black', s=95, marker='*', zorder=10)
        for t in [3, 6, 9]:
            ax.annotate(f'+{t}', a['true_map'][16+int(t/.75), :2], fontsize=8, xytext=(4, 4), textcoords='offset points')
        weights = ', '.join(f'{i+1}: {p:.0%}' for i, p in enumerate(a['component_weight']))
        ax.set(title=title(e)+'\nBranch weights '+weights,
               xlabel='UMAP 1', ylabel='UMAP 2')
        ax.grid(alpha=.17)
    fig.suptitle('Observed embedding paths and alternative 9 ps futures in UMAP\n'
                 'Four component means; sampled paths can be toggled in the interactive explorers', fontsize=15)
    fig.legend(handles=[h for h in path_legend() if h.get_label() != 'Sampled futures'],
               loc='outside lower center', ncol=4, fontsize=10)
    save(fig, output, 'embedding-futures')


def component_time(output, examples, arrays, projection):
    fig, axes = plt.subplots(4, 2, figsize=(14, 13), layout='constrained')
    future = np.arange(1, 13)*.75
    for row, (e, a) in enumerate(zip(examples, arrays)):
        for dim in range(2):
            ax = axes[row, dim]
            for sample in a['sample_map']:
                ax.plot(future, sample[:, dim], c='#999faa', alpha=.2, lw=.65)
            for k in range(4):
                ax.plot(future, a['component_map'][k, :, dim], c=COLORS[k], lw=1.4)
            ax.plot(future, a[PRIMARY+'_mean_map'][:, dim], c=MEAN, ls='--', lw=1.8)
            ax.plot(a['time_ps'][:17], a['true_map'][:17, dim], c=PAST, lw=1.7)
            ax.plot(a['time_ps'][16:], a['true_map'][16:, dim], c=FUTURE, lw=2.3, marker='.', ms=3)
            ax.axvline(0, c='black', lw=.7)
            delay = e['onset_delay_ps']
            if delay is not None and 0 < delay <= 9:
                ax.axvline(delay, c=FUTURE, ls=':', alpha=.7)
            ax.set(xlim=(-12, 9), xlabel='Time from forecast origin (ps)', ylabel=f'UMAP {dim+1}',
                   title=(TITLES[e['category']]+' · ' if dim == 0 else '')+f'UMAP {dim+1}')
            ax.grid(alpha=.15)
    fig.suptitle('UMAP coordinates through time: truth, branches, and 24 sampled futures\n'
                 'Black vertical line: forecast origin. Green dotted line: first sustained physical crystal onset.', fontsize=14)
    fig.legend(handles=path_legend(), loc='outside lower center', ncol=4, fontsize=9)
    save(fig, output, 'embedding-coordinate-evolution')


def state_band(ax, time, state, origin, onset):
    """Show actual instantaneous PTM labels, separate from predicted scores."""
    step = time[1]-time[0]
    ax.imshow(state[None], aspect='auto', interpolation='nearest',
              cmap=ListedColormap(['#dce5ed', FUTURE]), vmin=0, vmax=1,
              extent=(time[0]-step/2, time[-1]+step/2, 0, 1))
    ax.axvline(origin, color='#db7c13', lw=1.2)
    if onset is not None and time[0] <= onset <= time[-1]:
        ax.axvline(onset, color='black', ls=':', lw=1.2)
    ax.set(yticks=[], xlim=(time[0]-.5*step, time[-1]+.5*step))
    ax.tick_params(axis='x', labelbottom=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def channels(output, examples, arrays):
    with np.load(output/'technical/channel-layout.npz') as layout:
        weight, chosen, order = layout['weight'], layout['top_channels'], layout['order']
    contributions = [channel_contributions(a['true_z'], weight)[:, chosen] for a in arrays]
    bound = max(np.max(abs(c)) for c in contributions)
    fig = plt.figure(figsize=(17, 11), layout='constrained')
    grid = fig.add_gridspec(3, 4, height_ratios=[.25, 3.4, 1.5])
    heat_axes = []
    for col, (e, a, values) in enumerate(zip(examples, arrays, contributions)):
        time = a['time_ps']
        band = fig.add_subplot(grid[0, col])
        state_band(band, time, a['physical_crystal'], 0, e['onset_delay_ps'])
        band.set_title(TITLES[e['category']], fontsize=12, pad=12)
        ax = fig.add_subplot(grid[1, col]); heat_axes.append(ax)
        im = ax.imshow(values.T, aspect='auto', origin='upper', cmap='RdBu_r',
                       vmin=-bound, vmax=bound, extent=(-12.375, 9.375, len(chosen)-.5, -.5), interpolation='nearest')
        ax.set(yticks=range(len(chosen)), yticklabels=[f'Ch {int(j)}' for j in chosen],
               xticks=[-12, -6, 0, 3, 6, 9], xlabel='Time from forecast origin (ps)')
        ax.tick_params(axis='y', labelsize=8, length=0)
        ax.axvline(0, c='#db7c13', lw=1.5)
        if e['onset_delay_ps'] is not None and e['onset_delay_ps'] <= 9:
            ax.axvline(e['onset_delay_ps'], c='black', ls=':', lw=1)
        if col == 0:
            ax.set_ylabel('20 features ranked by score contribution variability on training data')
        score = fig.add_subplot(grid[2, col])
        score.plot(time, a['observed_margin'], c=FUTURE, lw=1.8, label='Actual embedding')
        score.plot(time[17:], a[PRIMARY+'_margin'], c=MEAN, ls='--', lw=1.7, label='Predicted mean embedding')
        score.axhline(0, color='#8893a1', lw=.8)
        score.axvline(0, color='#db7c13', lw=1)
        score.set(xlim=(-12, 9), xlabel='Time from forecast origin (ps)', xticks=[-12, -6, 0, 3, 6, 9])
        score.grid(alpha=.13)
        if col == 0:
            score.set_ylabel('Total crystal score\n(all 256 features)')
            score.legend(fontsize=8, loc='lower left')
    fig.colorbar(im, ax=heat_axes, shrink=.8, label='Change in crystal score contributed by one feature\n'
                 'blue: away from crystal · red: toward crystal')
    fig.suptitle('Which embedding features change around crystallization?\n'
                 'Each cell = feature weight × change from its mean observed history; same 20 training-ranked features in every case', fontsize=14)
    fig.supxlabel('Top strip: measured PTM state (green = crystal). Orange: prediction origin. Dotted black: first sustained onset.\n'
                  'Positive and negative contributions can cancel. Bottom curves use ALL 256 features; the heatmap shows only 20.', fontsize=10)
    save(fig, output, 'embedding-channel-evolution')

    a = arrays[0]; baseline = a['true_z'][:17].mean(0)
    values = [a['true_z'][17:]-baseline, a[PRIMARY+'_mean_z']-baseline,
              *(v-baseline for v in a['component_mean_z'])]
    bound = max(np.max(abs(v)) for v in values)
    labels = ['Actual future', 'Predicted mean embedding']+[
        f'Component {i+1} · weight {p:.1%}' for i, p in enumerate(a['component_weight'])]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout='constrained')
    for ax, value, label in zip(axes.flat, values, labels):
        im = ax.imshow(value[:, order].T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-bound, vmax=bound,
                       extent=(.375, 9.375, -.5, 255.5), interpolation='nearest')
        ax.set(title=label, xlabel='Future time (ps)', ylabel='Feature row (training-correlation order)')
    fig.colorbar(im, ax=axes, label='Feature change from mean observed history (training standard deviations)', shrink=.8)
    fig.suptitle('All 256 features: which predicted branch follows the measured changes?\n'
                 'Related features are grouped by training correlation; one row order and color scale for every panel', fontsize=14)
    save(fig, output, 'future-channel-branches')


def full_paths(output, examples, arrays, projection):
    fig = plt.figure(figsize=(17, 12), layout='constrained')
    grid = fig.add_gridspec(4, 3, width_ratios=[1, 1.15, 1.15])
    all_maps = np.concatenate([a['full_true_map'] for a in arrays])
    lower, upper = all_maps.min(0), all_maps.max(0)
    padding = (upper-lower)*.06
    for row, (e, a) in enumerate(zip(examples, arrays)):
        ax = fig.add_subplot(grid[row, 0])
        coordinates, time, crystal = a['full_true_map'], a['full_time_ps'], a['full_physical_crystal']
        ax.scatter(*coordinates[~crystal].T, color='#90a2b5', s=5, alpha=.25, linewidths=0)
        ax.scatter(*coordinates[crystal].T, color=FUTURE, s=7, alpha=.4, linewidths=0)
        ax.scatter(*coordinates[e['anchor']], color='#db7c13', s=100, marker='*', edgecolors='white', linewidths=.6, zorder=5)
        onset_index = e['onset_frame']
        if onset_index < len(time):
            ax.scatter(*coordinates[onset_index], color='black', s=35, marker='D', zorder=6)
        ax.set(title=TITLES[e['category']]+f" · atom {e['atom_id']}", xlabel='UMAP 1', ylabel='UMAP 2',
               xlim=(lower[0]-padding[0], upper[0]+padding[0]), ylim=(lower[1]-padding[1], upper[1]+padding[1]))
        ax.grid(alpha=.1)
        block_time, block_coordinates = time_blocks(time, coordinates)
        onset = time[onset_index] if onset_index < len(time) else None
        for dim in range(2):
            sub = grid[row, dim+1].subgridspec(2, 1, height_ratios=[.16, 1], hspace=.02)
            band = fig.add_subplot(sub[0])
            state_band(band, time, crystal, e['anchor_time_ps'], onset)
            band.set_title(f'UMAP {dim+1} through time', fontsize=11)
            timeline = fig.add_subplot(sub[1])
            timeline.scatter(time, coordinates[:, dim], s=2, c='#9ca8b6', alpha=.22, linewidths=0)
            timeline.plot(block_time, block_coordinates[:, dim], c=PAST, lw=1.1)
            timeline.axvspan(e['anchor_time_ps']-12, e['anchor_time_ps']+9, color='#f4a340', alpha=.18)
            if onset is not None:
                timeline.axvline(onset, c='black', ls=':', lw=1)
            timeline.set(xlim=(0, 600), xticks=[0, 150, 300, 450, 600], xlabel='Simulation time (ps)', ylabel=f'UMAP {dim+1}',
                         ylim=(lower[dim]-padding[dim], upper[dim]+padding[dim]))
            timeline.grid(alpha=.1)
    handles = [Line2D([], [], marker='o', ls='', color='#90a2b5', label='Measured noncrystal'),
               Line2D([], [], marker='o', ls='', color=FUTURE, label='Measured crystal'),
               Line2D([], [], marker='*', ls='', color='#db7c13', ms=10, label='Forecast origin'),
               Line2D([], [], marker='D', ls='', color='black', label='First sustained onset'),
               Line2D([], [], color=PAST, label='6 ps median of map coordinates'),
               Patch(color='#f4a340', alpha=.3, label='Detailed −12 to +9 ps window')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle('Where does the embedding go, and when does it change?\n'
                 'Left: all 801 frames without connecting lines. Right: time traces with physical-state strips; gray dots retain individual frames.', fontsize=14)
    save(fig, output, 'full-embedding-trajectories')


def readout(output, examples, arrays):
    fig = plt.figure(figsize=(17, 10), layout='constrained')
    grid = fig.add_gridspec(3, 4, height_ratios=[.23, 1, 1])
    for col, (e, a) in enumerate(zip(examples, arrays)):
        time, future = a['time_ps'], a['time_ps'][17:]
        band = fig.add_subplot(grid[0, col]); state_band(band, time, a['physical_crystal'], 0, e['onset_delay_ps'])
        band.set_title(TITLES[e['category']], fontsize=12, pad=15)
        ax = fig.add_subplot(grid[1, col])
        ax.plot(time, a['observed_margin'], c=FUTURE, lw=2, label='Actual embedding')
        for name, (label, color) in METHODS.items():
            ax.plot(future, a[name+'_margin'], c=color, ls='--', lw=1.4, label=label)
        ax.axhline(0, c='black', lw=.8)
        ax.axvline(0, c='#db7c13', lw=1)
        ax.set(xlim=(-12, 9), xlabel='Time from forecast origin (ps)', xticks=[-12, -6, 0, 3, 6, 9])
        ax.grid(alpha=.13)
        if col == 0:
            ax.set_ylabel('Crystal-classifier score\npositive favors crystal')
        prob = fig.add_subplot(grid[2, col])
        prob.plot(future, a['crystal_probability'], color=COLORS[0], lw=2, marker='.', ms=4)
        prob.axhline(e['threshold'], c='black', ls='--', lw=1)
        prob.text(.02, e['threshold']+.035, 'Validation warning threshold', transform=prob.get_yaxis_transform(), fontsize=8)
        if e['onset_delay_ps'] is not None and e['onset_delay_ps'] <= 9:
            prob.axvline(e['onset_delay_ps'], c=FUTURE, ls=':', lw=1)
        prob.set(ylim=(0, 1), xlim=(0, 9), xticks=[0, 3, 6, 9], yticks=[0, .25, .5, .75, 1], xlabel='Future time (ps)')
        prob.grid(alpha=.13)
        if col == 0:
            prob.set_ylabel('Probability that a sampled\nfuture embedding scores positive')
        if col == 3:
            fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle('Three different quantities: measured state, classifier score, and model probability\n'
                 'Top: actual PTM state. Middle: crystal score minus noncrystal score. Bottom: model P(score > 0) at each future frame.\n'
                 'A positive learned score supports crystal; its model probability is not calibrated sustained-crystallization risk.', fontsize=13)
    save(fig, output, 'crystal-readout-paths')


def interactive(output, e, a, projection):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'xy'}]],
                        subplot_titles=['Measured 80-atom local structure (Å)', 'Shared UMAP: observed and predicted paths'])
    def cloud_trace(frame):
        return go.Scatter3d(x=a['cloud_A'][frame, :, 0], y=a['cloud_A'][frame, :, 1],
            z=a['cloud_A'][frame, :, 2], mode='markers', name='Measured atoms',
            marker=dict(size=[10]+[4]*79, color=cloud_colors(a, frame), opacity=.85),
            customdata=a['neighbor_atom_ids'][frame, :, None],
            hovertemplate='Atom %{customdata[0]}<br>x %{x:.2f} Å<br>y %{y:.2f} Å<br>z %{z:.2f} Å<extra></extra>')
    def path_trace(pc, name, color, width=4, **kwargs):
        return go.Scatter(x=pc[:, 0], y=pc[:, 1], mode='lines+markers',
                            line=dict(color=color, width=width), marker=dict(color=color, size=2), name=name, **kwargs)
    fig.add_trace(cloud_trace(16), row=1, col=1)
    fig.add_trace(path_trace(a['true_map'][:17], 'Observed history', PAST), row=1, col=2)
    fig.add_trace(path_trace(a['true_map'][16:], 'Actual future', FUTURE, 6), row=1, col=2)
    fig.add_trace(path_trace(joined(a, a[PRIMARY+'_mean_map']), 'Mixture mean', MEAN), row=1, col=2)
    for i, (pc, p) in enumerate(zip(a['component_map'], a['component_weight'])):
        fig.add_trace(path_trace(joined(a, pc), f'Component {i+1} · {p:.1%}', COLORS[i]), row=1, col=2)
    for i, pc in enumerate(a['sample_map']):
        fig.add_trace(path_trace(joined(a, pc), '24 sampled future paths', '#aab1ba', 1,
                                opacity=.25, showlegend=i == 0, legendgroup='samples', hoverinfo='skip',
                                visible='legendonly'), row=1, col=2)
    cursor = len(fig.data)
    def current_trace(frame):
        pc = a['true_map'][frame]
        return go.Scatter(x=[pc[0]], y=[pc[1]], mode='markers', name='Selected time',
            marker=dict(size=8, color='#e43d30', symbol='diamond'), showlegend=False)
    fig.add_trace(current_trace(16), row=1, col=2)
    frames = []
    for i, t in enumerate(a['time_ps']):
        frames.append(go.Frame(name=str(i), data=[cloud_trace(i), current_trace(i)], traces=[0, cursor],
            layout=go.Layout(title=dict(text=title(e)+f"<br><sup>{t:+g} ps from origin · "
                f"simulation {a['absolute_time_ps'][i]:.2f} ps · center PTM {PTM[int(a['ptm_labels'][i])]}</sup>"))))
    fig.frames = frames
    limit = float(np.max(abs(a['cloud_A']))*1.06)
    scene = dict(aspectmode='cube', xaxis=dict(range=[-limit, limit], title='x (Å)'),
                 yaxis=dict(range=[-limit, limit], title='y (Å)'), zaxis=dict(range=[-limit, limit], title='z (Å)'))
    fig.update_layout(height=770, template='plotly_white', title=title(e)+'<br><sup>0 ps = forecast origin</sup>',
        scene=scene, xaxis=dict(title='UMAP 1'), yaxis=dict(title='UMAP 2'),
        uirevision='keep-camera', margin=dict(l=5, r=5, b=100, t=100), legend=dict(orientation='h', y=-.2),
        updatemenus=[dict(type='buttons', x=0, y=-.04, direction='left', buttons=[
            dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=450, redraw=True),
                 transition=dict(duration=0), fromcurrent=True)]),
            dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=16, x=.17, len=.8, y=-.03, currentvalue=dict(prefix='Relative time: ', suffix=' ps'),
            steps=[dict(label=f'{t:g}', method='animate', args=[[str(i)], dict(mode='immediate',
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for i, t in enumerate(a['time_ps'])])])
    notes = ('<p style="font:15px sans-serif;margin:20px">Drag the point cloud to rotate; pan or zoom the UMAP view. '
        'Use the slider to move through measured frames. Click legend entries to hide paths. '
        'Blue neighbors occur in the origin neighborhood; amber neighbors do not. The green center is PTM-crystalline. '
        'All point clouds are measured simulation data. The model predicts embeddings, with no coordinate decoder. '
        'Each gray sample selects one mixture component for the entire future; its conditional Gaussian residuals '
        'are independent across times/channels. Component curves are conditional means, not decoded physical branches. '
        'UMAP is fitted to training embeddings only. Map distances and axis values are not physical quantities. '
        'These four test cases were chosen for diagnostic outcomes, not as an accuracy estimate.</p>')
    page = fig.to_html(include_plotlyjs=True, full_html=True, auto_play=False)
    (output/'plots'/f"{e['category']}-explorer.html").write_text(page.replace('</body>', notes+'</body>'))


def animation(output, e, a, projection):
    fig = plt.figure(figsize=(11, 5.5), layout='constrained')
    left = fig.add_subplot(121, projection='3d'); right = fig.add_subplot(122)
    images = []; limit = np.max(abs(a['cloud_A']))*1.05
    for i, t in enumerate(a['time_ps']):
        left.clear(); right.clear(); draw_cloud(left, a, i, limit)
        left.set_title(f'Measured point cloud · {PTM[int(a["ptm_labels"][i])]}')
        for pc in a['sample_map'][:12]:
            right.plot(*joined(a, pc)[:, :2].T, c='#aab1ba', alpha=.2, lw=.7)
        for k, pc in enumerate(a['component_map']):
            right.plot(*joined(a, pc)[:, :2].T, c=COLORS[k], lw=1.4)
        right.plot(*a['true_map'][:17, :2].T, c=PAST, lw=2, label='Observed history')
        right.plot(*a['true_map'][16:, :2].T, c=FUTURE, lw=2.5, label='Actual future')
        right.scatter(*a['true_map'][i, :2], c='#df3225', s=80, zorder=10)
        right.set(xlabel='UMAP 1', ylabel='UMAP 2',
                  title='Observed embedding + alternative future paths')
        right.legend(fontsize=8); right.grid(alpha=.15)
        fig.suptitle(title(e)+f'\n{t:+g} ps from prediction origin', fontsize=13)
        fig.supxlabel('Measured coordinates and embeddings · colored branches = mixture component means · gray paths = samples', fontsize=9)
        buf = BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))
    images[0].save(output/'plots/structure-and-embedding.gif', save_all=True, append_images=images[1:],
                   duration=450, loop=0)
    plt.close(fig)


def render(output, examples, projection):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    arrays = []
    for e in examples:
        with np.load(output/'technical'/f"{e['category']}.npz") as data:
            arrays.append({key: data[key] for key in data.files})
    montage(output, examples, arrays)
    embedding_futures(output, examples, arrays, projection)
    component_time(output, examples, arrays, projection)
    channels(output, examples, arrays)
    full_paths(output, examples, arrays, projection)
    readout(output, examples, arrays)
    for e, a in zip(examples, arrays):
        interactive(output, e, a, projection)
        print(f"Rendered interactive explorer: {e['category']}", flush=True)
    animation(output, examples[0], arrays[0], projection)
    descriptions = {
        'point-cloud-evolution': 'Measured point clouds: the same center atom through time, with instantaneous neighbors.',
        'embedding-futures': 'Observed paths, actual futures and four conditional mean paths; samples are optional in the explorer.',
        'embedding-coordinate-evolution': 'UMAP coordinates through time, showing observed and predicted paths.',
        'embedding-channel-evolution': 'Which embedding features change the crystal-classifier score around each transition?',
        'future-channel-branches': 'Actual and predicted feature changes, with correlated channels grouped using training data.',
        'full-embedding-trajectories': '600 ps UMAP overview: occupancy in embedding space, time evolution, and measured crystal state.',
        'crystal-readout-paths': 'Crystal-classifier score, model probability of a positive score, and measured physical state.'}
    descriptions = {key: descriptions[key] for key in ('full-embedding-trajectories', 'embedding-channel-evolution',
        'crystal-readout-paths', 'embedding-futures', 'embedding-coordinate-evolution', 'future-channel-branches', 'point-cloud-evolution')}
    links = ' · '.join(f'<a href="plots/{e["category"]}-explorer.html">{TITLES[e["category"]]}</a>' for e in examples)
    intro = ('Four distinct test-source examples, selected by the existing validation threshold: a forecast within '
        '1.5 ps, one more than 3 ps early, a missed transition, and a false alarm within 9 ps. Positive examples '
        'place onset 3–6 ps ahead so that post-onset structure is visible. The source-order rule selects four '
        '400 K sources in this gallery. This is diagnostic selection, not '
        'a representative accuracy sample. Spatial mixture checkpoint: seed 20260913. '
        f"A single 2D UMAP is fitted to {projection['training_samples']:,} training embeddings; every path uses that same map. "
        'Physical coordinates are measured simulation data; predictions are embeddings. '
        'Mixture components are alternative conditional means, not established physical pathways. '
        'Each sampled path keeps one component for all 12 future frames; conditional residuals are independent.')
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Structures and embedding paths</title>',
        '<style>body{max-width:1350px;margin:35px auto;padding:0 25px;font:17px/1.6 system-ui;color:#243348;background:#f8fafc}'
        'img{width:100%;background:white;border:1px solid #e2e8f0;border-radius:10px}a{color:#2563a6}'
        'section{margin:35px 0}nav{padding:20px;background:#e8eff9;border-radius:10px}</style></head><body>',
        '<h1>Local structures and alternative embedding futures</h1>', f'<p>{intro}</p>',
        f'<nav><b>Interactive explorers — rotate the cloud, inspect UMAP, play and scrub time</b><br>{links}</nav>']
    explanation = ('Crystal-classifier score (formerly “crystal margin”) = the frozen ridge classifier’s '
        'crystal score minus its noncrystal score. Positive favors crystal; negative favors noncrystal. '
        'It is a learned diagnostic, not a physical order parameter. '
        'Model P(positive score) (formerly “readout probability”) = the probability that a future embedding '
        'sampled from the fitted trajectory mixture receives a positive classifier score. It is evaluated '
        'at each future frame and is not a calibrated probability of sustained physical crystallization. '
        'Green state strips are the independent PTM measurements. Heatmaps show signed feature contributions '
        'to score change from mean observed history; correlated features are not independent physical causes.')
    html.append(f'<section><h2>What do the score and probability mean?</h2><p>{explanation}</p></section>')
    md = ['# Local structures and alternative embedding futures', intro,
          '[Open the gallery](index.html). The explorers work offline; download/open their HTML in a browser.',
          explanation,
          '\n'.join(f'- [{TITLES[e["category"]]}](plots/{e["category"]}-explorer.html)' for e in examples),
          '![Synchronized measured structure and embedding animation](plots/structure-and-embedding.gif)']
    for name, caption in descriptions.items():
        encoded = base64.b64encode((output/'plots'/f'{name}.png').read_bytes()).decode()
        html.append(f'<section><h2>{caption}</h2><a href="plots/{name}.pdf">PDF</a> · '
                    f'<a href="plots/{name}.png">PNG</a><img src="data:image/png;base64,{encoded}" alt="{caption}"></section>')
        md.append(f'{caption}\n\n![{caption}](plots/{name}.png)\n\n[PDF](plots/{name}.pdf)')
    html.append('<p>Exact plotted arrays, selected center/origin identities, UMAP training samples, checkpoint and input hashes '
                'are retained in technical/. Definitions and coordinate CSV are in tables/.</p></body></html>')
    (output/'index.html').write_text('\n'.join(html))
    (output/'README.md').write_text('\n\n'.join(md)+'\n')
