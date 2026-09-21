"""Paper-style raster rendering; explanations live beside, never inside, figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from src.project_runtime.paths import resolve_path

COLORS = dict(direct='#2378a3', ar_mse='#d48728', mixture='#8c62a8', diffusion='#269c88')
NAMES = dict(direct='Direct', ar_mse='Autoregressive', mixture='Mixture', diffusion='Diffusion')
INK = '#252932'
T = np.arange(1, 33)*3
HISTORY = np.arange(-16, 33)*3


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'axes.titlesize': 10, 'axes.titleweight': 'normal', 'axes.linewidth': .65,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#82858b',
        'xtick.color': '#51555d', 'ytick.color': '#51555d', 'text.color': INK,
        'axes.labelcolor': INK, 'legend.frameon': False, 'legend.fontsize': 9,
        'lines.linewidth': 1.65, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': False})


def panel(ax, letter, title=''):
    ax.set_title(title, loc='left', pad=10)
    ax.text(-.12, 1.05, letter, transform=ax.transAxes, weight='bold', fontsize=11)


def save(fig, root, name, dpi):
    fig.savefig(root/'plots'/f'{name}.png', dpi=dpi, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def handles(methods, truth=False):
    values = [Line2D([], [], color=COLORS[n], label=NAMES[n]) for n in methods]
    if truth:
        values.insert(0, Line2D([], [], color=INK, label='Observed'))
    return values


def trajectories(root, cases, observations, predictions, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex=True)
    for i, ax in enumerate(axes.flat):
        y = observations[f'state_{i}'][:, 257]
        ax.axvspan(-48, 0, color='#f1f3f6', zorder=0)
        ax.axvline(0, color='#a9afb8', lw=.8, ls=':')
        ax.plot(HISTORY, y, color=INK, lw=1.5, zorder=4)
        for name in ('direct', 'ar_mse'):
            ax.plot(T, predictions[name]['paths'][i, 0, :, 257], color=COLORS[name])
        if cases[i]['onset_ps'] is not None:
            ax.axvline(cases[i]['onset_ps'], color=INK, lw=.8, ls='--', alpha=.65)
        panel(ax, chr(97+i), cases[i]['label'])
        ax.set(xlim=(-48, 96), xticks=[-48, 0, 24, 48, 72, 96], ylabel=r'$q_6$')
        if i >= 2:
            ax.set_xlabel('Time from forecast origin (ps)')
    fig.legend(handles=handles(('direct', 'ar_mse'), True), loc='upper center', ncol=3, bbox_to_anchor=(.52, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.3)
    save(fig, root, '01_structural_trajectories', dpi)


def onset(root, cases, predictions, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.7), sharex=True, sharey=True)
    x = np.arange(129)*.75
    for i, ax in enumerate(axes.flat):
        for name in NAMES:
            ax.plot(x, np.r_[0, predictions[name]['cdf'][i]], color=COLORS[name])
        if cases[i]['onset_ps'] is not None:
            ax.axvline(cases[i]['onset_ps'], color=INK, ls='--', lw=.9)
        ax.axvline(12, color='#bdc3cc', ls=':', lw=.8)
        panel(ax, chr(97+i), cases[i]['label'])
        ax.set(xlim=(0, 96), ylim=(-.025, 1.025), xticks=[0, 12, 24, 48, 72, 96], yticks=[0, .5, 1])
        if i % 2 == 0:
            ax.set_ylabel('Onset probability')
        if i >= 2:
            ax.set_xlabel('Forecast horizon (ps)')
    fig.legend(handles=handles(NAMES), loc='upper center', ncol=4, bbox_to_anchor=(.52, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.3)
    save(fig, root, '02_onset_probabilities', dpi)


def uncertainty(root, cases, observations, predictions, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.9), sharex=True)
    for row, name in enumerate(('mixture', 'diffusion')):
        for col, index in enumerate((1, 2)):
            ax = axes[row, col]; paths = predictions[name]['paths'][index, :, :, 257]
            low, high = np.quantile(paths, [.05, .95], axis=0)
            ax.fill_between(T, low, high, color=COLORS[name], alpha=.16, lw=0)
            for sample in paths[:5]:
                ax.plot(T, sample, color=COLORS[name], lw=.6, alpha=.18)
            ax.plot(T, paths.mean(0), color=COLORS[name], lw=2)
            ax.plot(T, observations[f'state_{index}'][17:, 257], color=INK, lw=1.5)
            if cases[index]['onset_ps'] is not None:
                ax.axvline(cases[index]['onset_ps'], color=INK, ls='--', lw=.8)
            panel(ax, chr(97+row*2+col), f"{NAMES[name]} · {cases[index]['label'].lower()}")
            ax.set(xlim=(0, 96), ylabel=r'$q_6$', xticks=[0, 24, 48, 72, 96])
            if row == 1:
                ax.set_xlabel('Forecast horizon (ps)')
    fig.tight_layout(h_pad=2.1)
    save(fig, root, '03_predictive_spread', dpi)


def aggregate(root, dpi):
    a = np.load(root/'technical/aggregate.npz')
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))
    for ax, key in zip(axes, ('physical', 'brier')):
        for name in NAMES:
            mean, low, high = a[f'{name}_{key}']; x = np.arange(1, len(mean)+1)*(3 if key == 'physical' else .75)
            ax.fill_between(x, low, high, color=COLORS[name], alpha=.10, lw=0)
            ax.plot(x, mean, color=COLORS[name])
        ax.set(xlabel='Forecast horizon (ps)', xlim=(0, 96), xticks=[0, 24, 48, 72, 96])
    mean, low, high = a['direct_persistence']
    axes[0].plot(T, mean, color='#8d939c', ls='--', lw=1.3)
    axes[0].fill_between(T, low, high, color='#8d939c', alpha=.07, lw=0)
    axes[0].set_ylabel('Standardized physical MSE'); axes[1].set_ylabel('Onset Brier score')
    panel(axes[0], 'a'); panel(axes[1], 'b')
    legend = handles(NAMES)+[Line2D([], [], color='#8d939c', ls='--', label='Persistence')]
    fig.legend(handles=legend, ncol=5, loc='upper center', bbox_to_anchor=(.51, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.4)
    save(fig, root, '04_forecast_quality', dpi)


def embedding(root, dpi):
    a = np.load(root/'technical/umap.npz'); xy = a['test_xy']; meta = a['metadata']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.99, top=.88, bottom=.25, wspace=.2)
    for label, color, mask in [('Other / liquid', '#c3c7cd', meta[:, 5] == 0), ('Crystalline', '#2378a3', meta[:, 5] == 1)]:
        axes[0].scatter(*xy[mask].T, s=6, color=color, alpha=.65, linewidths=0, label=label)
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, -.40), ncol=2, markerscale=2, fontsize=8)
    order = np.argsort(meta[:, 4]); q = axes[1].scatter(*xy[order].T, c=meta[order, 4], cmap='viridis', s=6, alpha=.8, linewidths=0)
    pos = axes[1].get_position()
    bar = fig.colorbar(q, cax=fig.add_axes([pos.x0, .115, pos.width, .025]), orientation='horizontal'); bar.set_label(r'$q_6$', labelpad=1)
    cmap = matplotlib.colors.ListedColormap(['#356794', '#568ca9', '#84b2b5', '#d3b171', '#bf753b'])
    temps = np.array([400, 450, 500, 510, 520]); category = np.searchsorted(temps, meta[:, 3])
    p = axes[2].scatter(*xy.T, c=category, cmap=cmap, vmin=-.5, vmax=4.5, s=6, alpha=.75, linewidths=0)
    pos = axes[2].get_position()
    bar = fig.colorbar(p, cax=fig.add_axes([pos.x0, .115, pos.width, .025]), orientation='horizontal', ticks=range(5))
    bar.ax.set_xticklabels(temps); bar.set_label('Temperature (K)', labelpad=1)
    for i, title in enumerate(('Local structure', 'Bond order', 'Condition')):
        ax = axes[i]; panel(ax, chr(97+i), title); ax.set(xticks=[], yticks=[], xlabel='UMAP 1', aspect='equal')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[0].set_ylabel('UMAP 2')
    save(fig, root, '05_embedding_umap', dpi)


def embedding_paths(root, cases, dpi):
    a = np.load(root/'technical/umap.npz'); background = a['test_xy']
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.0), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.scatter(*background.T, s=3, color='#d8dce1', alpha=.32, linewidths=0, zorder=0)
        actual = a[f'actual_{i}']
        ax.plot(*actual[:17].T, color='#8d939c', ls=':', lw=1.4)
        ax.plot(*actual[16:].T, color=INK, lw=1.4, marker='.', markersize=3)
        for name in ('direct', 'ar_mse'):
            path = np.vstack((actual[16], a[f'{name}_{i}']))
            ax.plot(*path.T, color=COLORS[name], lw=1.4)
            ax.scatter(*path[-1], color=COLORS[name], marker='s', s=20, zorder=5)
        ax.scatter(*actual[16], s=45, color='white', edgecolors=INK, linewidths=1.1, zorder=6)
        ax.scatter(*actual[-1], s=25, color=INK, marker='s', zorder=5)
        panel(ax, chr(97+i), cases[i]['label'])
        ax.set(xticks=[], yticks=[], aspect='equal'); ax.spines[['left', 'bottom']].set_visible(False)
        if i >= 2:
            ax.set_xlabel('UMAP 1')
        if i % 2 == 0:
            ax.set_ylabel('UMAP 2')
    fig.legend(handles=handles(('direct', 'ar_mse'), True), ncol=3, loc='upper center', bbox_to_anchor=(.52, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=2)
    save(fig, root, '06_forecast_umap_paths', dpi)


def spatial(root, dpi):
    a = np.load(root/'technical/context-clouds.npz'); method = json.loads((root/'technical/context-method.json').read_text())
    # One rigid camera rotation for all real clouds; no rearrangement of atoms.
    theta, phi = .38, .55
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    rotation = rz@rx; radius = method['local_radius_A']
    colors = ['#202b3b', '#3296b8', '#3cadad', '#5c88be', '#d9953c', '#c8764a', '#caa154']
    fig = plt.figure(figsize=(12.5, 5.0))
    ax = fig.add_axes([.025, .13, .31, .75]); full = a['full']@rotation; reps = a['representatives']@rotation
    order = np.argsort(full[:, 2]); ax.scatter(*full[order, :2].T, s=2.2, color='#bfc6ce', alpha=.23, linewidths=0)
    for r, ls in ((12, ':'), (25, '--')):
        ax.add_patch(Circle((0, 0), r, fill=False, ec='#6d7886', lw=.9, ls=ls))
    for j in range(7):
        x = a[f'local_{j}']@rotation+reps[j]
        ax.scatter(*x[:, :2].T, s=5.5, color=colors[j], alpha=.55, linewidths=0)
        ax.add_patch(Circle(reps[j, :2], radius, fill=False, ec=colors[j], lw=.65, alpha=.55))
        ax.scatter(*reps[j, :2], s=42, color=colors[j], edgecolors='white', lw=.7, zorder=5)
        ax.annotate(str(j), reps[j, :2], xytext=(4, 4), textcoords='offset points', fontsize=9, color=colors[j], weight='bold')
    ax.text(0, 26.5, '25 Å', ha='center', fontsize=9); ax.text(-1, -13.6, '12 Å', ha='center', fontsize=8, color='#687481')
    ax.plot([-30, -20], [-32, -32], color=INK, lw=1.5); ax.text(-25, -35, '10 Å', ha='center', fontsize=8)
    ax.set(xlim=(-36, 36), ylim=(-37, 37), aspect='equal'); ax.axis('off')
    fig.text(.025, .95, 'a', weight='bold', fontsize=11); fig.text(.055, .95, 'Spatial representatives', fontsize=10)
    fig.text(.375, .95, 'b', weight='bold', fontsize=11); fig.text(.405, .95, 'Local encoder inputs', fontsize=10)
    for j in range(7):
        col = j % 4; row = j//4; x0 = .375+col*.085; y0 = .56-row*.31
        patch = fig.add_axes([x0, y0, .082, .23]); x = a[f'local_{j}']@rotation
        order = np.argsort(x[:, 2]); patch.scatter(*x[order, :2].T, s=10, color=colors[j], alpha=.65, lw=.2, edgecolors='white')
        patch.scatter(0, 0, s=24, facecolors='white', edgecolors=INK, linewidths=.8, zorder=5)
        patch.add_patch(Circle((0, 0), radius, fill=False, ec='#bfc6ce', lw=.55))
        patch.set(aspect='equal', xlim=(-radius-1, radius+1), ylim=(-radius-1, radius+1)); patch.axis('off')
        patch.set_title(str(j), fontsize=9, color=colors[j], pad=3)
    fig.text(.54, .15, f'$R_{{local}}$ = {radius:.2f} Å', ha='center', fontsize=9)
    fig.text(.76, .95, 'c', weight='bold', fontsize=11); fig.text(.79, .95, 'Space–time context', fontsize=10)
    diagram = fig.add_axes([.75, .12, .245, .74]); diagram.set(xlim=(-1, 10), ylim=(-1.5, 5)); diagram.axis('off')
    for row, time in enumerate((-48, -12, -3, 0)):
        y = 4-row
        diagram.text(-.6, y, str(time), ha='right', va='center', fontsize=8)
        for j, color in enumerate(colors):
            diagram.scatter(j*.73, y, s=35, color=color, linewidths=0)
        diagram.add_patch(FancyArrowPatch((4.8, y), (6.25, y), arrowstyle='->', mutation_scale=9, lw=.8, color='#7d8793'))
        diagram.scatter(6.65, y, s=48, marker='s', color='#718599')
    diagram.text(-.6, 4.7, 'ps', ha='right', fontsize=8)
    diagram.text(2.2, 4.7, '7 embeddings', ha='center', fontsize=8)
    diagram.text(6.65, 4.7, 'Spatial', ha='center', fontsize=8)
    diagram.add_patch(FancyArrowPatch((7.6, 4.2), (7.6, .3), arrowstyle='->', mutation_scale=10, lw=1, color='#5f6d7f'))
    diagram.text(8.15, 2.3, 'Temporal', rotation=90, ha='center', va='center', fontsize=8)
    diagram.scatter(6.65, -.25, s=75, marker='s', color=COLORS['direct'])
    diagram.add_patch(FancyArrowPatch((6.65, .7), (6.65, -.02), arrowstyle='->', mutation_scale=9, lw=.8, color='#5f6d7f'))
    diagram.text(6.65, -.95, 'Forecast', ha='center', fontsize=9)
    fig.add_artist(FancyArrowPatch((.335, .52), (.37, .52), transform=fig.transFigure, arrowstyle='->', mutation_scale=11, lw=1, color='#8792a0'))
    fig.add_artist(FancyArrowPatch((.714, .46), (.752, .46), transform=fig.transFigure, arrowstyle='->', mutation_scale=11, lw=1, color='#8792a0'))
    save(fig, root, '07_spatial_context', dpi)


def captions(root, cases):
    method = json.loads((root/'technical/context-method.json').read_text())
    record = json.loads((root/'technical/prepared.json').read_text())
    umap = json.loads((root/'technical/umap-method.json').read_text())
    descriptions = {
        '01_structural_trajectories': ('Structural trajectories',
            'Observed local bond order $q_6$ (black) and open-loop direct (blue) and autoregressive (orange) predictions. '
            'Grey shading marks the observed past; zero is the forecast origin. Dashed vertical lines mark the actual first sustained local crystalline onset. '
            'Future structural targets are sampled every 3 ps through 96 ps. The physical/order heads predict these quantities alongside the future embedding; '
            'these are not reconstructed atomic trajectories or quantities decoded from predicted embeddings. No future values are fed back. '
            'Direct/AR models see snapshots at −48, −12, −3 and 0 ps; the dense black past is shown for orientation.'),
        '02_onset_probabilities': ('Crystallization onset probabilities',
            'Archived cumulative onset probabilities for the same four windows. Vertical dashed lines denote observed onset, '
            'and the light dotted line marks 12 ps. “No onset” means no event observed within 96 ps, not permanent stability. '
            '“Missed early onset” is an intentional failure example under the direct predictor’s calibration-only 5% false-positive-rate operating point. '
            'Onset uses the tracked center becoming crystalline for three consecutive original 0.75 ps frames. The CDF is a prediction of event timing, '
            'distinct from the instantaneous crystallinity state head.'),
        '03_predictive_spread': ('Probabilistic trajectory spread',
            'Mixture and diffusion predictions for the later-onset and surviving examples. Colored lines are the sample means; shading shows pointwise '
            '5–95% intervals from 64 fixed-seed samples. Five predetermined sample paths are shown faintly, without choosing the closest to truth. '
            'Black is observed $q_6$. Bands describe model-generated spread, not uncertainty in the mean, not simultaneous coverage, and not a claim of calibration. '
            'These two selected models use −12, −3 and 0 ps input frames. Fresh CPU samples are used for illustrations; the aggregate scores retain the original archived samples.'),
        '04_forecast_quality': ('Held-out forecast quality',
            'All 44,385 at-risk test windows across 30 sources. Left: standardized physical-packet MSE of the predictive mean and a constant-current-state persistence baseline. '
            'Right: onset Brier score at each lead time. Lower is better. Every source has equal total weight. Shading is a 95% paired source-bootstrap interval '
            '(1,000 resamples); it does not represent seed uncertainty. Models are the four family promotions chosen on development data, with a 36-epoch maximum '
            'and early stopping. Their observed context differs as documented; this is not a capacity- and input-matched architecture comparison.'),
        '05_embedding_umap': ('Frozen MACE embedding space',
            f'The same {umap["test_rows"]:,} held-out atomic states in three colorings: instantaneous PTM crystalline status (FCC/HCP/BCC), local $q_6$, and temperature. '
            f'The 128-channel encoder is the original fixed trajectory backbone, SHA-256 `{record["feature_checkpoint_sha256"]}`. '
            f'Per-channel standardization and UMAP are fitted only on {umap["training_rows"]:,} states from 90 training sources. '
            'Sampling is uniform without replacement over each source’s center/time grid, independently of outcomes, spanning 0–594 ps. '
            'UMAP uses 30 neighbors, min_dist 0.15, Euclidean distance and seed 20260921. Held-out points are transformed afterward. '
            'Colors never enter the fit. Apparent clusters, overlaps and distances are projection-dependent and do not establish physical phases or predictive sufficiency. '
            'These are the frozen MACE features used by the trajectory study; they are not the newly training BCR codes.'),
        '06_forecast_umap_paths': ('Observed and predicted paths in embedding space',
            'The same training-fitted UMAP, with held-out states in light grey. Each panel corresponds to the examples in Figures 1–2. '
            'Grey dotted lines show past observed embeddings, black shows future observed embeddings, and blue/orange show the direct/AR future predictions. '
            'The white circle marks the shared forecast origin; squares mark 96 ps endpoints. Each 128D prediction is transformed into the fixed map. '
            'A short or visually close 2D path is not proof of small error: UMAP can compress, distort or fold forecast deviations, including off-manifold predictions. '
            'The numeric quality comparison remains in the original standardized target space.'),
        '07_spatial_context': ('Spatial context from real atomic neighborhoods',
            f'Original periodic Al coordinates: source {method["source"]}, atom {method["tracked_atom_id"]}, time {method["time_ps"]:g} ps. '
            'Panel a is a fixed orthographic projection of a real 3D neighborhood, with minimum-image displacements. Dashed/dotted circle silhouettes mark '
            '25 Å and 12 Å spherical radii; projected overlap does not mean the atoms coincide in 3D. Token 0 is the tracked center. '
            'Within each of (0,12] Å and (12,25] Å, the nearest atom seeds a farthest-point selection of three representatives. '
            'Each representative is independently encoded from its own local point cloud, shown in panel b with the same camera. '
            f'The recorded local cutoff is {method["local_radius_A"]:.3f} Å after converting the normalized cutoff back to physical units; '
            f'the seven actual atom counts are {method["atom_counts"]}. The neighboring encoder crops can extend beyond the 25 Å center-selection radius. '
            'Panel c shows the direct/AR 48 ps input schedule: seven features at each of −48, −12, −3 and 0 ps. Representatives are selected afresh in each frame, '
            'not tracked as fixed neighbor identities. Learned spatial attention uses feature interactions, pairwise geometry and a smooth radial key weight; '
            'weighted pooling produces one summary per frame, then causal temporal attention combines those summaries. The current-center residual and known '
            'conditions also enter the predictor. Extra observed shell/descriptor features enter a separate auxiliary head (omitted from the diagram). '
            'Arrows represent computation, not measured attention or causal influence.')}
    header = '# Trajectory prediction and atomic context — figure gallery\n\n'
    header += 'Seven PNG figures at 300 dpi, with separate captions. No PDF or SVG files are generated. '
    header += 'Completed, development-selected context-night forecasts; existing fixed MACE backbone; one training seed. '
    header += 'The historical test sources have already been inspected. No predictors were refitted.\n\n'
    header += 'Example selection is deliberately transparent: median physical/order/crystallinity error per source, '
    header += 'then the median eligible source in each predefined outcome stratum, excluding already chosen sources. '
    header += 'The fourth example is explicitly a failure case, not an estimate of its frequency.\n\n'
    header += '| Panel | Source | Center slot (0–15) | Origin (ps) | Onset lead (ps) |\n|---|---:|---:|---:|---:|\n'
    for c in cases:
        header += f'| {c["label"]} | {c["source"]} | {c["center"]} | {c["frame"]*.75:g} | {c["onset_ps"] if c["onset_ps"] is not None else ">96"} |\n'
    content = header
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>Trajectory figures</title>',
            '<style>body{font:16px/1.65 system-ui;color:#252932;max-width:1120px;margin:48px auto;padding:0 24px}img{width:100%;height:auto}figure{margin:50px 0 70px}figcaption{max-width:1000px;color:#444}h1{font-weight:600}a{color:#2378a3}</style></head><body>',
            '<h1>Trajectory prediction and atomic context</h1><p>Completed MACE trajectory study · PNG only · 300 dpi · captions below each figure.</p>',
            '<p><a href="README.md">Methods and example selection</a> · <a href="tables/METRICS.md">Metric definitions</a></p>']
    import html as escaping
    for name, (title, caption) in descriptions.items():
        content += f'\n## {title}\n\n![{title}](plots/{name}.png)\n\n{caption}\n'
        html.append(f'<figure><a href="plots/{name}.png"><img src="plots/{name}.png" alt="{title}"></a><figcaption><strong>{title}.</strong> {escaping.escape(caption)}</figcaption></figure>')
    content += '\nReproduce: `python -m src.research.crystallization_paths.figures --config configs/analysis/crystallization_figures.json`. '
    content += 'CPU replay checks, exact example identities, source/checkpoint hashes, UMAP fit and point-cloud arrays are retained in `technical/`.\n'
    (root/'README.md').write_text(content); (root/'index.html').write_text('\n'.join(html+['</body></html>']))


def render(config):
    style(); root = resolve_path(config['output']); (root/'plots').mkdir(exist_ok=True)
    cases = json.loads((root/'technical/examples.json').read_text())
    observations = np.load(root/'technical/observed.npz')
    predictions = {n: dict(np.load(root/'technical'/f'{n}-examples.npz')) for n in NAMES}
    dpi = config['dpi']
    trajectories(root, cases, observations, predictions, dpi)
    onset(root, cases, predictions, dpi)
    uncertainty(root, cases, observations, predictions, dpi)
    aggregate(root, dpi); embedding(root, dpi); embedding_paths(root, cases, dpi); spatial(root, dpi)
    captions(root, cases)
    print('Wrote seven paper-style PNG figures and separate captions:', root, flush=True)
