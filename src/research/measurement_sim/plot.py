"""Show actual measured MD paths and coherent individual neural trajectories."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot(out):
    data = torch.load(out/'technical/example-trajectories.pt', map_location='cpu', weights_only=False)
    columns = data['columns']
    features = [('bond_order/l6_q', 'Bond order q6'), ('velocity/shell0/radial_mean', 'Radial velocity (A/ps)'),
                ('density/radial1', 'Radial density')]
    folder = out/'plots'; folder.mkdir(exist_ok=True)
    for regime in ['original', 'liquid_liquid', 'growing_front']:
        cases = [case for case in data['cases'] if case['row']['regime'] == regime]
        fig, axes = plt.subplots(len(cases), len(features), figsize=(13, 3*len(cases)), squeeze=False)
        for row, case in enumerate(cases):
            for col, (name, title) in enumerate(features):
                i = columns.index(name); ax = axes[row, col]
                actual = case['truth'][3:, i].numpy(); future = case['particles'][:, :, i].numpy()
                paths = np.concatenate((np.full((len(future), 1), actual[0]), future), 1)
                time = np.arange(61)*.1
                ax.fill_between(time, *np.quantile(paths, [.05, .95], axis=0), color='#2584be', alpha=.13)
                ax.plot(time, paths[:4].T, color='#2584be', alpha=.28, linewidth=.7)
                ax.plot(time, paths.mean(0), color='#2584be', label='Neural mean')
                ax.plot(time, actual, color='black', linewidth=1.2, label='Saved MD')
                ax.set_title(title+' — atom '+str(case['row']['center'])); ax.set_xlabel('Elapsed time (ps)')
                if row == col == 0: ax.legend()
        fig.suptitle(regime+' | 32 neural paths; no independent decoder noise')
        fig.tight_layout(); fig.savefig(folder/(regime+'-trajectories.png'), dpi=160); plt.close(fig)
