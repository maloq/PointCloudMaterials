"""Publication-only rendering of frozen descriptor-audit tables."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    with (root / 'tables/blocks.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['block'] != 'bond_order']
    models = list(dict.fromkeys(r['model'] for r in rows))
    blocks = list(dict.fromkeys(r['block'] for r in rows))
    lookup = {(r['model'], r['block']): 100 * float(r['r2']) for r in rows}
    data = np.array([[lookup[model, block] for model in models] for block in blocks])
    fig, ax = plt.subplots(figsize=(9, 8.5), layout='constrained')
    picture = ax.imshow(data, vmin=0, vmax=100, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(models)), models, fontsize=10)
    ax.set_yticks(range(len(blocks)), [b.replace('_', ' · ') for b in blocks], fontsize=10)
    ax.tick_params(axis='both', length=0, pad=8)
    for row in range(len(blocks)):
        for col in range(len(models)):
            ax.text(col, row, f'{data[row,col]:.1f}', ha='center', va='center', fontsize=11,
                    color='black' if data[row,col] > 65 else 'white')
    for edge in (2.5, 5.5, 8.5, 11.5, 14.5):
        ax.axhline(edge, color='white', linewidth=1)
    ax.set_title('TDA prediction after 12 epochs\n4,096 Al validation patches from 15 sources', fontsize=13, pad=14)
    bar = fig.colorbar(picture, ax=ax, shrink=.65, pad=.035)
    bar.set_label('R² × 100 — higher is better')
    fig.text(.04, -.012, 'n32 / n80: neighborhood size    H0: components    H1: loops    H2: cavities', fontsize=10)
    for extension in ('png', 'pdf'):
        fig.savefig(root / 'plots' / f'tda-difficulty.{extension}', dpi=170, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('root', type=Path)
    render(parser.parse_args().root)
