"""Descriptive one-to-one color assignment from saved dense contingency tables."""
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_table


def summarize(table, mapping):
    """Rows are original neural IDs; columns are original descriptor IDs."""
    table = np.asarray(table, np.int64)
    nr = table.sum(1); nd = table.sum(0); n = int(table.sum())
    matched = int(table[np.arange(7), mapping].sum())
    pairs = {}
    for neural, descriptor in enumerate(mapping):
        shared = int(table[neural, descriptor]); union = int(nr[neural]+nd[descriptor]-shared)
        pairs[str(neural)] = dict(descriptor=int(descriptor), neural_rows=int(nr[neural]),
            descriptor_rows=int(nd[descriptor]), intersection=shared, union=union,
            neural_fraction=shared/int(nr[neural]) if nr[neural] else None,
            descriptor_fraction=shared/int(nd[descriptor]) if nd[descriptor] else None,
            iou=shared/union if union else None)
    return dict(rows=n, matched_rows=matched, matched_fraction=matched/n if n else None, pairs=pairs)


def build(source):
    source = Path(source); path = source/'technical/metrics.json'
    out = source.parent/'cluster-color-matching-v1'
    receipt = out/'technical/provenance.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['source_sha256'] != sha(path) or previous['implementation_sha256'] != sha(__file__):
            raise ValueError('Matching calculation changed; use a new numerical revision')
        return json.loads((out/'technical/matches.json').read_text()), out
    saved = json.loads(path.read_text()); frames = ['166ps', '170ps', '174ps', '175ps', '177ps', '240ps']
    if set(saved) != set(frames): raise ValueError('Expected the six static Al snapshots')
    matches = {}; metrics = {}
    for model in saved[frames[0]]['models']:
        matches[model] = {}; metrics[model] = {}
        for family in ('joint', 'tda', 'bond_order', 'cna'):
            tables = []
            for frame in frames:
                record = saved[frame]['models'][model][family]['all_grid']
                # Original producer stores descriptor rows × neural columns.
                table = np.asarray(record['contingency'], np.int64).T
                if table.shape != (7, 7) or table.sum() != saved[frame]['rows'] or (table < 0).any():
                    raise ValueError(f'Invalid saved contingency: {frame}/{model}/{family}')
                tables.append(table)
            total = np.sum(tables, axis=0)
            if total.sum() != 684723: raise ValueError('Changed dense matching population')
            rows, columns = linear_sum_assignment(total, maximize=True)
            if not np.array_equal(rows, np.arange(7)): raise ValueError('Incomplete cluster assignment')
            matches[model][family] = dict(neural_to_descriptor=columns.tolist(),
                contingency=total.tolist(), reference=summarize(total, columns))
            metrics[model][family] = dict(pooled=summarize(total, columns),
                snapshots={f:summarize(t, columns) for f,t in zip(frames,tables)})
    write_metric_table(metrics, out, family='cluster_color_matching')
    write_json(out/'technical/matches.json', matches)
    write_json(out/'technical/provenance.json', dict(source=str(path), source_sha256=sha(path),
        implementation_sha256=sha(__file__), scipy_version=version('scipy'),
        reference_rows=684723, assignment='maximize pooled shared membership, one-to-one',
        neural_training=False, cluster_refit=False, projection_refit=False))
    return matches, out
