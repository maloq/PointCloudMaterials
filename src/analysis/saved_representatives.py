"""Restyle published native-MACE snapshots using their frozen representative IDs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.result_records import file_hash, location, resolve_reference
from src.project_runtime.paths import resolve_path
from .publication import inventory, refresh_publication_record
from .representative_style import render_representatives


def render_saved(run, analysis_name='standard-v1'):
    run = Path(run).resolve()
    bundle = run/'analyses'/analysis_name
    source = bundle/'data'
    record_path = run/'run.json'
    record = json.loads(record_path.read_text())
    analysis = next(a for a in record['analyses'] if a['source'] == location(source))
    artifacts = {a['relative']: a for a in analysis['artifacts']}
    # Verify the evidence we are preserving, including the original legacy figures.
    frozen_hashes = {name: a['sha256'] for name, a in artifacts.items() if a.get('sha256')}
    for name, checksum in frozen_hashes.items():
        if file_hash(resolve_reference(artifacts[name]['path'])) != checksum:
            raise ValueError(f'Published evidence changed: {name}')
    protocol = json.loads((source/'structural-inference-protocol.json').read_text())
    if protocol['protocol'] != 'native_capacity_mace_static_v1':
        raise ValueError('Saved representative reconstruction requires native_capacity_mace_static_v1')
    receipt_path = bundle/'technical/representative-rendering-v2.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'frames': []}
    owned = {p['relative']: p['sha256'] for f in previous['frames'] for p in f['outputs']}
    receipt = dict(schema_version=1, analysis_id=analysis['id'], mode='saved_representatives_only',
                   producer_sha256={p: file_hash(p) for p in (
                       __file__, str(Path(__file__).with_name('representative_style.py')),
                       str(Path(__file__).with_name('cluster_geometry.py')))},
                   selection='Original per-frame representative IDs; no encoder, clustering or PTM/CNA recomputation',
                   frames=[])
    new_paths = set()
    for frame in protocol['frames']:
        name = Path(frame['file']).stem
        assignments_path = source/f'snapshots/{name}/md_space/local_structure_coords_clusters.npz'
        with np.load(assignments_path) as assigned:
            centers, labels = assigned['coords'], assigned['clusters']
        if len(centers) != frame['centers']:
            raise ValueError(f'Saved frame population differs from protocol: {name}')
        original = resolve_path(frame['file'])
        if file_hash(original) != frame['source_sha256']:
            raise ValueError(f'Source coordinates changed: {original}')
        points = np.load(original).astype(np.float64)
        tree = cKDTree(points)
        summaries = sorted((source/f'snapshots/{name}').glob(
            'figure_set_k*/10_cluster_representatives_structure_analysis_k*.json'))
        if not summaries:
            raise FileNotFoundError(f'No frozen representative summary for {name}')
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            if summary['analysis_points_source'] != 'local_points' or not summary['cna_enabled']:
                raise ValueError(f'Requires recorded local-point CNA shell: {summary_path}')
            k = summary_path.stem.rsplit('_k', 1)[1]
            palette_path = summary_path.with_name(f'cluster_color_assignment_k{k}.json')
            palette = json.loads(palette_path.read_text())['assignment']
            render_records, identities = [], []
            for rep in summary['representatives']:
                cid, row = rep['cluster_id'], rep['sample_index']
                if labels[row] != cid or rep['center_atom_index'] != 0:
                    raise ValueError(f'Representative identity mismatch: {name}, C{cid+1}, row {row}')
                distances, ids = tree.query(centers[row], k=64)
                if distances[0] != 0:
                    raise ValueError(f'Representative is not an exact source atom: {name}, row {row}')
                local = points[ids] - centers[row]
                physical_shell = np.linalg.norm(local, axis=1)[rep['cna']['shell_indices']]
                saved_shell = np.asarray(rep['cna']['shell_distances'])
                scale = float(np.median(physical_shell/saved_shell))
                if not np.allclose(physical_shell, scale*saved_shell, rtol=2e-5, atol=1e-5):
                    raise ValueError(f'Source does not reproduce the saved shell: {name}, row {row}')
                render_records.append(dict(cluster_id=cid, sample_index=row, base_color=palette[str(cid)],
                    points=local, cutoff=rep['cna']['cutoff']*scale, units='Å',
                    cutoff_source='Frozen focal CNA shell cutoff, converted to source Å'))
                identities.append(dict(cluster_id=cid, frame_row=row, source_atom_rows=ids.tolist(),
                                       saved_coordinate_scale_to_A=scale))
            output = summary_path.with_name(f'04_cluster_representatives_k{k}.png')
            outputs = [output, output.with_suffix('.html')]
            for path in outputs:
                relative = str(path.relative_to(source))
                if path.exists() and (relative not in owned or file_hash(path) != owned[relative]):
                    raise FileExistsError(f'Refusing to overwrite an unowned/modified rendering: {path}')
            result = render_representatives(render_records, output, title=f'Cluster representatives · {name}')
            receipt['frames'].append(dict(frame=name, selected_k=int(k),
                source=frame['file'], source_sha256=frame['source_sha256'],
                assignments_sha256=file_hash(assignments_path),
                representative_summary=str(summary_path.relative_to(source)),
                representative_summary_sha256=file_hash(summary_path), palette_sha256=file_hash(palette_path),
                identities=identities, rendering=result,
                outputs=[dict(relative=str(p.relative_to(source)), sha256=file_hash(p)) for p in outputs]))
            new_paths.update(str(p.relative_to(source)) for p in outputs)
            write_json(receipt_path, receipt)
            print(f'{name}: {len(render_records)} original representatives → PNG + offline HTML', flush=True)
    # Only explicitly owned new renderings enter the existing analysis. Never refresh
    # historical hashes or replace scientific metrics with current implementations.
    for name, checksum in frozen_hashes.items():
        if name not in new_paths and file_hash(resolve_reference(artifacts[name]['path'])) != checksum:
            raise ValueError(f'Original evidence changed while rendering: {name}')
    additions = {a['relative']: a for a in inventory(source, bundle) if a['relative'] in new_paths}
    if set(additions) != new_paths:
        raise ValueError('Incomplete representative figure inventory')
    record = json.loads(record_path.read_text())
    analysis = next(a for a in record['analyses'] if a['id'] == receipt['analysis_id'])
    analysis['artifacts'] = [a for a in analysis['artifacts'] if a['relative'] not in additions]
    analysis['artifacts'].extend(additions.values())
    write_json(record_path, record)
    refresh_publication_record(record_path)
    return dict(run=str(run), frames=len(receipt['frames']), new_figures=len(new_paths),
                analysis_id=analysis['id'], receipt=str(receipt_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--analysis', default='standard-v1')
    args = parser.parse_args()
    print(json.dumps(render_saved(args.run, args.analysis), indent=2))


if __name__ == '__main__':
    main()
