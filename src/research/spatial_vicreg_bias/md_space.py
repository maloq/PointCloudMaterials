"""Publish independent dense MD panels beside unchanged saved PaCMAP layouts."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def publish(config, dense_config):
    c = json.loads(Path(config).read_text()); out=Path(c['output']); dest=Path(c['publication'])
    dc = json.loads(Path(dense_config).read_text()); dense=Path(dc['output'])
    corr=json.loads(Path(c['correspondence_config']).read_text()); parent=json.loads(Path(corr['parent']).read_text())
    if not out.is_absolute() or not dest.is_absolute() or not dense.is_absolute():
        raise ValueError('Use resolved submission configurations')
    manifest_path=dense/'technical/manifest.json'
    snapshots=json.loads(manifest_path.read_text())['snapshots'] if manifest_path.exists() else []
    models=[]; neural_fields={}
    for item in dc['models']:
        run, epoch=item['run'],item['epoch']; records={}
        for snap in snapshots:
            key=f'{snap["key"]}-{run}-epoch{epoch}'; receipt=dense/'technical'/f'{key}.json'
            if receipt.exists():
                record=json.loads(receipt.read_text()); asset=dest/'md-data'/Path(record['asset']).name
                if sha(asset)!=record['asset_sha256']:raise ValueError(f'Changed neural asset: {asset}')
                records[snap['key']]={'asset':record['asset'],'key':key}
        for rep in ('encoder','projector'):
            title=f'{run} epoch {epoch} {rep} K=7'; identity=f'{run}-epoch{epoch}-{rep}'
            models.append(dict(id=identity,title=title,representation=rep,snapshots=records))
            assignment=Path(parent['output'])/run/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz'
            with np.load(assignment) as z:neural_fields[identity]=(title,z['cluster'])
    template_path=Path(__file__).with_name('pacmap_md_view.html'); template=template_path.read_text()
    # Keep previously published completed views while the GPU sweep grows.
    roots=[Path(c['inherit_descriptors_from']),out] if 'inherit_descriptors_from' in c else [out]
    views={}
    for root in roots:
        for receipt in sorted((root/'technical/views').glob('*.json')):views[receipt.stem]=(root,receipt)
    if not views:raise ValueError('No completed PaCMAP views')
    for part in ('interactive','plots','assets','technical/views','technical/rendering'):(dest/part).mkdir(parents=True,exist_ok=True)
    # The inherited views already carry the pinned plotting asset before the
    # new GPU sweep starts producing its own output directory.
    shutil.copy2(roots[0]/'assets/plotly.min.js',dest/'assets/plotly.min.js')
    records={}; rows=[]
    default='S1-seed17-epoch24-encoder'
    for name,(root,receipt) in sorted(views.items()):
        r=json.loads(receipt.read_text()); data_path=root/'data'/f'{name}.npz'
        if sha(data_path)!=r['coordinate_sha256']:raise ValueError(f'Changed projection data: {name}')
        page_path=root/'interactive'/f'{name}.html'
        payload=json.loads(page_path.read_text().split('const D=',1)[1].split(';\nconst palette',1)[0])
        with np.load(data_path) as z:
            ids=z['original_row']
            for key in ('source','frame','atom'):
                if not np.array_equal(z[key],payload[key]):raise ValueError(f'Projection identity mismatch: {name}/{key}')
        match=re.match(r'^(S.+)-epoch(\d+)-(encoder|projector)-',name)
        own=f'{match[1]}-epoch{int(match[2])}-{match[3]}' if match else default
        if own not in neural_fields:own=default
        # Each MD selector has its matching PaCMAP coloring from the original
        # saved assignments, without fitting or guessing a color correspondence.
        for title, labels in neural_fields.values():
            payload['fields'][title]=labels[ids].tolist()
        payload['md']=dict(snapshots=snapshots,models=models,default_model=own)
        encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
        rendered=template.replace('__TITLE__',html.escape(payload['title'])).replace('__DATA__',encoded)
        target=dest/'interactive'/f'{name}.html'; temporary=target.with_suffix('.html.building')
        temporary.write_text(rendered);temporary.replace(target)
        shutil.copy2(root/'plots'/f'{name}.png',dest/'plots'/f'{name}.png')
        shutil.copy2(receipt,dest/'technical/views'/receipt.name)
        records[name]=dict(source_html_sha256=sha(page_path),published_html_sha256=sha(target),projection_sha256=r['coordinate_sha256'],rows=len(ids),dense_snapshots=len(snapshots),default_neural_model=own)
        rows.append(f'<tr><td>{html.escape(r["title"])}</td><td>{len(ids):,}</td><td><a href="plots/{name}.png">2D panels</a></td><td><a href="interactive/{name}.html">PaCMAP + two dense MD views</a></td></tr>')
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
          '<title>PaCMAP and dense MD gallery</title><style>body{font:16px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}a{color:#145bc0}</style>'
          '<h1>PaCMAP and dense MD comparison</h1><p>Each page includes 2D and 3D PaCMAP plus two independent MD views of the same full snapshot: neural clusters and TDA + bond-order + CNA clusters. Cluster colors agree with the corresponding PaCMAP color field. Atom selection between panels is disabled.</p>'
          f'<p>{len(rows)} completed projection views; {len(snapshots)} full MD snapshot(s) prepared. Dense neural coloring appears as frozen inference completes. All scientific projections and existing cluster assignments are preserved.</p>'
          '<table><thead><tr><th>Feature space / population</th><th>Samples</th><th>2D</th><th>Interactive</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></html>')
    temporary=dest/'index.html.building';temporary.write_text(page);temporary.replace(dest/'index.html')
    (dest/'README.md').write_text('# PaCMAP and dense MD comparison\n\n[Open gallery](index.html).\n\n'
        f'{len(rows)} completed views, {len(snapshots)} full dense snapshots. Two independent MD panels compare neural and classical descriptor clusters with matching PaCMAP palettes. No cross-panel atom selection. Dense assignments use frozen models; existing projections are unchanged.\n')
    rendering=dict(kind='dense MD display',atom_linkage=False,cluster_colors_preserved=True,source=str(out),dense_source=str(dense),implementation_sha256=sha(__file__),template_sha256=sha(template_path),views=records)
    (dest/'technical/rendering/md-space.json').write_text(json.dumps(rendering,indent=2)+'\n')
    publication=dict(mode='real copies plus recorded dense MD rendering',source=str(out),complete=(out/'technical/complete.json').exists(),rendering='rendering/md-space.json',files={str(p.relative_to(dest)):sha(p) for p in [dest/'index.html',dest/'README.md',*(dest/'interactive'/f'{n}.html' for n in records)]})
    (dest/'technical/publication.json').write_text(json.dumps(publication,indent=2)+'\n')
    print(f'Published {len(records)} pages with independent dense MD views: {dest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True);parser.add_argument('--dense-config',required=True)
    args=parser.parse_args();publish(args.config,args.dense_config)
