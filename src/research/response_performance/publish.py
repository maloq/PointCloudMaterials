"""Plot already measured throughput; never rerun trajectories or replace metrics."""
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.experiment_runner.artifacts import file_hash, write_json
    out=Path(root)/'analyses/benchmark-v1'
    config=json.loads((Path(root)/'technical/code/config.json').read_text())
    records=[json.loads((out/'technical/variants'/f'{v["name"]}.json').read_text())
             for v in config['variants'] if (out/'technical/variants'/f'{v["name"]}.json').exists()]
    complete=[r for r in records if r['state']=='complete']
    validation=out/'technical/validation/complete.json'
    repeats={}
    repeat_files=sorted((out/'technical/validation').glob('repeat-*.json'))
    for path in repeat_files:
        for row in json.loads(path.read_text())['measurements']:
            repeats[(row['variant'],row['kind'])]=row
    if validation.exists():
        verified=json.loads(validation.read_text())
        complete=[r for r in complete if r['variant']['name'] in verified['energy_gated_variants']]
        repeats={(row['variant'],row['kind']):row for row in verified['measurements']}
    for record in complete:
        tolerance=config['tolerance'][record['variant']['dtype']]['static_relative']
        for field in ('energy','periodic_energy'):
            if record['gates'][0]['static'][field]['relative'] > tolerance:
                raise ValueError(f'Saved energy gate failed: {record["variant"]["name"]}/{field}')
        for row in record['measurements']:
            key=(row['variant'],row['kind'])
            if key in repeats:
                row['seconds_per_trajectory']=repeats[key]['seconds_per_trajectory']
    (out/'plots').mkdir(parents=True,exist_ok=True)
    reference=next(r for r in complete if r['variant']['name']=='reference64')
    baseline={r['kind']:r['seconds_per_trajectory'] for r in reference['measurements']}
    fig,axes=plt.subplots(1,2,figsize=(13,max(5,.43*len(complete)+1.8)),constrained_layout=True)
    for ax,kind in zip(axes,('value','response'),strict=True):
        rows=[next(row for row in r['measurements'] if row['kind']==kind) for r in complete]
        labels=[r['variant'] for r in rows]
        times=np.array([r['seconds_per_trajectory'] for r in rows])
        colors=['#ba7842' if r['dtype']=='float32' else '#29758a' for r in rows]
        ax.barh(np.arange(len(rows)),times,color=colors)
        ax.set_yticks(np.arange(len(rows)),labels=labels);ax.invert_yaxis()
        for i,(seconds,row) in enumerate(zip(times,rows,strict=True)):
            ax.text(seconds,i,f' {seconds:.2f}s · {baseline[kind]/seconds:.2f}×',va='center',fontsize=9)
        ax.set_xlim(0,times.max()*1.35);ax.set_xlabel('Seconds per completed query · lower is better')
        ax.set_title('Ordinary trajectory' if kind=='value' else 'Value + two responses')
    fig.suptitle('Fixed MACE teacher · Al256 · 100 fs · measured GPU throughput\nPassed numerical gates only; reference/control timings use isolated repeats' if repeats else
                 'Fixed MACE teacher · Al256 · 100 fs · preliminary throughput\nInitial reference overlapped other GPU inference; speedup factors are provisional')
    for suffix in ('png','pdf'):fig.savefig(out/'plots'/f'throughput.{suffix}',dpi=180)
    plt.close(fig)
    table=[]
    for record in complete:
        value=next(r for r in record['measurements'] if r['kind']=='value')
        response=next(r for r in record['measurements'] if r['kind']=='response')
        response_error=max([response['response_relative_error']]+[
            g['response']['relative'] for g in record['gates'] if 'response' in g])
        table.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
            record['variant']['name'],f'{value["seconds_per_trajectory"]:.2f}',
            f'{response["seconds_per_trajectory"]:.2f}',
            f'{baseline["response"]/response["seconds_per_trajectory"]:.2f}×',
            f'{response_error:.3g}',f'{response["peak_allocated_GiB"]:.2f}'))+'</tr>')
    failures=''.join(f'<li>{html.escape(r["variant"]["name"])}: {html.escape(r.get("error",""))}</li>'
                     for r in records if r['state']=='failed') or '<li>None in this sweep.</li>'
    note=('Where available, timings use completed isolated repeats of the same variant. '
          'The original reference overlapped PaCMAP inference; the original CSV is retained as measured.'
          if repeats else 'Preliminary: reference timings overlapped other GPU inference.')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Response-oracle performance</title>
<style>body{font:16px/1.55 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#172d38}
img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #d7e0e4}a{color:#17667e}</style>
<h1>Faster response acquisition with the same MACE teacher</h1>
<p>Al256 periodic cells · 1-fs BAOAB · 20/100-fs observables · two response directions.
Values and derivatives are compared against the archived eager float64 calculation with identical random streams.
No encoder fitting or new physical potential is involved.</p>'''
    page+=f'<p>{note}</p><img src="plots/throughput.png" alt="Measured value and response query throughput">'
    page+='<table><tr><th>Variant</th><th>Value s/query</th><th>Response s/query</th><th>Response speedup</th><th>Largest checked response relative error</th><th>Response peak GiB</th></tr>'+''.join(table)+'</table>'
    page+='''<p>Times divide batch elapsed time by completed queries. A finite-difference query includes five
trajectories. Float32 uses a separate approximation budget; it is not exact float64 equivalence.
The batch4 backends also check three additional displacement levels. These are short throughput
measurements on one GPU, not statistical timing confidence intervals.</p><h2>Rejected variants</h2><ul>'''+failures+'</ul>'
    page+='''<p><a href="tables/throughput.csv">Original sweep CSV</a> · <a href="tables/METRICS.md">Frozen measurement definitions</a> ·
<a href="technical/validation/repeat-reference64.json">Isolated reference repeat</a> ·
<a href="../default-acceptance-v1/technical/complete.json">Float32 collection/restart acceptance</a> ·
<a href="plots/throughput.pdf">PDF figure</a></p></html>'''
    (out/'index.html').write_text(page)
    shutil.copy2(__file__,out/'technical/publish.py')
    write_json(out/'technical/publication.json',dict(producer_sha256=file_hash(__file__),
        page_sha256=file_hash(out/'index.html'),
        inputs={r['variant']['name']:file_hash(out/'technical/variants'/f'{r["variant"]["name"]}.json') for r in records},
        passed_variants=[r['variant']['name'] for r in complete],
        validation_sha256=file_hash(validation) if validation.exists() else None,
        repeat_sha256={p.name:file_hash(p) for p in repeat_files},
        failed_variants=[r['variant']['name'] for r in records if r['state']=='failed']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',required=True)
    render(parser.parse_args().root)
