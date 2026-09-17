"""Offline HTML browser and readable dataset cards from the same registry JSON."""
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from .paths import REPO


STYLE = '''
:root{color-scheme:light;--ink:#172b3b;--muted:#526678;--line:#d9e2ea;--blue:#165c9b}
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px 32px}h1{font-size:32px;letter-spacing:-.8px;margin:5px 0 12px}
h2{font-size:21px;margin-top:30px}a{color:var(--blue);overflow-wrap:anywhere}p{max-width:100ch}
.muted{color:var(--muted)}.stats{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.stat{background:white;border:1px solid var(--line);padding:12px 20px;border-radius:8px;min-width:150px}.stat strong{display:block;font-size:27px}
.filters{position:sticky;top:0;background:#f5f7faf5;padding:14px 0;display:flex;gap:10px;flex-wrap:wrap;z-index:2}
input,select{padding:10px;border:1px solid #bdcbd7;background:white;border-radius:6px;font:inherit}input{flex:1;min-width:260px}
.table-wrap{overflow:auto;background:white;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%}th,td{text-align:left;vertical-align:top;padding:12px 14px;border-bottom:1px solid var(--line)}th{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:.5px}td small{display:block;color:var(--muted);margin-top:4px;max-width:480px}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#eaf0f6;font-size:12px;margin:2px}.warning{background:#fff1d3;color:#775005}.bad{background:#ffe7e6;color:#842927}.good{background:#e3f3ed;color:#175844}
details{background:white;border:1px solid var(--line);border-radius:7px;padding:12px 16px;margin:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f7;padding:14px;font-size:12px;max-height:500px;overflow:auto}code{font-size:12px;overflow-wrap:anywhere}li{margin:5px 0}[hidden]{display:none!important}
@media(max-width:700px){main{padding:16px}h1{font-size:25px}td,th{padding:9px}.filters{position:static}}
'''


def esc(value):
    return html.escape(str(value), quote=True)


def url(path, base):
    return quote(os.path.relpath(path, base), safe='/')


def page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>'


def compact(value, limit=12):
    if not value:
        return 'Not recorded'
    if isinstance(value,list) and len(value)>limit:
        return json.dumps(value[:limit], ensure_ascii=False)+f' … ({len(value)} values; see JSON)'
    return json.dumps(value, ensure_ascii=False)


def render_registry(registry, output):
    output = Path(output)
    (output/'cards').mkdir(exist_ok=True)
    potentials = {p['id']:p for p in registry['potentials']}
    by_id = {d['id']:d for d in registry['datasets']}
    cards, markdown = [], ['# Dataset registry', '', '[Open the searchable browser](index.html) · [Potentials](potentials.md) · [Registry JSON](registry.json) · [CSV](datasets.csv) · [How to refresh](GUIDE.md)', '',
        f"Observed: {registry['generated_at']}", '', registry['methodology'], '',
        '| Dataset | Materials | Classification | Available | Binary records | Potential |',
        '| --- | --- | --- | --- | ---: | --- |']
    for d in registry['datasets']:
        card = output/'cards'/f'{d["slug"]}.html'
        potential_names = [potentials[k]['name'] for k in d['potential_ids']]
        count = d['observed']['available_complete_binary_records']
        available = d['location']['available']
        missing = ', '.join(d['missing_metadata']) or 'None in the core fields'
        badges = f'<span class="badge">{esc(d["classification"])}</span>'
        if d['issues']: badges += f'<span class="badge bad">{len(d["issues"])} integrity issues</span>'
        if d['missing_metadata']: badges += f'<span class="badge warning">{len(d["missing_metadata"])} metadata gaps</span>'
        search = ' '.join([d['id'],d['title'],d['description'],*d['materials'],*potential_names,
                           d['location']['resolved'],json.dumps(d['facts'])])
        cards.append(f'<tr data-search="{esc(search.lower())}" data-materials="{esc("|".join(d["materials"]))}" data-role="{esc(d["role"])}" data-classification="{esc(d["classification"])}" data-potentials="{esc("|".join(d["potential_ids"]))}">'
            f'<td><a href="cards/{d["slug"]}.html"><strong>{esc(d["title"])}</strong></a><small>{esc(d["description"])}</small><small><code>{esc(d["id"])}</code></small></td>'
            f'<td>{esc(", ".join(d["materials"]) or "Unknown")}</td><td>{badges}<small>{esc(d["role"])}</small></td>'
            f'<td><span class="badge {"good" if available else "bad"}">{"Available" if available else "Missing"}</span><small>{d["observed"]["allocated_bytes"]/2**30:.2f} GiB owned</small></td>'
            f'<td>{count:,} binary records<small>{d["observed"]["stored_frames"]:,} stored frames<br>{len(d["loose_arrays"]):,} array files indexed<br>{d["observed"]["binary_records_in_duplicate_groups"]} in duplicate groups</small></td>'
            f'<td>{esc("; ".join(potential_names) or "Unknown / not applicable")}</td></tr>')
        markdown.append(f'| [{d["title"].replace("|","/")}](cards/{d["slug"]}.md) | {", ".join(d["materials"]) or "Unknown"} | {d["classification"]} | {"yes" if available else "no"} | {count} | {"; ".join(potential_names) or "Unknown / not applicable"} |')
        record_path = output/d['records_file']
        record_data = json.loads(record_path.read_text())['records']
        rows = ''.join(f'<tr><td>{esc(k)}</td><td><code>{esc(compact(v))}</code></td></tr>' for k,v in d['facts'].items())
        links = ''.join(f'<li><a href="{url(r["path"],card.parent)}">{esc(r["relative_path"])}</a> · {esc(r.get("recorded_state") or r["kind"])} · <code>{esc(r["sha256"])}</code></li>' for r in record_data)
        potential_detail = ''
        for key in d['potential_ids']:
            potential = potentials[key]
            file_links = ''.join(f'<li><a href="{url(f["location"],card.parent)}">{esc(Path(f["location"]).name)}</a> · {"hash verified" if f["sha256_verified"] else "unavailable"}</li>' for f in potential['files'])
            potential_detail += f'<details><summary>{esc(potential["name"])}</summary><ul>{file_links}</ul><pre>{esc(json.dumps(potential,indent=2))}</pre></details>'
        def related(ids):
            return ', '.join(f'<a href="{by_id[k]["slug"]}.html">{esc(k)}</a>' if k in by_id else esc(k) for k in ids) or 'None recorded'
        evidence_links = []
        for reference in d['metadata'].get('evidence', []):
            resolved = reference
            for key,value in by_id.items():
                resolved = resolved.replace('${dataset:'+key+'}', value['location']['resolved'])
            evidence = Path(resolved)
            if not evidence.is_absolute(): evidence = REPO/evidence
            evidence_links.append(f'<li><a href="{url(evidence,card.parent)}">{esc(reference)}</a>{" (missing here)" if not evidence.exists() else ""}</li>')
        usage_links = ''.join(f'<li><a href="{url(REPO/p,card.parent)}">{esc(p)}</a></li>' for p in d['references'])
        body = f'<a href="../index.html">← All datasets</a><h1>{esc(d["title"])}</h1><p>{esc(d["description"])}</p>{badges}'
        body += f'<p><code>{esc(d["id"])}</code> · {esc(", ".join(d["materials"]) or "Materials unknown")}</p><h2>Location and availability</h2><p><a href="{url(d["location"]["resolved"],card.parent)}">{esc(d["location"]["resolved"])}</a></p>'
        body += f'<p>{d["observed"]["files"]:,} owned files · {d["observed"]["allocated_bytes"]/2**30:.3f} GiB allocated · {count:,} available complete binary records. Counts are not independent trajectories; nested registered datasets are excluded from parent storage.</p>'
        body += f'<h2>Use and provenance</h2><pre>{esc(json.dumps(d["metadata"],indent=2))}</pre><ul>{"".join(evidence_links)}</ul><p>Dependencies: {related(d["dependencies"])}<br>Nested datasets: {related(d["contains_registered"])}<br>Same resolved location: {related(d["same_location_as"])}</p>'
        body += '<h2>Generating potentials</h2>'+ (potential_detail or '<p>Not established from the inspected metadata. Do not infer a potential from the material name.</p>')
        body += f'<h2>Missing metadata and integrity</h2><p>{esc(missing)}</p><pre>{esc(json.dumps(d["issues"],indent=2))}</pre>'
        body += '<h2>Recorded scientific metadata</h2><p>Values retain their producer meaning; planned settings, source conditions and current arrays may describe different stages. Consult the linked record for its JSON field path.</p><div class="table-wrap"><table>'+rows+'</table></div>'
        body += f'<details><summary>Array schemas, file inventory and symlinks</summary><pre>{esc(json.dumps({k:d[k] for k in ("observed","loose_arrays","symlinks")},indent=2))}</pre></details>'
        body += f'<details><summary>Referenced by maintained configs/code ({len(d["references"])})</summary><p>A reference establishes discoverability, not that a training job consumed the dataset.</p><ul>{usage_links}</ul></details>'
        body += f'<details><summary>Producer evidence ({len(record_data)} records)</summary><p><a href="{url(record_path,card.parent)}">Complete structured metadata with field paths and current binary schemas</a></p><ul>{links}</ul></details>'
        body += f'<p class="muted">Observed {esc(registry["generated_at"])}. {esc(registry["methodology"])}</p>'
        card.write_text(page(d['title'],body))
        md = [f'# {d["title"]}', '', f'[All datasets](../README.md) · [Browsable card]({d["slug"]}.html) · [Full metadata](../{d["records_file"]})', '',
            d['description'], '', f'- ID: `{d["id"]}`', f'- Materials: {", ".join(d["materials"]) or "Unknown"}',
            f'- Classification: **{d["classification"]}**; role: {d["role"]}', f'- Location: `{d["location"]["resolved"]}`',
            f'- Present on this machine: {available}', f'- Potentials: {"; ".join(potential_names) or "Unknown / not applicable"}',
            f'- Complete binary records with arrays present: {count}; these are not independent-source counts.',
            f'- Stored frames: {d["observed"]["stored_frames"]}; duplicate-group records: {d["observed"]["binary_records_in_duplicate_groups"]}',
            f'- Allocated storage, excluding registered nested datasets: {d["observed"]["allocated_bytes"]/2**30:.3f} GiB',
            f'- Missing metadata: {missing}', '', '## Notes and relationships', '', '```json', json.dumps(d['metadata'],indent=2), '```', '',
            '## Recorded fields', '', '| Field | Values |', '| --- | --- |']
        md.extend(f'| {k} | {compact(v).replace("|","/").replace(chr(10)," ")} |' for k,v in d['facts'].items())
        md.extend(['', '## Evidence', '', f'All {len(record_data)} producer records, their hashes, field paths and current array schemas: [metadata JSON](../{d["records_file"]}).', '',
            'Sources remain at their original locations. Referenced configs/code do not prove actual training use.', '',
            f'Observed {registry["generated_at"]}.', '', registry['methodology']])
        card.with_suffix('.md').write_text('\n'.join(md)+'\n')
    materials = sorted({m for d in registry['datasets'] for m in d['materials']})
    def select(name, label, values):
        return f'<label>{label} <select id="{name}"><option value="">All</option>'+''.join(f'<option value="{esc(k)}">{esc(v)}</option>' for k,v in values)+'</select></label>'
    body = '<p class="muted">PointCloudMaterials / data inventory</p><h1>Find the data before starting another experiment</h1><p>Raw simulations, static structures, derived targets and caches. Search by metal, potential, conditions, dataset ID or path. Every collection has an evidence-linked card.</p>'
    body += '<p><a href="README.md">Markdown index</a> · <a href="potentials.html">Potential files and provenance</a> · <a href="registry.json">Registry JSON</a> · <a href="datasets.csv">CSV</a> · <a href="GUIDE.md">Refresh and register data</a> · <a href="../../output/registry/index.html">Experiments</a></p>'
    body += '<div class="stats">'+''.join(f'<div class="stat"><strong>{value}</strong>{label}</div>' for value,label in [(len(registry['datasets']),'registered collections'),(len(materials),'material labels'),(len(potentials),'potential definitions'),(sum(d['location']['available'] for d in registry['datasets']),'locations available')])+'</div>'
    body += '<div class="filters"><input id="search" type="search" aria-label="Search datasets" placeholder="Search Al, Ti, Mg, Ta, Zr, MEAM, history, temperature…">'
    body += select('material','Material',[(m,m) for m in materials])+select('role','Kind',[(v,v) for v in sorted({d['role'] for d in registry['datasets']})])+select('classification','Use',[(v,v) for v in sorted({d['classification'] for d in registry['datasets']})])+select('potential','Potential',[(p['id'],p['name']) for p in registry['potentials']])+'</div><p id="count" role="status"></p>'
    body += '<div class="table-wrap"><table id="datasets"><thead><tr><th>Dataset</th><th>Materials</th><th>Use / review</th><th>Storage</th><th>Binary evidence</th><th>Generating potential</th></tr></thead><tbody>'+''.join(cards)+'</tbody></table></div>'
    body += f'<h2>Directories awaiting registration ({len(registry["unregistered_directories"])})</h2><p>Visible discovery candidates, not automatically accepted datasets.</p><details><summary>Inspect candidates</summary><pre>{esc(json.dumps(registry["unregistered_directories"],indent=2))}</pre></details>'
    body += f'<h2>Other machines</h2><p>Reported copies are not additional independent data. Remote availability is not verified by this local scan.</p><pre>{esc(json.dumps(registry["remote_holdings"],indent=2))}</pre>'
    body += f'<p class="muted">Observed {esc(registry["generated_at"])}. {esc(registry["methodology"])}</p>'
    body += '''<script>
const controls=['search','material','role','classification','potential'].map(id=>document.getElementById(id));
const rows=[...document.querySelectorAll('#datasets tbody tr')];
function filter(){const [q,m,r,c,p]=controls.map(x=>x.value);let n=0;for(const row of rows){row.hidden=!(row.dataset.search.includes(q.toLowerCase())&&(!m||row.dataset.materials.split('|').includes(m))&&(!r||row.dataset.role===r)&&(!c||row.dataset.classification===c)&&(!p||row.dataset.potentials.split('|').includes(p)));if(!row.hidden)n++;}document.getElementById('count').textContent=`${n} of ${rows.length} collections`;}
controls.forEach(x=>x.addEventListener('input',filter));filter();
</script>'''
    (output/'index.html').write_text(page('Materials dataset registry',body))
    markdown.extend(['', '## Discovery and other machines', '', f'{len(registry["unregistered_directories"])} unregistered directories are listed in [registry.json](registry.json) and the browser. Remote holdings are reported separately; they are not reverified or counted as new independent data.'])
    (output/'README.md').write_text('\n'.join(markdown)+'\n')
    potential_body = '<a href="index.html">← All datasets</a><h1>Interaction potentials and model files</h1><p>File availability does not establish that a dataset was generated with that model. Dataset links below follow recorded potential fields or reviewed annotations.</p>'
    potential_md = ['# Interaction potentials and model files', '', '[All datasets](README.md) · [Browsable potential catalog](potentials.html)', '',
        'Supported potential elements are distinct from the species actually present in a simulation. File availability alone does not establish use.']
    for potential in registry['potentials']:
        users = [d for d in registry['datasets'] if potential['id'] in d['potential_ids']]
        links = ', '.join(f'<a href="cards/{d["slug"]}.html">{esc(d["title"])}</a>' for d in users) or 'No dataset match established'
        file_links = ''.join(f'<li><a href="{url(f["location"],output)}">{esc(Path(f["location"]).name)}</a> · {"hash verified" if f["sha256_verified"] else "unavailable"}<br><code>{esc(f["sha256"])}</code></li>' for f in potential['files'])
        potential_body += f'<h2 id="{esc(potential["id"])}">{esc(potential["name"])}</h2><p>{esc(potential["family"])} · supported elements: {esc(", ".join(potential["elements"]) or "not catalogued")}</p><p>{esc(potential["citation"] or "Citation not recorded")}</p><p>{esc(potential["mapping"])}</p><ul>{file_links}</ul><details><summary>Recorded metadata and provenance</summary><pre>{esc(json.dumps(potential,indent=2))}</pre></details><p>Dataset references: {links}</p>'
        potential_md.extend(['',f'## {potential["name"]}', '', f'- ID: `{potential["id"]}`',f'- Family: {potential["family"]}',f'- Mapping: {potential["mapping"]}',f'- Citation: {potential["citation"] or "Not recorded"}'])
        for item in potential['files']:
            potential_md.append(f'- [{Path(item["location"]).name}]({url(item["location"],output)}): `{item["sha256"]}`; '+('hash verified' if item['sha256_verified'] else 'unavailable here'))
        potential_md.extend(['','Dataset references: '+(', '.join(f'[{d["title"]}](cards/{d["slug"]}.md)' for d in users) or 'No match established')])
    (output/'potentials.html').write_text(page('Materials potential registry',potential_body))
    (output/'potentials.md').write_text('\n'.join(potential_md)+'\n')
    with (output/'datasets.csv').open('w',newline='') as stream:
        fields=['id','title','materials','kind','classification','location','available','allocated_bytes','complete_binary_records','potential_ids','missing_metadata','integrity_issues']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for d in registry['datasets']:
            writer.writerow(dict(id=d['id'],title=d['title'],materials=';'.join(d['materials']),kind=d['role'],classification=d['classification'],location=d['location']['resolved'],available=d['location']['available'],allocated_bytes=d['observed']['allocated_bytes'],complete_binary_records=d['observed']['available_complete_binary_records'],potential_ids=';'.join(d['potential_ids']),missing_metadata=';'.join(d['missing_metadata']),integrity_issues=';'.join(d['issues'])))
