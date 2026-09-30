"""Explicit data sidecars for offline cluster viewers and their browser assets."""
import base64
import html
import json
from pathlib import Path
import re

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json


PAGE_SCHEMA = 'spatial_cluster_view_v1'
ASSET_SCHEMA = 'spatial_cluster_asset_v1'
SLOTS = re.compile(r'__([A-Z_]+)__')


def read_payload(path, schema=PAGE_SCHEMA):
    sidecar = Path(path).with_suffix('.json')
    record = json.loads(sidecar.read_text())
    if record['schema'] != schema:
        raise ValueError(f'Wrong viewer payload schema in {sidecar}: {record["schema"]}; expected {schema}')
    return record['data']


def fill_template(template, **slots):
    return SLOTS.sub(lambda match: slots.get(match[1], match[0]), template)


def write_page(path, template, payload, *, title=None, base='', **slots):
    values = dict(TITLE=html.escape(title or payload['title']), BASE=base,
        DATA=json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c'), **slots)
    missing = set(SLOTS.findall(template)) - values.keys()
    if missing:
        raise ValueError(f'Unresolved viewer template slots in {path}: {sorted(missing)}')
    rendered = fill_template(template, **values)
    write_json(Path(path).with_suffix('.json'), dict(schema=PAGE_SCHEMA, data=payload))
    temporary = Path(path).with_suffix('.html.building')
    temporary.write_text(rendered); temporary.replace(path)


def write_comparison(path, template, payload, *, title=None, index=False):
    write_page(path, template, payload, title=title, base='<base href="./interactive/">' if index else '',
        SCRIPT_HASH=sha(Path(__file__).with_name('cluster_comparison.js'))[:16],
        EXTENSION_HASH=sha(Path(__file__).with_name('viewer_extensions.js'))[:16])


def md_template(*, static=False):
    return fill_template(Path(__file__).with_name('pacmap_md_view.html').read_text(),
        MD_POPULATION='dense static grid' if static else 'full snapshot', MD_TITLE='static grid' if static else 'full snapshot',
        MD_COORDINATES=('actual nonperiodic source coordinates; the outline shows coordinate bounds, not a periodic cell. '
            'Each grid center uses the full-source neighbor context' if static else 'actual periodic-cell coordinates'))


def write_asset(path, key, data, variable='MD_SNAPSHOTS'):
    write_json(Path(path).with_suffix('.json'), dict(schema=ASSET_SCHEMA, namespace=variable, key=key, data=data))
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    temporary = Path(path).with_suffix('.building.js')
    temporary.write_text(f'window.{variable}=window.{variable}||{{}};window.{variable}[{json.dumps(key)}]={encoded};\n')
    temporary.replace(path)


def read_asset(path):
    return read_payload(path, ASSET_SCHEMA)


def write_vector_asset(path, key, z, data):
    """Pack float32 browser vectors while retaining their explicit data payload."""
    packed = np.asarray(z, dtype='<f4'); rows, width = packed.shape
    write_json(Path(path).with_suffix('.json'), dict(schema=ASSET_SCHEMA,
        namespace='TRAVEL_EMBEDDINGS', key=key, data=dict(data, z=packed.tolist())))
    encoded = base64.b64encode(packed.tobytes()).decode('ascii')
    script = 'window.TRAVEL_EMBEDDINGS=window.TRAVEL_EMBEDDINGS||{};(()=>{'
    script += f'const text=atob({json.dumps(encoded)}),bytes=Uint8Array.from(text,c=>c.charCodeAt(0)),view=new DataView(bytes.buffer);'
    script += f'const data={json.dumps(data,separators=(",",":"),allow_nan=False)};'
    script += f'data.z=Array.from({{length:{rows}}},(_,i)=>Array.from({{length:{width}}},(_,j)=>view.getFloat32(4*(i*{width}+j),true)));'
    script += f'window.TRAVEL_EMBEDDINGS[{json.dumps(key)}]=data;}})();\n'
    Path(path).write_text(script)
