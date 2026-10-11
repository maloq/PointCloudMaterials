"""Restore the October 2 archived viewers from their explicit JSON payloads."""
import json
from pathlib import Path
import shutil

import numpy as np
from plotly.offline import get_plotlyjs

from src.data.fixed_cohort.protocol import sha, write_json
from .comparison_layout import refresh
from .viewer_payload import read_payload, write_asset, write_vector_asset


def asset_paths(value):
    if isinstance(value, dict):
        for key, entry in value.items():
            if key == 'asset':
                yield entry
            else:
                yield from asset_paths(entry)
    elif isinstance(value, list):
        for entry in value:
            yield from asset_paths(entry)


def restore(config):
    from .response_comparison import load
    c,_=load(config)
    archive=Path(c['viewer_archive']); destination=Path(c['viewer_root'])
    repository=Path(__file__).resolve().parents[3]
    inventory=json.loads((archive.parents[1]/'preserved.json').read_text())
    expected={str((repository/r['saved']).resolve()):r['sha256'] for r in inventory}
    copied=set();assets={};unavailable={}

    def verified(source):
        digest=sha(source)
        if digest!=expected[str(source.resolve())]:raise ValueError('Changed archived viewer: '+str(source))
        return digest

    def copy(source):
        target=destination/source.relative_to(archive)
        if target in copied:return
        verified(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);copied.add(target)

    bundles=[('mace-rh2-best-20260930/analyses/matched','matched','comparison-all_test.html'),
        ('mace-rh2-best-20260930/analyses/static','static','comparison-all_static.html')]
    unavailable['GeoFormer']=['full held-out MD labels','samples','lattice','travel']
    for relative,kind,primary in bundles:
        source=archive/relative;dest=destination/relative
        copy(source/'index.json');copy(source/'index.html')
        payload=read_payload(source/'index.html')
        (dest/'interactive').mkdir(exist_ok=True)
        shutil.copy2(dest/'index.json',dest/'interactive'/Path(primary).with_suffix('.json'))
        shutil.copy2(dest/'index.html',dest/'interactive'/primary)
        for url in sorted(set(asset_paths(payload))):
            source_asset=(source/'interactive'/url).resolve().with_suffix('.json')
            target=destination/source_asset.relative_to(archive)
            if str(target) in assets:continue
            digest=verified(source_asset);record=json.loads(source_asset.read_text())
            if record['schema']!='spatial_cluster_asset_v1':raise ValueError('Wrong asset schema '+str(source_asset))
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)==digest and target.with_suffix('.js').exists():
                pass
            elif record['namespace']=='TRAVEL_EMBEDDINGS':
                data=dict(record['data']);z=np.asarray(data.pop('z'),np.float32)
                write_vector_asset(target.with_suffix('.js'),record['key'],z,data)
            else:
                write_asset(target.with_suffix('.js'),record['key'],record['data'],record['namespace'])
            # Keep the preserved JSON byte-for-byte, including historical formatting.
            shutil.copy2(source_asset,target)
            assets[str(target)]=dict(archived=str(source_asset),payload_sha256=digest,js_sha256=sha(target.with_suffix('.js')))
        for folder in ('tables','technical/metric-contracts','technical/table-contracts'):
            for path in (source/folder).rglob('*'):
                if path.is_file():copy(path)
        original=source/'technical/metric-contract.json'
        if original.exists():copy(original)
        (dest/'assets').mkdir(exist_ok=True)
        (dest/'assets/plotly.min.js').write_text(get_plotlyjs())
        refresh(dest,kind)
        print('Restored archived comparison: '+relative,flush=True)
    # The descriptor references retain their own immutable metric definitions.
    for kind in ('matched','static'):
        source=archive/'general-descriptors-20260930/analyses'/kind
        for folder in ('tables','technical/metric-contracts','technical/table-contracts'):
            for path in (source/folder).rglob('*'):
                if path.is_file():copy(path)
    write_json(Path(c['output'])/'technical/viewer-recovery.json',dict(archive=str(archive),
        assets=assets,scientific_payloads_unchanged=True,metric_recomputation=False,
        neural_training=False,checkpoint_weights_restored=False,unavailable_geoformer_sections=unavailable))
