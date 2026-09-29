"""Copy completed visualization artifacts to the explicitly recorded absolute destination.

No path-alias rewriting, numerical calculation or model fitting is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def fingerprint(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def publish(config):
    c=json.loads(Path(config).read_text());source=Path(c['output']);dest=Path(c['publication'])
    if not source.is_absolute() or not dest.is_absolute():raise ValueError('Use a resolved submission config')
    if source.resolve()==dest.resolve():raise ValueError('Publication source equals destination')
    files=[]
    for part in ('plots','interactive','assets','tables'):
        directory=source/part
        if directory.exists():files.extend(p for p in directory.rglob('*') if p.is_file())
    for part in ('views','metric-contracts','table-contracts'):
        directory=source/'technical'/part
        if directory.exists():files.extend(p for p in directory.rglob('*') if p.is_file())
    for name in ('README.md','index.html','technical/complete.json','technical/metric-contract.json'):
        if (source/name).exists():files.append(source/name)
    hashes={}
    for path in files:
        relative=path.relative_to(source);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.copying');shutil.copy2(path,temporary);temporary.replace(target)
        digest=fingerprint(target)
        if digest!=fingerprint(path):raise ValueError(f'Publication copy differs: {relative}')
        hashes[str(relative)]=digest
    (dest/'technical').mkdir(parents=True,exist_ok=True)
    (dest/'technical/publication.json').write_text(json.dumps(dict(mode='real copies',source=str(source),
        complete=(source/'technical/complete.json').exists(),files=hashes),indent=2)+'\n')
    print(f'Published {len(files)} files as real copies: {dest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    publish(parser.parse_args().config)
