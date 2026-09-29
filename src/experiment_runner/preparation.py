"""Verified preparation shards with explicit producer progress fields."""

from dataclasses import dataclass
import json
from pathlib import Path

from .artifacts import file_hash, write_json


@dataclass
class PreparationShard:
    root: Path
    identity: str

    def verified(self, **expected):
        path = self.root / 'complete.json'
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        for name, value in dict(expected, identity=self.identity).items():
            if record[name] != value:
                raise ValueError(f'Changed prepared shard: {path}: {name}')
        for name, checksum in record['files'].items():
            if file_hash(self.root / name) != checksum:
                raise ValueError(f'Changed prepared file: {self.root / name}')
        return record

    @property
    def resuming(self):
        return (self.root / 'progress.json').exists()

    def offset(self, field, *, total):
        if not self.resuming:
            return 0
        path = self.root / 'progress.json'
        progress = json.loads(path.read_text())
        if progress['identity'] != self.identity:
            raise ValueError(f'Changed partial preparation: {path}')
        offset = progress[field]
        if not 0 <= offset <= total:
            raise ValueError(f'{path}: invalid {field}={offset} for {total} rows')
        return offset

    def progress(self, **fields):
        write_json(self.root / 'progress.json', dict(fields, identity=self.identity))

    def complete(self, filenames, **fields):
        record = dict(fields, identity=self.identity,
                      files={name: file_hash(self.root / name) for name in filenames})
        write_json(self.root / 'complete.json', record)
        return record
