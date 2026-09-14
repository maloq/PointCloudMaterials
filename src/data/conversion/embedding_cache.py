"""Convert a paused forecast cache to float16 and publish its new storage protocol."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.registry import sha256, write_json
from .training_cache import convert_array


def convert_cache(config_path, producer_path, audit_path):
    config = json.loads(Path(config_path).read_text())['data']
    root = Path(config['cache'])
    audit_path = Path(audit_path)
    marker = root / 'storage_migration.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit['config'] != config or audit['producer_sha256'] != sha256(
            producer_path
        ):
            raise ValueError(
                f'Migration config or producer changed: {audit_path}'
            )
    else:
        original = json.loads((root / 'protocol.json').read_text())
        if config != dict(original['config'], storage_dtype='float16'):
            raise ValueError(
                'Embedding migration may change only storage_dtype to float16.'
            )
        if original['config'].get('storage_dtype', 'float32') != 'float32':
            raise ValueError(
                'Embedding migration requires the original float32 protocol.'
            )
        if marker.exists():
            raise FileExistsError(
                f'Resume using the recorded migration audit: {marker}'
            )
        shards = [
            json.loads(p.read_text())
            for p in sorted(root.glob('source_*/manifest.json'))
        ]
        audit = dict(
            state='planned',
            config=config,
            producer_sha256=sha256(producer_path),
            original_protocol=original,
            original_shards=shards,
            original_manifest=(
                json.loads((root / 'manifest.json').read_text())
                if (root / 'manifest.json').exists()
                else None
            ),
        )
        write_json(audit_path, audit)
    write_json(marker, dict(state='running', audit=str(audit_path.resolve())))
    records, shards = [], []
    try:
        for original in audit['original_shards']:
            directory = root / original['directory']
            path = directory / 'embeddings.npy'
            for name, digest in original['checksums'].items():
                if (
                    name != 'embeddings.npy'
                    and sha256(directory / name) != digest
                ):
                    raise ValueError(
                        'Embedding metadata checksum changed:'
                        f' {directory / name}'
                    )
            values = np.load(path, mmap_mode='r')
            if values.shape != (original['centers'], original['frames'], 256):
                raise ValueError(
                    f'Unexpected forecast embedding shape: {path},'
                    f' {values.shape}'
                )
            sidecar = path.with_suffix('.float16.json')
            if values.dtype == np.float32:
                if sha256(path) != original['checksums']['embeddings.npy']:
                    raise ValueError(
                        f'Original embedding checksum changed: {path}'
                    )
            elif values.dtype == np.float16:
                if (
                    json.loads(sidecar.read_text())['source_sha256']
                    != original['checksums']['embeddings.npy']
                ):
                    raise ValueError(
                        'Converted embeddings have different provenance:'
                        f' {path}'
                    )
            else:
                raise ValueError(
                    f'Unsupported embedding dtype: {path}, {values.dtype}'
                )
            del values
            record = convert_array(
                path,
                last_dim=256,
                interpretation=(
                    'Lossy storage of frozen forecast embeddings; windows'
                    ' decode to float32. Scaling and metrics use these rounded'
                    ' embeddings.'
                ),
            )
            shard = dict(original, checksums=dict(original['checksums']))
            shard['checksums']['embeddings.npy'] = record['float16_sha256']
            write_json(directory / 'manifest.json', shard)
            records.append(record)
            shards.append(shard)
            print(
                'Verified float16 shard'
                f' {len(shards)}/{len(audit["original_shards"])}:'
                f' {directory.name}',
                flush=True,
            )
        protocol = dict(
            audit['original_protocol'],
            config=config,
            producer_sha256=audit['producer_sha256'],
        )
        write_json(root / 'protocol.json', protocol)
        if audit['original_manifest'] is not None:
            write_json(
                root / 'manifest.json',
                dict(
                    audit['original_manifest'],
                    protocol=protocol,
                    shards=shards,
                ),
            )
        audit.update(
            state='complete',
            converted_shards=len(records),
            saved_payload_bytes=sum(
                r['source_payload_bytes'] - r['target_payload_bytes']
                for r in records
            ),
            max_absolute_error=max(
                (r['max_absolute_error'] for r in records), default=0
            ),
            rmse=(
                (
                    sum(r['rmse'] ** 2 * np.prod(r['shape']) for r in records)
                    / sum(np.prod(r['shape']) for r in records)
                )
                ** 0.5
                if records
                else 0
            ),
        )
        write_json(audit_path, audit)
        write_json(
            marker, dict(state='complete', audit=str(audit_path.resolve()))
        )
        return audit
    except BaseException as error:
        write_json(
            marker,
            dict(
                state='failed',
                audit=str(audit_path.resolve()),
                error=repr(error),
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help='New forecast config; only data.storage_dtype changes.',
    )
    parser.add_argument(
        '--producer',
        required=True,
        type=Path,
        help='Exact data.py used by the new preparation snapshot.',
    )
    parser.add_argument(
        '--audit',
        required=True,
        type=Path,
        help='Retained original protocol/manifests and conversion report.',
    )
    args = parser.parse_args(argv)
    result = convert_cache(args.config, args.producer, args.audit)
    print(
        f'Converted {result["converted_shards"]} shards; saved'
        f' {result["saved_payload_bytes"] / 2**30:.3f} GiB'
    )


if __name__ == '__main__':
    main()
