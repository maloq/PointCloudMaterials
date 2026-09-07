"""Convert explicitly selected derived neighborhood NPY caches to float16."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np

from src.data_utils.temporal_campaign import write_json


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def convert_file(path: Path):
    if not (path.name in ('clouds.npy', 'benchmark_clouds.npy')
            or path.name.endswith(('.views.npy', '.context.npy'))):
        raise ValueError(f'Not a supported derived neighborhood cache: {path}')
    record_path = path.with_suffix('.float16.json')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.dtype == np.float16:
        record = json.loads(record_path.read_text())
        if sha256(path) != record['float16_sha256']:
            raise RuntimeError(f'Converted cache checksum changed: {path}')
        record['state'] = 'complete'
        record['source_payload_bytes'] = int(np.prod(record['shape'])) * 4
        record['target_payload_bytes'] = int(np.prod(record['shape'])) * 2
        record['target_allocated_bytes'] = path.stat().st_blocks * 512
        write_json(record_path, record)
        return record
    if values.dtype != np.float32 or values.shape[-1] != 3:
        raise ValueError(f'Expected float32 neighborhood coordinates ending in xyz: {path}, {values.shape}, {values.dtype}')
    stat = path.stat()
    temporary = path.with_name(path.name + '.float16-building')
    if temporary.exists():
        raise FileExistsError(f'Inspect interrupted conversion before retry: {temporary}')
    step = max(1, (16 * 1024 * 1024) // (values[0].size * 4))
    # Complete range validation before creating or replacing any array.
    for start in range(0, len(values), step):
        chunk = values[start:start + step]
        if not np.isfinite(chunk).all() or np.abs(chunk).max() > np.finfo(np.float16).max:
            raise ValueError(f'Non-finite or float16-overflowing coordinates at row {start}: {path}')
    source_hash = sha256(path)
    target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=values.shape)
    maximum_error = 0.0
    square_error = 0.0
    for start in range(0, len(values), step):
        chunk = values[start:start + step]
        target[start:start + step] = chunk.astype(np.float16)
    target.flush()
    del target
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    target = np.load(temporary, mmap_mode='r')
    for start in range(0, len(values), step):
        chunk = values[start:start + step]
        np.testing.assert_array_equal(target[start:start + step], chunk.astype(np.float16))
        error = target[start:start + step].astype(np.float32) - chunk
        maximum_error = max(maximum_error, float(np.abs(error).max()))
        square_error += float(np.sum(error.astype(np.float64) ** 2))
    record = dict(state='verified_before_replace', path=str(path), shape=list(values.shape),
                  source_dtype='float32', storage_dtype='float16', source_sha256=source_hash,
                  float16_sha256=sha256(temporary), max_absolute_error=maximum_error,
                  rmse=(square_error / values.size) ** .5, source_allocated_bytes=stat.st_blocks * 512,
                  source_payload_bytes=values.nbytes, target_payload_bytes=values.size * 2,
                  target_allocated_bytes=temporary.stat().st_blocks * 512,
                  interpretation='Lossy storage of derived local coordinates; decode to float32 for geometry.')
    del target, values
    if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f'Cache changed during conversion; original retained: {path}')
    write_json(record_path, record)
    os.chmod(temporary, stat.st_mode & 0o777)
    os.replace(temporary, path)
    record['state'] = 'complete'
    record['target_allocated_bytes'] = path.stat().st_blocks * 512
    write_json(record_path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path, help='JSON with explicit files and optional wait_for_processes guards.')
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    status = args.plan.with_suffix('.status.json')
    records = []
    try:
        if 'wait_for_processes' in plan:
            write_json(status, dict(state='waiting_for_training_queue', processes=plan['wait_for_processes']))
            for process in plan['wait_for_processes']:
                if socket.gethostname() != process['host']:
                    raise RuntimeError(f'Process guard belongs to {process["host"]}, not this host')
                stat_path = Path(f'/proc/{process["pid"]}/stat')
                while True:
                    try:
                        stat = stat_path.read_text().rsplit(')', 1)[1].split()
                    except FileNotFoundError:
                        break  # The guarded process has exited.
                    if stat[19] != process['start_ticks'] or stat[0] == 'Z':
                        break
                    time.sleep(15)
        for index, name in enumerate(plan['files']):
            path = Path(name)
            write_json(status, dict(state='running', completed=len(records), total=len(plan['files']), current=str(path)))
            records.append(convert_file(path))
            if index % 10 == 0:
                print(f'Converted {index + 1}/{len(plan["files"])}: {path}', flush=True)
        write_json(status, dict(state='complete', count=len(records),
                               saved_payload_bytes=sum(r['source_payload_bytes']-r['target_payload_bytes'] for r in records),
                               maximum_coordinate_error=max((r['max_absolute_error'] for r in records), default=0)))
    except BaseException as error:
        write_json(status, dict(state='failed', completed=len(records), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
