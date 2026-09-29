"""Bounded disposable caches. Checkpoints and research results never live here."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import digest, sha, write_json


class RetainedCache:
    """Global LRU admission with process leases, including across Slurm lanes.

    Lock files stay outside entries: eviction must never create a second lock
    inode for a live key. Kernel locks disappear automatically on process death.
    Only marked, immediate children of this cache's entries directory are deleted.
    """
    def __init__(self, root, limit):
        self.root=Path(root).resolve();self.limit=limit
        if limit<1:raise ValueError('Positive cache retention required')
        self.entries=self.root/'entries';self.entries.mkdir(parents=True,exist_ok=True)
        (self.root/'locks').mkdir(exist_ok=True)

    def _record(self, path):
        if path.is_symlink() or path.resolve().parent!=self.entries:
            raise ValueError(f'Unsafe disposable cache entry: {path}')
        record=json.loads((path/'entry.json').read_text())
        if record['key']!=path.name or record['owner']!='equivariant_context_cache_v1':
            raise ValueError(f'Unmanaged directory in cache: {path}')
        return record

    def _evict(self, path):
        record=self._record(path)
        with (self.root/'locks'/f'{path.name}.lock').open('a') as lease:
            try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
            # The registry lock remains held through deletion and replacement.
            with (self.root/'evictions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(record,evicted_at=time.time(),path=str(path)))+'\n')
            shutil.rmtree(path)
            return True

    @contextmanager
    def lease(self, key, *, deadline, metadata, shared=False):
        if not re.fullmatch('[0-9a-f]{64}',key):raise ValueError(f'Invalid cache key: {key}')
        path=self.entries/key
        # Network filesystems require read access for a shared (read) lock.
        with (self.root/'locks'/f'{key}.lock').open('a+') as lease:
            while True:
                if time.time()>deadline-120:raise TimeoutError(f'Waiting for a disposable cache slot: {path}')
                admitted=False
                with (self.root/'registry.lock').open('a') as registry:
                    fcntl.flock(registry,fcntl.LOCK_EX)
                    entries={p:self._record(p) for p in self.entries.iterdir()}
                    # Free space before admission: the count never exceeds limit.
                    needed=0 if path in entries else 1
                    for candidate in sorted(entries,key=lambda p:entries[p]['last_used']):
                        if len(entries)+needed<=self.limit:break
                        if candidate!=path and self._evict(candidate):del entries[candidate]
                    if len(entries)+needed<=self.limit:
                        try:
                            fcntl.flock(lease,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
                        except BlockingIOError:pass
                        else:
                            if path not in entries:
                                path.mkdir()
                                record=dict(owner='equivariant_context_cache_v1',key=key,metadata=metadata)
                            else:
                                record=entries[path]
                                if record['metadata']!=metadata:raise ValueError(f'Cache key collision: {path}')
                            record['last_used']=time.time()
                            write_json(path/'entry.json',record)
                            admitted=True
                if admitted:break
                time.sleep(2)
            try:yield path
            finally:
                with (self.root/'registry.lock').open('a') as registry:
                    fcntl.flock(registry,fcntl.LOCK_EX)
                    record=self._record(path);record['last_used']=time.time()
                    write_json(path/'entry.json',record)
                fcntl.flock(lease,fcntl.LOCK_UN)


@contextmanager
def encoder_cache(study, domain, checkpoint, deadline):
    policy=study.config['cache_policy']
    metadata=dict(identity=study.identity,domain=domain,encoder_sha256=sha(checkpoint))
    key=digest(metadata)
    cache=RetainedCache(resolve_path(policy['features']),policy['encoders_kept'])
    with cache.lease(key,deadline=deadline,metadata=metadata) as root:
        # This small durable pointer is a receipt, not a promise of cache residency.
        write_json(study.technical/f'feature-cache-{domain}.json',dict(metadata,key=key,path=str(root)))
        yield root
