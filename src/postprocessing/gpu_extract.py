"""Bounded, opt-in GPU extraction with overlapped reading and compression.

This writes a separate cache and execution receipts. It never publishes or
replaces a birth catalogue. One process owns one GPU; CPU threads read/compress.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
from pathlib import Path
import time
import fcntl
import numpy as np
from src.experiment_runner.artifacts import file_hash, json_digest, write_json
from src.postprocessing.io import geometry, arrays
from src.postprocessing.descriptors.gpu import structure
from src.postprocessing.descriptors.gpu.ptm import implementation_files


def implementation():
    root = Path(__file__).parent
    files = [Path(__file__), root/'io.py', root/'descriptors/gpu/__init__.py',
             root/'descriptors/gpu/kernels.cu', *implementation_files()]
    return {str(p): file_hash(p) for p in files}


def save_chunk(root, start, frames, values, config, identity):
    """Same selective-detector NPZ layout as CPU extraction, separate provenance."""
    name = f'clusters-{start:06d}-{start+len(frames):06d}'
    path = root/(name+'.npz'); temporary = root/(name+'.building.npz')
    stacked = np.stack([v[0] for v in values])
    packet = {f'clusters_{d}':stacked[:,i] for i,d in enumerate(config['detectors'])}
    packet.update(ptm=np.stack([v[3] for v in values]), solid_counts=np.stack([v[1] for v in values]),
                  bond_histogram=np.stack([v[2] for v in values]), frames=frames)
    np.savez_compressed(temporary, **packet); temporary.replace(path)
    done = dict(identity=identity, start=start, stop=start+len(frames), frames=frames,
                layout='detector_arrays_v1', sha256=file_hash(path), bytes=path.stat().st_size)
    write_json(root/(name+'.json'), done)
    return done


def extract(source, frames, config, output, *, chunk_frames=16, prefetch=2, compression_workers=1,
            ptm_ordering='voronoi', execution='resident', input_receipt=None):
    """Process exact requested frames once with bounded read/write queues.

    An optional input receipt must come from checksum-verified CPU preparation;
    without one, verify raw arrays before extracting. Geometry still verifies
    source manifests on every newly opened part. Caches live outside the repo.
    """
    if chunk_frames < 1 or prefetch < 1 or compression_workers < 1:
        raise ValueError('Chunk size, prefetch and compression workers must be positive')
    if execution not in ('joint', 'resident', 'cuda') or ptm_ordering not in ('voronoi', 'distance'):
        raise ValueError('Explicit CUDA execution and PTM ordering required')
    frames = list(frames)
    if not frames or any(type(f) is not int for f in frames) or frames != sorted(set(frames)):
        raise ValueError('Frames must be nonempty, strictly increasing exact integer indices')
    root = Path(output).resolve(); repo = Path(__file__).resolve().parents[2]
    if root.is_relative_to(repo):
        raise ValueError('Write generated extraction caches outside the repository')
    root.mkdir(parents=True, exist_ok=True)
    definition = dict(source_id=source['id'], input_identity=source['input_identity'], frames=frames,
                      descriptor_config=config, ptm_ordering=ptm_ordering, execution=execution,
                      chunk_frames=chunk_frames, implementation=implementation())
    identity = json_digest(definition)
    with (root/'extract.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract = root/'extraction.json'
        if contract.exists() and json.loads(contract.read_text())['identity'] != identity:
            raise ValueError(f'Changed GPU extraction contract: {root}; use a separate output')
        write_json(contract, dict(identity=identity, definition=definition, source=source))
        verified = root/'input-verified.json'
        if verified.exists():
            if json.loads(verified.read_text())['input_identity'] != source['input_identity']:
                raise ValueError('Changed input identity')
        else:
            if input_receipt is not None:
                receipt = Path(input_receipt)
                if json.loads(receipt.read_text())['input_identity'] != source['input_identity']:
                    raise ValueError(f'Input verification receipt does not match source: {receipt}')
                provenance = dict(reused_receipt=str(receipt.resolve()), receipt_sha256=file_hash(receipt))
            else:
                from src.data.trajectories import array_sha256
                for part in source['parts']:
                    for key, value in arrays(part['path'],part['manifest_sha256']).items():
                        if array_sha256(value) != part['array_sha256'][key]:
                            raise ValueError(f'Changed input array: {part["path"]}/{key}')
                provenance = dict(verified_raw_arrays=True)
            write_json(verified, dict(input_identity=source['input_identity'], **provenance))
        completed=[]; pending=[]
        for start in range(0,len(frames),chunk_frames):
            stop=min(start+chunk_frames,len(frames)); stem=f'clusters-{start:06d}-{stop:06d}'
            receipt=root/(stem+'.json'); path=root/(stem+'.npz')
            if receipt.exists():
                done=json.loads(receipt.read_text())
                if done['identity']!=identity or done['frames']!=frames[start:stop] or file_hash(path)!=done['sha256']:
                    raise ValueError(f'Changed GPU chunk: {path}')
                completed.append(done)
            else:pending.extend((i,frames[i]) for i in range(start,stop))
        if not pending and (root/'complete.json').exists():
            return dict(json.loads((root/'complete.json').read_text()), reused=True)
        started=time.perf_counter(); reads=deque(); writes=deque(); values=[]; chunk=[]; next_item=iter(pending)
        # A single reader avoids duplicate manifest/memmap initialization; writer
        # futures never exceed compression_workers. Neither queue grows with time.
        with ThreadPoolExecutor(max_workers=1) as reader, ThreadPoolExecutor(max_workers=compression_workers) as writer:
            def enqueue():
                item=next(next_item,None)
                if item is not None:
                    i,f=item; reads.append((i,f,reader.submit(geometry,source,f)))
            for _ in range(prefetch):enqueue()
            while reads:
                i,frame,future=reads.popleft(); points,box=future.result(); enqueue()
                if not chunk:chunk_start=i
                result=structure(points,box,config,backend=execution,ptm_backend=ptm_ordering,include_ptm=True)
                chunk.append(frame); values.append(result)
                if (i+1)%chunk_frames==0 or i==len(frames)-1:
                    if len(writes)>=compression_workers:completed.append(writes.popleft().result())
                    writes.append(writer.submit(save_chunk,root,chunk_start,chunk,values,config,identity))
                    chunk=[];values=[]
            for future in writes:completed.append(future.result())
        done=dict(identity=identity,frames=len(frames),new_frames=len(pending),seconds=time.perf_counter()-started,
                  prefetch=prefetch,compression_workers=compression_workers,chunks=sorted(completed,key=lambda x:x['start']))
        write_json(root/'complete.json',done)
        return done


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();c=json.loads(Path(args.config).read_text())
    print(json.dumps(extract(**c)),flush=True)


if __name__=='__main__':main()
