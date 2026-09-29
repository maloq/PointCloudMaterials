"""One shared cohort's geometry, independent of encoder checkpoint and treatment."""
from contextlib import contextmanager
import fcntl
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import digest, sha, write_json
from src.research.structural_state.data import graph_arrays
from src.research.structured_context import geometry
from .cache import RetainedCache


class GeometryCache:
    def __init__(self, root, cutoff):
        self.root=root;self.cutoff=cutoff

    def frame(self, source, frame, rows, atoms, parent, *, pin_memory):
        from .data import paired_frame
        start=time.perf_counter();stem=f'{source["id"]}-{frame}'
        dest=self.root/f'{stem}.npz';receipt=self.root/f'{stem}.json'
        with (self.root/f'{stem}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            hit=receipt.exists()
            if not hit:
                views,inverse,queries=paired_frame(source,frame,atoms,parent)
                values=dict(rows=rows,inverse=inverse,atoms=queries)
                for domain,view in views.items():
                    arrays=graph_arrays(view['patches'],self.cutoff)
                    values.update({f'{domain}_{k}':v for k,v in arrays.items()})
                    values[f'{domain}_actual']=view['actual']
                temporary=dest.with_suffix('.tmp.npz');np.savez(temporary,**values);temporary.replace(dest)
                write_json(receipt,dict(sha256=sha(dest)))
            if sha(dest)!=json.loads(receipt.read_text())['sha256']:
                raise ValueError(f'Shared geometry changed: {dest}')
            with np.load(dest) as values:
                np.testing.assert_array_equal(values['rows'],rows)
                np.testing.assert_array_equal(values['atoms'][:,0],atoms)
                output=dict(frame=frame,ids=values['rows'],inverse=values['inverse'],atoms=values['atoms'])
                # Read just the requested domain; the file contains both domains.
                domain=self.domain
                arrays={k:values[f'{domain}_{k}'] for k in ('positions','edges','offsets','edge_offsets')}
                arrays['groups']=np.repeat(np.arange(len(arrays['offsets'])-1),np.diff(arrays['offsets']))
                for key in ('positions','edges','groups'):
                    value=torch.from_numpy(arrays[key]);arrays[key]=value.pin_memory() if pin_memory else value
                output.update(actual=values[f'{domain}_actual'],arrays=arrays)
        output.update(prepare_s=time.perf_counter()-start,geometry_cache_hit=hit,read_membership_s=0. if hit else time.perf_counter()-start)
        return output


@contextmanager
def shared_geometry(study, domain, cutoff, deadline):
    from .data import paired_frame
    inventory=study.technical/'inventory.json'
    plan=json.loads((study.technical/'plan.json').read_text())
    if sha(inventory)!=plan['inventory_sha256']:raise ValueError('Prepared geometry inventory changed')
    # Includes exact row/source ancestry, coordinate quantization, membership,
    # edge ordering and geometric stencil; excludes encoder/training settings.
    repo=Path(__file__).resolve().parents[3]
    files=['src/data/trajectories/shooting.py','src/data/trajectories/lammps.py']
    metadata=dict(protocol='paired_context_geometry_v1',inventory_sha256=sha(inventory),cutoff=cutoff,
        paired_frame=inspect.getsource(paired_frame),graph_arrays=inspect.getsource(graph_arrays),
        stencil_sha256=sha(geometry.__file__),producer_sha256=sha(__file__),
        trajectory_producers={name:sha(repo/name) for name in files})
    key=digest(metadata)
    cache=RetainedCache(resolve_path(study.config['cache_policy']['geometry']),1)
    with cache.lease(key,deadline=deadline,metadata=metadata,shared=True) as root:
        with (study.technical/'geometry-reference.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            write_json(study.technical/f'geometry-cache-{domain}.json',dict(key=key,path=str(root),cutoff=cutoff))
        result=GeometryCache(root,cutoff);result.domain=domain
        yield result


def prepare(study,deadline):
    """CPU producer, reusable across every treatment and both domains."""
    from .data import inventory,population,parent_plan
    from .features import prepared_frames
    frozen=json.loads((study.technical/'inventory.json').read_text())
    if inventory(study.config)!=frozen:raise ValueError('Shared geometry ancestry changed')
    pop=population(study.config);parent=parent_plan(study.config)
    sources={s['id']:s for s in parent['sources']}
    base=json.loads(resolve_path(study.config['base_configs']['hot']).read_text())
    runtime=study.config['extraction'];count=0
    with shared_geometry(study,'hot',base['encoder']['cutoff'],deadline) as cache:
        items=[(int(sid),int(frame)) for sid in np.unique(pop['source'])
               for frame in np.unique(pop['frame'][pop['source']==sid])]
        def build(item):
            if time.time()>deadline-120:raise TimeoutError('Shared geometry preparation checkpointed')
            sid,frame=item;rows=np.flatnonzero((pop['source']==sid)&(pop['frame']==frame))
            cache.frame(sources[sid],frame,rows,pop['atom'][rows],parent,pin_memory=False)
            return item
        with prepared_frames(items,build,workers=runtime['workers'],capacity=runtime['prefetch']) as ready:
            for sid,frame in ready:
                count+=1
                if count%100==0:print(json.dumps(dict(stage='geometry',frames=count,total=len(items))),flush=True)
        write_json(cache.root/'complete.json',dict(frames=count,inventory_sha256=sha(study.technical/'inventory.json')))
    return dict(frames=count)
