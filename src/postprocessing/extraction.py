"""Shared full-cell structure extraction and physical-time embryo tracking.

No prediction-history filtering. No single PTM atom defines an embryo or its fate.
"""
from src.postprocessing.descriptors.structure import structure
from src.postprocessing.descriptors.ptm import producer_identity

import fcntl
from functools import lru_cache
import os
from pathlib import Path
import time

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path
from src.postprocessing.io import arrays, geometry
from src.postprocessing.descriptors.ptm import full_labels
from src.postprocessing.io import read


@lru_cache(maxsize=8)
def resolved_root(value):
    return resolve_path(value)


def source_root(p,s):
    return resolved_root(p['config']['cache'])/'sources'/s['id']


def extract(p,task):
    s=next(s for s in p['sources'] if s['id']==task['source']);root=source_root(p,s)
    root.mkdir(parents=True,exist_ok=True)
    name=f'clusters-{task["start"]:06d}-{task["stop"]:06d}'
    identity=digest(dict(plan=p['identity'],task=task))
    with (root/(name+'.lock')).open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        receipt=root/(name+'.json');target=root/(name+'.npz')
        if receipt.exists():
            done=read(receipt)
            if done['identity']!=identity or sha(target)!=done['sha256']:raise ValueError(f'Changed extraction {target}')
            return done
        started=time.monotonic()
        with (root/'input.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            verified=root/'input-verified.json'
            if verified.exists():
                if read(verified)['input_identity']!=s['input_identity']:raise ValueError('Input identity changed')
            else:
                from src.data.trajectories import array_sha256
                for part in s['parts']:
                    a=arrays(part['path'],part['manifest_sha256'])
                    for k,v in a.items():
                        if array_sha256(v)!=part['array_sha256'][k]:raise ValueError(f'Changed input array {part["path"]}/{k}')
                write_json(verified,dict(input_identity=s['input_identity'],parts=s['parts']))
        values=[];counts=[];hist=[];types=[]
        selected=s['sampled_frames'][task['start']:task['stop']]
        for f in selected:
            data,n,h,ptm=structure(*geometry(s,f),p['config'],include_ptm=True)
            values.append(data);counts.append(n);hist.append(h);types.append(ptm)
        temp=target.with_suffix('.building.npz')
        # Separate ZIP members let a tracking worker inflate only its detector.
        # np.savez's former [frame, detector, atom] member required all detectors.
        stacked=np.stack(values)
        packet={f'clusters_{detector}':stacked[:,i] for i,detector in enumerate(p['config']['detectors'])}
        packet.update(ptm=np.stack(types),solid_counts=counts,frames=selected)
        if hist[0] is not None:packet['bond_histogram']=hist
        np.savez_compressed(temp,**packet)
        temp.replace(target)
        done=dict(identity=identity,**task,sha256=sha(target),seconds=time.monotonic()-started,
            layout='detector_arrays_v1',
            ptm_producer_sha256=producer_identity(),
            ptm_execution_sha256=sha(Path(full_labels.__code__.co_filename)),
            descriptor_threads=int(os.environ.get('OVITO_THREAD_COUNT','1')),
            descriptor_producer_sha256=sha(Path(__file__)))
        write_json(receipt,done)
        return done


class Frames:
    def __init__(self,p,s,detector):
        self.p,self.s,self.detector=p,s,detector
        self.lookup={f:i for i,f in enumerate(s['sampled_frames'])}
        self.root=source_root(p,s)
        self.loaded=-1;self.values=None;self.types=None;self.ptm_producer=None
        self.has_ptm=False;self.path=None;self.checksum=None

    def labels(self,f):
        i=self.lookup[f];chunk=self.p['config']['chunk_frames'];start=i//chunk*chunk
        if self.loaded!=start:
            stop=min(start+chunk,len(self.s['sampled_frames']));name=f'clusters-{start:06d}-{stop:06d}'
            root=self.root;done=read(root/(name+'.json'))
            if sha(root/(name+'.npz'))!=done['sha256']:raise ValueError('Changed extracted structures')
            with np.load(root/(name+'.npz'),allow_pickle=False) as a:
                if done.get('layout')=='detector_arrays_v1':
                    self.values=a[f'clusters_{self.detector}']
                elif 'layout' not in done:
                    self.values=a['clusters'][:,self.p['config']['detectors'].index(self.detector)].copy()
                else:raise ValueError(f'Unknown extraction layout: {root/name}')
                # Historical extraction chunks discarded atom-level PTM;
                # absence means it must be computed, never inferred from
                # minimum-size component labels (which omit small embryos).
                self.has_ptm='ptm' in a.files;self.types=None
                self.ptm_producer=done.get('ptm_producer_sha256')
            self.path=root/(name+'.npz')
            self.checksum=done['sha256']
            self.loaded=start
        return self.values[i-start]

    def ptm_labels(self,f,cutoff,producer):
        self.labels(f)
        if not self.has_ptm:return None
        if cutoff!=self.p['config']['ptm_rmsd_cutoff'] or producer!=self.ptm_producer:return None
        return self.raw_ptm(f)

    def raw_ptm(self,f):
        """Exact classifications retained by this checksum-verified extraction."""
        self.labels(f)
        if not self.has_ptm:
            raise ValueError(f'Atom-level PTM was not retained: {self.root}, frame {f}')
        if self.types is None:
            if sha(self.path)!=self.checksum:raise ValueError(f'Changed extracted PTM: {self.path}')
            with np.load(self.path,allow_pickle=False) as a:self.types=a['ptm']
        return self.types[self.lookup[f]-self.loaded]


def cluster_members(labels):
    atoms=np.flatnonzero(labels)
    if not len(atoms):return {}
    atoms=atoms[np.argsort(labels[atoms],kind='stable')]
    split=np.flatnonzero(np.diff(labels[atoms]))+1
    return {int(labels[x[0]]):x for x in np.split(atoms,split)}


def position(points,box,members):
    x=points[members];d=x-x[0];d-=box*np.rint(d/box)
    return np.mod(x[0]+d.mean(0),box),bool(np.any(np.ptp(d,axis=0)>box/2))
