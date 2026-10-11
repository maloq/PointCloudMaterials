"""Pinned pure-Ni EAM/FS file and LAMMPS-compatible cubic interpolation.

Spline construction follows LAMMPS PairEAM::interpolate (GPL-2.0-or-later).
See NOTICE.md. No potential refitting, smoothing or table resampling.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np


@dataclass
class Tables:
    path: Path
    sha256: str
    mass: float
    cutoff: float
    dr: float
    drho: float
    embedding: np.ndarray
    density: np.ndarray
    pair_r: np.ndarray


def spline(values):
    y=np.asarray(values,dtype=np.float64)
    slope=np.empty_like(y)
    slope[0]=y[1]-y[0];slope[1]=(y[2]-y[0])/2
    slope[-2]=(y[-1]-y[-3])/2;slope[-1]=y[-1]-y[-2]
    slope[2:-2]=((y[:-4]-y[4:])+8*(y[3:-1]-y[1:-3]))/12
    dy=y[1:]-y[:-1]
    return np.stack((y[:-1],slope[:-1],3*dy-2*slope[:-1]-slope[1:],slope[:-1]+slope[1:]-2*dy),axis=1)


def read(path,expected_sha256=None):
    path=Path(path);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError(f'Potential checksum mismatch: {path}: {digest}')
    lines=raw.decode('ascii').splitlines()
    if lines[3].split()!=['1','Ni']:raise ValueError('Only the single-element Ni EAM/FS file is supported')
    nrho,drho,nr,dr,cut=lines[4].split();nrho=int(nrho);nr=int(nr)
    atom,mass,lattice,structure=lines[5].split()
    if int(atom)!=28 or structure.lower()!='fcc':raise ValueError('Unexpected Ni metadata')
    values=np.fromstring(' '.join(lines[6:]),sep=' ')
    if values.size!=nrho+2*nr or not np.isfinite(values).all():raise ValueError('Invalid EAM/FS table length or values')
    return Tables(path,digest,float(mass),float(cut),float(dr),float(drho),spline(values[:nrho]),spline(values[nrho:nrho+nr]),spline(values[nrho+nr:]))
