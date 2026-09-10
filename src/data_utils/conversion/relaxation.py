"""Verified float16 storage for one converged full-cell relaxed snapshot."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.data_utils.temporal_lammps_binary import write_temporal_lammps_binary
from src.data_utils.conversion.position_storage import QuantizationError
from src.data_utils.temporal_campaign import write_json
from src.simulation.relaxation import sha256


def read_relaxed(directory):
    directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text())
    path=directory/'relaxed.dump'
    with path.open() as handle:
        header=TemporalLAMMPSDumpDataset._read_frame_header(handle,source_path=path)
        table=np.loadtxt(handle,max_rows=metadata['atom_count'])
        if handle.read().strip():raise ValueError(f'Unexpected extra frames in {path}')
    np.testing.assert_array_equal(table[:,0],np.arange(1,metadata['atom_count']+1))
    np.testing.assert_array_equal(table[:,1],np.ones(metadata['atom_count']))
    np.testing.assert_allclose(header['box_low'],metadata['box_low'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(header['box_high'],metadata['box_high'],rtol=0,atol=1e-10)
    if header['timestep']!=metadata['source_timestep'] or metadata['state']!='relaxed':
        raise ValueError(f'Invalid relaxed-frame identity or state: {directory}')
    return table[:,2:5],metadata


def convert(directory,delete_source=False):
    directory=Path(directory);x,metadata=read_relaxed(directory)
    low=np.asarray(metadata['box_low'],dtype=np.float32);high=np.asarray(metadata['box_high'],dtype=np.float32)
    x=np.mod(x-low,high-low).astype(np.float32);stored=x.astype(np.float16)
    error=QuantizationError();error.add(x,stored,high-low)
    source_hash=sha256(directory/'relaxed.dump')
    binary=write_temporal_lammps_binary(directory/'relaxed_binary_float16',positions=stored[None],
        timesteps=np.array([metadata['source_timestep']],dtype=np.int64),box_low=low[None],box_high=high[None],
        atom_ids=np.arange(1,len(x)+1,dtype=np.int64),atom_types=np.ones(len(x),dtype=np.int32),
        atom_columns=('id','type','x','y','z'),source=dict(path=str(directory/'relaxed.dump'),sha256=source_hash),
        provenance=dict(producer='src.simulation.relaxation',metadata=metadata,quantization=error.report()))
    checksums=binary.verify_checksums();np.testing.assert_array_equal(binary.positions[0],stored)
    report=dict(state='complete',checksums=checksums,source_sha256=source_hash,quantization=error.report(),
                training_precision='Training neighborhoods are extracted before global float16 storage, then stored as centered float16 offsets.',source_deleted=False)
    write_json(directory/'conversion.json',report)
    if delete_source:
        (directory/'relaxed.dump').unlink();(directory/'input.data').unlink()
        report['source_deleted']=True;write_json(directory/'conversion.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--delete-source',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(convert(args.directory,args.delete_source),indent=2))
