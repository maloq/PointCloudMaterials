"""Verify sorted position dumps from the elemental campaign producer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.data_utils.temporal_lammps_binary import write_temporal_lammps_binary
from src.data_utils.conversion.position_storage import QuantizationError


def convert(branch: Path, *, delete_source: bool = False, storage_dtype: str = "float16") -> dict:
    metadata = json.loads((branch / 'metadata.json').read_text())
    if metadata['state'] != 'dynamics_complete':
        raise RuntimeError(f'Only validated complete dynamics can be converted: {branch}')
    source = branch / 'trajectory.lammpstrj'
    source_stat = source.stat()
    frames, atoms = metadata['frame_count'], metadata['atom_count']
    scratch = branch / 'conversion_positions.npy'
    if scratch.exists():
        raise FileExistsError(f'Interrupted conversion requires inspection: {scratch}')
    positions = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.dtype(storage_dtype),
                                         shape=(frames, atoms, 3))
    steps = np.arange(frames, dtype=np.int64) * metadata['dump_every_steps']
    ids = np.arange(1, atoms + 1, dtype=np.int64)
    low = np.empty((frames, 3), dtype=np.float32)
    high = np.empty_like(low)
    semantic_hash = hashlib.sha256()
    error = QuantizationError()
    with source.open() as handle:
        for frame in range(frames):
            header = TemporalLAMMPSDumpDataset._read_frame_header(handle, source_path=source)
            if header is None or header['num_atoms'] != atoms or header['timestep'] != steps[frame]:
                raise RuntimeError(f'Wrong frame/atom count/timestep at frame {frame}: {source}')
            if tuple(header['atom_columns']) != ('id', 'type', 'x', 'y', 'z'):
                raise RuntimeError(f'Unexpected position columns: {header}')
            table = np.loadtxt(handle, max_rows=atoms)
            if table.shape != (atoms, 5) or not np.array_equal(table[:, 0], ids) or not np.all(table[:, 1] == 1):
                raise RuntimeError(f'Changed IDs/types or incomplete frame {frame}: {source}')
            low[frame], high[frame] = header['box_low'], header['box_high']
            lengths = high[frame] - low[frame]
            wrapped = np.mod(table[:, 2:5].astype(np.float32) - low[frame], lengths)
            wrapped = np.minimum(wrapped, np.nextafter(lengths, np.zeros(3, dtype=np.float32)))
            positions[frame] = wrapped
            error.add(wrapped, positions[frame], lengths)
            semantic_hash.update(positions[frame].tobytes())
            if frame % 100 == 0:
                print(f'{branch.name}: decoded {frame + 1}/{frames} frames', flush=True)
        if handle.read().strip():
            raise RuntimeError(f'Unexpected trailing frames: {source}')
    positions.flush()
    with source.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    binary = write_temporal_lammps_binary(
        branch / f'trajectory_binary_{storage_dtype}', positions=positions, timesteps=steps,
        box_low=low, box_high=high, atom_ids=ids, atom_types=np.ones(atoms, dtype=np.int32),
        atom_columns=('id', 'type', 'x', 'y', 'z'),
        source={'trajectory_lammpstrj': str(source), 'sha256': source_hash},
        provenance={'producer': 'src.simulation.campaigns.elemental',
                    'metadata': metadata, 'position_semantic_sha256': semantic_hash.hexdigest(),
                    'quantization': error.report()},
        consume_positions_file=scratch)
    checksums = binary.verify_checksums()
    actual_hash = hashlib.sha256()
    for frame in range(frames):
        actual_hash.update(binary.positions[frame].tobytes())
    if actual_hash.hexdigest() != semantic_hash.hexdigest():
        raise RuntimeError(f'Binary coordinates differ from decoded source: {branch}')
    del positions
    report = {'state': 'complete', 'storage_dtype': storage_dtype, 'quantization': error.report(), 'binary_path': str(binary.root), 'frame_count': frames,
              'atom_count': atoms, 'source_sha256': source_hash, 'checksums': checksums,
              'position_semantic_sha256': semantic_hash.hexdigest(), 'raw_text_preserved': True}
    (branch / 'binary_conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    if delete_source:
        current = source.stat()
        if (current.st_size, current.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError(f'Source changed during conversion; refusing deletion: {source}')
        report['source_deleted_apparent_bytes'] = current.st_size
        report['source_deleted_allocated_bytes'] = current.st_blocks * 512
        source.unlink()
        report['raw_text_preserved'] = False
        (branch / 'binary_conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('branch', type=Path)
    parser.add_argument('--delete-source', action='store_true',
                        help='Remove raw text only after checksum and coordinate verification.')
    parser.add_argument('--storage-dtype', choices=('float16', 'float32'), default='float16')
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.branch.resolve(), delete_source=args.delete_source, storage_dtype=args.storage_dtype), indent=2))


if __name__ == '__main__':
    main()
