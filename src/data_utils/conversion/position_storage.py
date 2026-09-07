"""Quantize verified temporal trajectory positions, preserving all other arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from src.data_utils.temporal_lammps_binary import (
    TemporalLAMMPSBinaryTrajectory, write_temporal_lammps_binary,
)


class QuantizationError:
    """Per-coordinate minimum-image error in angstrom relative to float32 input."""
    def __init__(self):
        self.maximum = 0.0
        self.square_sum = 0.0
        self.count = 0

    def add(self, original, stored, lengths):
        delta = stored.astype(np.float32) - original
        delta -= lengths * np.rint(delta / lengths)
        self.maximum = max(self.maximum, float(np.max(np.abs(delta))))
        self.square_sum += float(np.einsum('ij,ij->', delta, delta, dtype=np.float64))
        self.count += delta.size
        if not np.all(np.isfinite(stored)):
            raise ValueError('Position storage overflow: float16 cannot represent this simulation box')

    def report(self):
        return {'reference_dtype': 'float32', 'metric': 'minimum-image per Cartesian coordinate',
                'max_abs_error_A': self.maximum, 'rms_error_A': (self.square_sum / self.count) ** 0.5,
                'coordinate_count': self.count}


def compress(binary_path: Path, *, delete_source=False):
    source = TemporalLAMMPSBinaryTrajectory.load(binary_path)
    if source.positions.dtype != np.float32:
        raise ValueError(f'Expected original float32 trajectory: {source.root}')
    original_checksums = source.verify_checksums()
    target = source.root.with_name(source.root.name.removesuffix('_float32') + '_float16')
    scratch = source.root.parent / 'float16_positions.building.npy'
    if target.exists() or scratch.exists():
        raise FileExistsError(f'Inspect interrupted or existing float16 conversion: {target}, {scratch}')
    positions = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float16, shape=source.positions.shape)
    error = QuantizationError()
    expected_hash = hashlib.sha256()
    for index, frame in enumerate(source.positions):
        positions[index] = frame
        error.add(frame, positions[index], source.box_high[index] - source.box_low[index])
        expected_hash.update(positions[index].tobytes())
        if index % 50 == 0:
            print(f'{source.root}: quantized {index + 1}/{source.frame_count}', flush=True)
    positions.flush()
    converted = write_temporal_lammps_binary(
        target, positions=positions, timesteps=source.timesteps, box_low=source.box_low,
        box_high=source.box_high, atom_ids=source.atom_ids, atom_types=source.atom_types,
        atom_columns=tuple(source.manifest['atom_columns']), source=source.manifest['source'],
        provenance={'float32_manifest': source.manifest, 'quantization': error.report()},
        consume_positions_file=scratch)
    checksums = converted.verify_checksums()
    if checksums['positions'] != expected_hash.hexdigest():
        raise RuntimeError(f'Float16 coordinates differ from the expected rounding: {target}')
    for name in original_checksums:
        if name != 'positions' and checksums[name] != original_checksums[name]:
            raise RuntimeError(f'Non-position array changed: {name}, {target}')
    report = {'state': 'complete', 'storage_dtype': 'float16', 'binary_path': str(target),
              'original_binary_path': str(source.root), 'checksums': checksums,
              'original_checksums': original_checksums, 'quantization': error.report(),
              'position_semantic_sha256': checksums['positions'],
              'position_bytes_saved': source.positions.nbytes - converted.positions.nbytes,
              'original_deleted': False}
    report_path = source.root.parent / 'float16_conversion.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    # Preserve the campaign verification entry point and its original provenance.
    campaign_report_path = source.root.parent / 'binary_conversion.json'
    if campaign_report_path.exists():
        previous = json.loads(campaign_report_path.read_text())
        updated = {**previous, **report, 'original_conversion': previous}
        campaign_report_path.write_text(json.dumps(updated, indent=2) + '\n')
    if delete_source:
        # Published target is checksum verified and fsynced before retiring originals.
        original_root = source.root
        del frame, positions, source
        shutil.rmtree(original_root)
        original_root.symlink_to(target.name, target_is_directory=True)
        report['original_deleted'] = True
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        if campaign_report_path.exists():
            updated['original_deleted'] = True
            campaign_report_path.write_text(json.dumps(updated, indent=2) + '\n')
        with report_path.open('rb') as handle:
            os.fsync(handle.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binaries', nargs='+', type=Path)
    parser.add_argument('--delete-source', action='store_true',
                        help='Delete verified float32 originals; keep old paths as compatibility symlinks.')
    args = parser.parse_args(argv)
    for path in args.binaries:
        print(json.dumps(compress(path, delete_source=args.delete_source), indent=2), flush=True)


if __name__ == '__main__':
    main()
