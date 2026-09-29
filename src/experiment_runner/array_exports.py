"""Numpy array export mechanics; callers own row selection and model inference."""

from pathlib import Path
import numpy as np


class ArrayExport:
    def __init__(self, folder, specifications):
        self.folder = Path(folder)
        self.specifications = specifications
        self.arrays = {}
        self.offset = 0

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, (shape, dtype) in self.specifications.items():
            self.arrays[name] = np.lib.format.open_memmap(
                self.folder / f'{name}.building.npy', mode='w+',
                shape=shape, dtype=dtype,
            )
        return self

    def write(self, start, **values):
        if values.keys() != self.arrays.keys():
            raise ValueError(f'{self.folder}: export fields differ: {tuple(values)}')
        if start != self.offset:
            raise ValueError(f'{self.folder}: expected row {self.offset}, received {start}')
        count = len(next(iter(values.values())))
        for name, value in values.items():
            target = self.arrays[name][start:start + count]
            if value.shape != target.shape:
                raise ValueError(f'{self.folder}/{name}: expected {target.shape}, received {value.shape}')
            self.arrays[name][start:start + len(value)] = value
        self.offset += count

    def __exit__(self, kind, error, traceback):
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        if kind is None:
            for name, (shape, _) in self.specifications.items():
                if self.offset != shape[0]:
                    raise ValueError(f'{self.folder}/{name}: incomplete export: '
                                     f'{self.offset}/{shape[0]} rows')
            for name in self.specifications:
                (self.folder / f'{name}.building.npy').replace(
                    self.folder / f'{name}.npy'
                )
        return False
