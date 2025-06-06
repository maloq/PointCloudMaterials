import sys,os
sys.path.append(os.getcwd())
import numpy as np
import logging 
import torch
from torch.utils.data import Dataset
from src.data_utils.prepare_data import get_regular_samples, get_random_samples
from src.data_utils.prepare_data import read_off_file
from src.utils.logging_config import setup_logging
import pandas as pd
from typing import Union, Optional, Sequence
from pathlib import Path

logger = setup_logging()



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class AtomicDataset(Dataset):
    
    def __init__(self, root,
                 data_files,
                 sample_type='regular',
                 radius=8,
                 overlap_fraction=0.0,
                 n_samples=1000,
                 num_points=100,
                 label=0,
                 pre_normalize=True):
        """Initialize the dataset with samples from OFF files.
        Args:
            root: Path to directory containing OFF files
            TODO: Add more details
        """
        self.root = root
        self.npoints = num_points
        self.radius = radius
        self.n_samples = n_samples
        self.label = label
        self.samples = []
        for off_file in data_files:
            logger.debug(f"Reading {off_file}")
            points = read_off_file(os.path.join(root, off_file), verbose=False)
            size = self.radius
            if sample_type == 'regular':
                samples = get_regular_samples(points,
                                              size=size,
                                              n_points=self.npoints,
                                              overlap_fraction=overlap_fraction)
            elif sample_type == 'random':
                samples = get_random_samples(points,
                                             n_samples=self.n_samples,
                                             size=size,
                                             n_points=self.npoints,
                                             overlap_fraction=overlap_fraction)
            else:
                raise ValueError(f"Invalid sample type: {sample_type}")

            if pre_normalize:
                import time
                start_time = time.time()
                samples = [pc_normalize(s).astype(np.float32) for s in samples]
                elapsed_time = time.time() - start_time
                logger.print(f"Pre-normalization took {elapsed_time:.4f} seconds")
            self.samples.extend(samples)
            logger.debug(f"Point set shape: {samples[0].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_set = self.samples[index]
        if point_set.dtype != np.float32:
            point_set = pc_normalize(point_set).astype(np.float32)
        point_set_tensor = torch.tensor(point_set, dtype=torch.float32)
        label_tensor = torch.tensor(self.label, dtype=torch.long)
        return point_set_tensor, label_tensor
    



class RegularDataset(Dataset):
    def __init__(
        self, 
        points: np.ndarray, 
        size: float,  #radius for spheric sampling
        n_points: int = 128, 
        overlap_fraction: float = 0.0,
    ):
        self.samples = get_regular_samples(
            points,
            size=size,
            n_points=n_points,
            return_coords=True,
            overlap_fraction=overlap_fraction,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_points, coords = self.samples[idx]
        sample_points = pc_normalize(sample_points).astype(np.float32)
        return torch.tensor(sample_points, dtype=torch.float32), coords


class SoapCoordDataset(Dataset):
    """
    PyTorch Dataset over one or more Parquet files that contain:
      • first num_coord_dims columns → Cartesian coords
      • remaining columns            → SOAP features

    Returns (soap, coord) per sample as torch.Tensors.
    """

    def __init__(
        self,
        parquet_paths: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        dtype: torch.dtype = torch.float32,
        preload: bool = True,
    ) -> None:
        # unify to list of Paths
        if isinstance(parquet_paths, (str, Path)):
            self.files = [Path(parquet_paths)]
        else:
            self.files = [Path(p) for p in parquet_paths]
            
        self.num_coord_dims = 3
        self.dtype = dtype
        self.preload = preload

        if preload:
            print("SOAP dataset in preload mode")
            # load & stack everything
            coords_list = []
            soap_list = []
            for p in self.files:
                mat = pd.read_parquet(p, engine="pyarrow").to_numpy()
                if mat.shape[1] <= 3:
                    raise ValueError(
                        f"{p!r} has only {mat.shape[1]} cols but "
                        f"num_coord_dims={3}."
                    )
                coords_list.append(mat[:, :3])
                soap_list.append(mat[:, 3:])
            all_coords = torch.as_tensor(
                np.vstack(coords_list), dtype=dtype
            )
            all_soap   = torch.as_tensor(
                np.vstack(soap_list), dtype=dtype
            )
            self.coords = all_coords
            self.soap   = all_soap
            self._len   = all_coords.shape[0]
        else:
            raise NotImplementedError("Lazy mode not implemented")
   

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < self._len):
            raise IndexError(f"Index {idx} out of range 0..{self._len-1}")

        if self.preload:
            return self.soap[idx], self.coords[idx]
    

if __name__ == '__main__':
    data = AtomicDataset(root="datasets/Al/inherent_configurations_off",
                         data_files=["240ps.off"],
                         n_samples=6000,
                         num_points=200,
                         label=0)
    
    loader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for i, (point, label) in enumerate(loader):
        print(point.shape)
        print(label.shape)
        if i > 3:
            break
        
    paths = ["datasets/Al/soap_features/166ps_selected.parquet",
             "datasets/Al/soap_features/170ps_selected.parquet",
             "datasets/Al/soap_features/174ps_selected.parquet",
             "datasets/Al/soap_features/175ps_selected.parquet",
             "datasets/Al/soap_features/177ps_selected.parquet"]


    ds = SoapCoordDataset(
        parquet_paths=paths,
        num_coord_dims=3,
        preload=True,     
    )
    print("SoapCoordDataset length: ", len(ds))

    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, pin_memory=True)

    for i, (soap, xyz) in enumerate(loader):
        print(soap.shape)
        print(xyz.shape)
        if i > 3:
            break