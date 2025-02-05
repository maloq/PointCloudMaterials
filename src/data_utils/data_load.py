import sys,os
sys.path.append(os.getcwd())
import numpy as np
import logging 
import torch
from torch.utils.data import Dataset
from src.data_utils.prepare_data import get_regular_samples, get_random_samples
from src.data_utils.prepare_data import read_off_file
from src.utils.logging_config import setup_logging
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
                 sample_shape='spheric',
                 sample_type='regular',
                 radius=8,
                 cube_side=16,
                 overlap_fraction=0.0,
                 n_samples=1000,
                 num_points=100,
                 label=0,
                 pre_normalize=True):
        """Initialize the dataset with cubic samples from OFF files.
        Args:
            root: Path to directory containing OFF files
            TODO: Add more details
        """
        self.root = root
        self.npoints = num_points
        self.cube_side = cube_side
        self.radius = radius
        self.n_samples = n_samples
        self.label = label
        self.samples = []
        for off_file in data_files:
            logger.debug(f"Reading {off_file}")
            points = read_off_file(os.path.join(root, off_file), verbose=False)
            if sample_shape == 'spheric':
                size = self.radius
            elif sample_shape == 'cubic':
                size = self.cube_side
            else:
                raise ValueError(f"Invalid sample shape: {sample_shape}")
            if sample_type == 'regular':
                
                samples = get_regular_samples(points,
                                              sample_shape=sample_shape,
                                              size=size,
                                              n_points=self.npoints,
                                              overlap_fraction=overlap_fraction)
            elif sample_type == 'random':
                samples = get_random_samples(points,
                                             sample_shape=sample_shape,
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
        label = np.array(self.label).astype(np.int32)
        return point_set, label
    

class CubeDataset(Dataset):
    def __init__(
        self, 
        points: np.ndarray, 
        size: float, # cube_size actually
        n_points: int = 128, 
        overlap_fraction: float = 0.0
    ):
        self.samples = get_regular_samples(
            points,
            size=size, 
            n_points=n_points,
            return_coords=True,
            overlap_fraction=overlap_fraction,
            sample_shape='cubic'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cube_points, coords = self.samples[idx]        
        cube_points = pc_normalize(cube_points).astype(np.float32)
        return torch.tensor(cube_points, dtype=torch.float32), coords


class SphericDataset(Dataset):
    def __init__(
        self, 
        points: np.ndarray, 
        size: float, # radius actually
        n_points: int = 128, 
        overlap_fraction: float = 0.0
    ):
        self.samples = get_regular_samples(
            points,
            size=size,
            n_points=n_points, 
            return_coords=True,
            overlap_fraction=overlap_fraction,
            sample_shape='spheric'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spheres, coords = self.samples[idx]        
        spheres = pc_normalize(spheres).astype(np.float32)
        return torch.tensor(spheres, dtype=torch.float32), coords


if __name__ == '__main__':
    data = AtomicDataset(root="datasets/Al/inherent_configurations_off",
                         data_files=["240ps.off"],
                         cube_size=16,
                         n_samples=6000,
                         num_points=200,
                         label=0)
    
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)


