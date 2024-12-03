
import sys,os
sys.path.append(os.getcwd())

import numpy as np
import warnings
import pickle
import logging 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from src.data_utils.prepare_data import read_off_file, get_cubic_samples, get_spheric_samples
from src.data_utils.prepare_data import get_regular_cubic_samples, get_regular_spheric_samples


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



class AtomicDataset(Dataset):
    
    def __init__(self, root, data_files, sample_shape = 'spheric', sample_type = 'regular', radius=8, cube_size=16, n_samples=1000, num_points=100, label=0):
        """Initialize the dataset with cubic samples from OFF files.
        Args:
            root: Path to directory containing OFF files
            TODO: Add more details
        """
        self.root = root
        self.npoints = num_points
        self.cube_size = cube_size
        self.radius = radius
        self.n_samples = n_samples
        self.label = label
        logging.debug(f"Sample number of points {num_points}")
        for off_file in data_files:
            print(f"Reading {off_file}")
            points = read_off_file(os.path.join(root, off_file), verbose=False)
            if sample_type == 'regular' and sample_shape == 'cubic':
                self.samples = get_regular_cubic_samples(points,
                                                         cube_size=self.cube_size,
                                                         n_points=self.npoints)
            elif sample_type == 'regular' and sample_shape == 'spheric':
                self.samples = get_regular_spheric_samples(points, 
                                                           radius=self.radius,
                                                           n_points=self.npoints)
            elif sample_type == 'random' and sample_shape == 'cubic':
                self.samples = get_cubic_samples(points,
                                                n_samples=self.n_samples,
                                                cube_size=self.cube_size,
                                                n_points=self.npoints)
            elif sample_type == 'random' and sample_shape == 'spheric':
                self.samples = get_spheric_samples(points,
                                                   n_samples=self.n_samples,
                                                   radius=self.radius,
                                                   n_points=self.npoints)
            else:
                raise ValueError(f"Invalid sample type: {sample_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_set = self.samples[index]
        assert len(point_set) == self.npoints, f"Expected {self.npoints}, got {len(point_set)}"
        point_set = pc_normalize(point_set).astype(np.float32)
        label = np.array(self.label).astype(np.int32)
        
        return point_set, label


if __name__ == '__main__':
    data = AtomicDataset(root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
                         data_files=["240ps.off"],
                         cube_size=16,
                         n_samples=6000,
                         num_point=200,
                         label=0)
    
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
