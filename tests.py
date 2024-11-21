import time
import unittest
import numpy as np
import torch


from data_utils.prepare_data import get_cubic_samples
from data_utils.data_load import AtomicDataset

# Fix the import in data_utils/data_load.py



class TestGetCubicSamples(unittest.TestCase):

    def test_speed_get_cubic_samples(self):
        points = np.random.rand(100000, 3)*267
        print(points.max(axis=0))
        print(points.min(axis=0))
        n_samples = 1000
        cube_size = 16.0

        start_time = time.time()
        samples = get_cubic_samples(points, n_samples, cube_size)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"get_cubic_samples took {elapsed_time:.4f} seconds")
        self.assertLess(elapsed_time, 5.0, "get_cubic_samples is too slow")


    def test_atomic_dataset_and_dataloader(self):
        dataset = AtomicDataset(
            root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
            data_files=["166ps.off", "170ps.off"],
            cube_size=16,
            n_samples=100,
            num_point=200,
            label=0
        )
        
        self.assertEqual(len(dataset), 100, "Dataset length should match n_samples")
        
        point_set, label = dataset[0]
        self.assertEqual(point_set.shape, (200, 3), "Point set should have correct shape")
        self.assertEqual(label, 0, "Label should match input label")
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        batch = next(iter(dataloader))
        points, labels = batch
        
        self.assertEqual(points.shape, (32, 200, 3), "Batch shape should be correct")
        self.assertEqual(labels.shape, (32,), "Labels shape should be correct")


if __name__ == '__main__':
    unittest.main()