import time
import unittest
import numpy as np
import torch
from data_utils.prepare_data import get_cubic_samples



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


if __name__ == '__main__':
    unittest.main()