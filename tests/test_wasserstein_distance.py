import torch
import numpy as np
from scipy.stats import wasserstein_distance
import sys,os
sys.path.append(os.getcwd())
from src.loss.reconstruction_loss import wasserstein_distance_loss

def test_wasserstein_loss_single():
    nbins = 10
    # Create two delta distributions:
    # - rdf1 has all mass at bin index 2.
    # - rdf2 has all mass at bin index 7.
    rdf1 = torch.zeros(1, nbins)
    rdf2 = torch.zeros(1, nbins)
    rdf1[0, 2] = 1.0
    rdf2[0, 7] = 1.0

    # PyTorch function expects batched input
    loss = wasserstein_distance_loss(rdf1, rdf2)

    bin_centers = np.arange(nbins)
    scipy_loss = wasserstein_distance(
        bin_centers, bin_centers,
        u_weights=rdf1.numpy()[0],
        v_weights=rdf2.numpy()[0]
    )
    print("Test 1: Delta functions")
    print("PyTorch Wasserstein loss:", loss.item())
    print("SciPy Wasserstein loss: ", scipy_loss)


def test_wasserstein_loss_random():
    nbins = 20
    batch_size = 5

    rdf1 = torch.rand(batch_size, nbins)
    rdf2 = torch.rand(batch_size, nbins)

    loss = wasserstein_distance_loss(rdf1, rdf2)

    bin_centers = np.arange(nbins)
    scipy_losses = []
    for i in range(batch_size):
        scipy_loss = wasserstein_distance(
            bin_centers, bin_centers,
            u_weights=rdf1[i].numpy(),
            v_weights=rdf2[i].numpy()
        )
        scipy_losses.append(scipy_loss)
    scipy_avg = np.mean(scipy_losses)

    print("Test 2: Random distributions")
    print("PyTorch Wasserstein loss (average over batch):", loss.item())
    print("SciPy average Wasserstein loss:", scipy_avg)

def test_wasserstein_loss_fixed():
    nbins = 10

    rdf1 = torch.tensor([[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0]])
    rdf2 = torch.tensor([[0.0, 0.0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.0]])

    loss = wasserstein_distance_loss(rdf1, rdf2)

    bin_centers = np.arange(nbins)
    scipy_loss = wasserstein_distance(
        bin_centers, bin_centers,
        u_weights=rdf1.numpy()[0],
        v_weights=rdf2.numpy()[0]
    )

    print("Test 3: Fixed distributions")
    print("PyTorch Wasserstein loss:", loss.item())
    print("SciPy Wasserstein loss: ", scipy_loss)
    print()


if __name__ == "__main__":
    print("Running tests for Wasserstein distance loss functions:")
    test_wasserstein_loss_single()
    test_wasserstein_loss_random()
    test_wasserstein_loss_fixed()