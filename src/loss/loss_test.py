import time
import torch
import sys,os
sys.path.append(os.getcwd())

from src.loss.reconstruction_loss import calculate_rdf_old, calculate_rdf_fast


def main():
    # Select the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Parameters for RDF calculation
    batch_size = 64
    point_size = 64
    sphere_radius = 5.0
    dr = 0.05
    drop_first_n_bins = 2

    # Create a random point cloud with shape (B, 3, N)
    point_cloud = torch.rand((batch_size, 3, point_size), device=device)

    # Compute RDF using the old (loop-based) implementation and the fast (vectorized) version.
    with torch.no_grad():
        rdf_old, r_mid_old = calculate_rdf_old(point_cloud, sphere_radius, dr, drop_first_n_bins=drop_first_n_bins)
        rdf_fast, r_mid_fast = calculate_rdf_fast(point_cloud, sphere_radius, dr, drop_first_n_bins=drop_first_n_bins)

    # Compare the outputs: they should be almost identical.
    rdf_close = torch.allclose(rdf_old, rdf_fast, rtol=1e-3, atol=1e-3)
    r_mid_close = torch.allclose(r_mid_old, r_mid_fast, rtol=1e-3, atol=1e-3)

    if rdf_close and r_mid_close:
        print("Success: The RDF and radial midpoint outputs from both implementations match!")
    else:
        print("Error: Discrepancies found between implementations.")
        print("Max diff in RDF:", (rdf_old - rdf_fast).abs().max().item())
        print("Max diff in r_mid:", (r_mid_old - r_mid_fast).abs().max().item())

    # Optional warm-up for GPU timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Now, compare the runtime speed of each function.
    n_iter = 20

    # Timing the old implementation
    total_time_old = 0.0
    for _ in range(n_iter):
        start = time.perf_counter()
        _ = calculate_rdf_old(point_cloud, sphere_radius, dr, drop_first_n_bins=drop_first_n_bins)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Ensure GPU work is complete before timing
        end = time.perf_counter()
        total_time_old += (end - start)
    avg_time_old = total_time_old / n_iter

    # Timing the fast implementation
    total_time_fast = 0.0
    for _ in range(n_iter):
        start = time.perf_counter()
        _ = calculate_rdf_fast(point_cloud, sphere_radius, dr, drop_first_n_bins=drop_first_n_bins)
        if device.type == "cuda":
            torch.cuda.synchronize()  # Ensure GPU work is complete before timing
        end = time.perf_counter()
        total_time_fast += (end - start)
    avg_time_fast = total_time_fast / n_iter

    print(f"\nAverage time over {n_iter} iterations:")
    print(f" - Old implementation: {avg_time_old * 1000:.3f} ms per iteration")
    print(f" - Fast implementation: {avg_time_fast * 1000:.3f} ms per iteration")

if __name__ == '__main__':
    main()