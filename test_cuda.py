# cuda_test_torch.py
import time, torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is NOT visible to PyTorch – aborting.")

device = torch.device("cuda")
print("CUDA device     :", torch.cuda.get_device_name(0))

# Warm-up (first CUDA call lazily loads kernels)
torch.randn(1, device=device)  

# A tiny benchmark – matrix multiply on GPU vs CPU
M, N, K = 4096, 4096, 4096            # change sizes if you run out of memory
cpu_a = torch.randn(M, K)
cpu_b = torch.randn(K, N)

# --- GPU ---
gpu_a = cpu_a.to(device)
gpu_b = cpu_b.to(device)
torch.cuda.synchronize()
t0 = time.perf_counter()
gpu_c = gpu_a @ gpu_b                # matmul
torch.cuda.synchronize()
t_gpu = time.perf_counter() - t0

# --- CPU (for comparison) ---
t0 = time.perf_counter()
cpu_c = cpu_a @ cpu_b
t_cpu = time.perf_counter() - t0

# Verify results
max_diff = (gpu_c.cpu() - cpu_c).abs().max().item()

print(f"\nResults:")
print(f"  GPU time : {t_gpu:7.3f} s")
print(f"  CPU time : {t_cpu:7.3f} s")
print(f"  Max |Δ|   : {max_diff:e}")

# --- PyTorch3D test ---
print("\n--- PyTorch3D Test ---")
try:
    import pytorch3d
    from pytorch3d.ops import knn_points
    print(f"PyTorch3D version: {pytorch3d.__version__}")

    # Create random point clouds on the CUDA device
    p1 = torch.randn(1, 100, 3, device=device)
    p2 = torch.randn(1, 200, 3, device=device)

    # Synchronize before timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # This uses a custom CUDA kernel in PyTorch3D
    dists, idxs, nn = knn_points(p1, p2, K=3)

    torch.cuda.synchronize()
    t_pt3d = time.perf_counter() - t0

    print(f"  knn_points on GPU took: {t_pt3d:.4f} s")
    print("  PyTorch3D test PASSED.")

except ImportError:
    print("  PyTorch3D not found. Skipping test.")
except Exception as e:
    print(f"  PyTorch3D test FAILED: {e}")