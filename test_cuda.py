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
=======
#!/usr/bin/env python3
"""
PyTorch CUDA Test Script
Checks if CUDA is available and working properly with PyTorch
"""

import torch
import time

def check_cuda_availability():
    """Check basic CUDA availability"""
    print("=== CUDA Availability Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is not available!")
        return False
    
    return True

def test_cuda_computation():
    """Test basic CUDA computations"""
    print("\n=== CUDA Computation Test ===")
    
    # Test tensor creation and basic operations
    try:
        # Create tensors on GPU
        device = torch.device('cuda')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        print("✓ Successfully created tensors on GPU")
        
        # Test matrix multiplication
        start_time = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # Wait for GPU computation to complete
        gpu_time = time.time() - start_time
        
        print(f"✓ Matrix multiplication on GPU completed in {gpu_time:.4f} seconds")
        
        # Test moving tensors between CPU and GPU
        c_cpu = c.cpu()
        c_gpu = c_cpu.cuda()
        
        print("✓ Successfully moved tensors between CPU and GPU")
        
        return True
        
    except Exception as e:
        print(f"✗ CUDA computation failed: {e}")
        return False

def benchmark_cuda_vs_cpu():
    """Compare performance between CUDA and CPU"""
    print("\n=== Performance Benchmark ===")
    
    size = 2000
    iterations = 5
    
    # CPU benchmark
    print("Running CPU benchmark...")
    cpu_times = []
    for i in range(iterations):
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
    
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    print(f"Average CPU time: {avg_cpu_time:.4f} seconds")
    
    if torch.cuda.is_available():
        # GPU benchmark
        print("Running GPU benchmark...")
        gpu_times = []
        device = torch.device('cuda')
        
        for i in range(iterations):
            a_gpu = torch.randn(size, size, device=device)
            b_gpu = torch.randn(size, size, device=device)
            
            torch.cuda.synchronize()  # Ensure GPU is ready
            start_time = time.time()
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()  # Wait for completion
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)
        
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        print(f"Average GPU time: {avg_gpu_time:.4f} seconds")
        print(f"Speedup: {avg_cpu_time / avg_gpu_time:.2f}x")
    else:
        print("GPU benchmark skipped (CUDA not available)")

def test_neural_network():
    """Test a simple neural network on GPU"""
    print("\n=== Neural Network Test ===")
    
    if not torch.cuda.is_available():
        print("Skipping neural network test (CUDA not available)")
        return
    
    try:
        device = torch.device('cuda')
        
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        ).to(device)
        
        # Create dummy data
        x = torch.randn(32, 784, device=device)
        target = torch.randint(0, 10, (32,), device=device)
        
        # Forward pass
        output = model(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        print("✓ Neural network forward and backward pass successful on GPU")
        print(f"  Model parameters on GPU: {next(model.parameters()).device}")
        print(f"  Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")

def main():
    """Run all CUDA tests"""
    print("PyTorch CUDA Test Suite")
    print("=" * 50)
    
    # Check CUDA availability
    if not check_cuda_availability():
        print("\nCUDA is not available. Please check your installation.")
        return
    
    # Test basic CUDA computations
    if not test_cuda_computation():
        print("\nBasic CUDA computations failed!")
        return
    
    # Run performance benchmark
    benchmark_cuda_vs_cpu()
    
    # Test neural network
    test_neural_network()
    
    print("\n" + "=" * 50)
    print("All tests completed! CUDA is working properly.")

if __name__ == "__main__":
    main()
