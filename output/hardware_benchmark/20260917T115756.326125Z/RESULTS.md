# Hardware benchmark results

Run: 20260917T115756.326125Z; status: **complete**.
Started: 2026-09-17T11:57:56.356427+00:00; finished: 2026-09-17T11:59:14.118763+00:00.
CPU: AMD EPYC 9124 16-Core Processor; 16 allowed logical CPUs. Host: `node53`.
Python: 3.12.13; NumPy: 1.26.4.
Storage target: `/home/infres/vmorozov/PointCloudMaterials`; exact file size: 1,073,741,760 bytes.
Filesystem: nfs on ssd.enst.fr:/data/ir800 mounted at /home/infres
CPU r1_t1: LAMMPS (22 Jul 2025 - Update 4); launcher: `[]`.
GPU: NVIDIA H100 NVL; PyTorch: 2.11.0+cu128; CUDA: 12.8; driver: 595.84.
GPU timing: FP32; matmul=highest; eager; AdamW; 30 updates/trial; warmup=10; host threads=4.
MACE software: mace-torch=0.3.16; e3nn=0.4.4; cuEquivariance-torch=0.10.0.

Rates use median trial duration. Ranges are observed min–max throughput, not confidence intervals.
GPU examples count original inputs, not paired views; updates include forward, backward and AdamW.

| Benchmark | Workload | Trials | Median throughput | Min–max throughput | ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
| Storage: generation + hash + write + fsync | 1.000000 GiB; float16; 80 points/cloud | 3 | 212.61 MiB/s | 210.88–221.01 MiB/s | — |
| Storage: write calls + fsync (diagnostic) | 1.000000 GiB; float16; 80 points/cloud | 3 | 1,831.09 MiB/s | 1,776.14–1,899.16 MiB/s | — |
| Storage: sequential read, eviction advised | 1.000000 GiB; float16; 80 points/cloud | 3 | 1,629.18 MiB/s | 1,497.23–1,648.72 MiB/s | — |
| Storage: sequential read, warm | 1.000000 GiB; float16; 80 points/cloud | 3 | 11,603.39 MiB/s | 10,941.13–11,673.89 MiB/s | — |
| Storage: random mmap, eviction advised | 1.000000 GiB; float16; 80 points/cloud; batch=512; 128 batches | 3 | 24,516.26 clouds/s | 24,204.57–25,728.82 clouds/s | — |
| Storage: random mmap, warm | 1.000000 GiB; float16; 80 points/cloud; batch=512; 128 batches | 3 | 1,462,101.38 clouds/s | 1,455,993.91–1,475,322.70 clouds/s | — |
| CPU: LAMMPS | 32,000 atoms; 1 MPI ranks x 1 threads; 500 steps/trial; warmup=100; lj/cut | 3 | 63.63 MD steps/s | 63.56–63.72 MD steps/s | — |
| GPU: PointNet | batch=128; 80 points; 2 views/example | 3 | 12,019.98 examples/s | 12,013.71–12,024.68 examples/s | 10.65 |
| GPU: MACE | batch=16; 80 points; 2 views/example; channels=64; e3nn | 3 | 618.85 examples/s | 618.32–618.87 examples/s | 25.85 |
| GPU: forecast transformer | batch=512; history/future=16/16; dim=256; width=256 | 3 | 90,977.44 examples/s | 90,973.11–91,184.96 examples/s | 5.63 |

Buffered eviction-advised reads are not guaranteed cold-device reads. Warm reads include caching.
The primary write result includes generation and hashing; write-call-only throughput is diagnostic.
Background load is not controlled. Host load averages at start (1/5/15 min): [3.7939453125, 3.791015625, 3.45751953125].

Full precision: `tables/summary.csv`. Raw trials and hardware/software details: `technical/results.json`.
Resolved settings: `technical/config.json`. Definitions and implementation hashes: `tables/METRICS.md` and `technical/metric-contract.json`.
