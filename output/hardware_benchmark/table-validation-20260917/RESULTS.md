# Hardware benchmark results

Run: table-validation-20260917; status: **complete**.
Started: 2026-09-17T10:18:41.097283+00:00; finished: 2026-09-17T10:18:52.643669+00:00.
CPU: AMD EPYC 9124 16-Core Processor; 16 allowed logical CPUs. Host: `node53`.
Python: 3.12.13; NumPy: 1.26.4.
Storage target: `/home/infres/vmorozov/PointCloudMaterials`; exact file size: 8,589,600 bytes.
Filesystem: nfs on ssd.enst.fr:/data/ir800 mounted at /home/infres
CPU r1_t1: LAMMPS (22 Jul 2025 - Update 4); launcher: `[]`.
CPU r2_t1: LAMMPS (22 Jul 2025 - Update 4); launcher: `['mpiexec', '-n', '2']`.
GPU: NVIDIA H100 NVL; PyTorch: 2.11.0+cu128; CUDA: 12.8; driver: 595.84.
GPU timing: FP32; matmul=highest; eager; AdamW; 2 updates/trial; warmup=1; host threads=4.
MACE software: mace-torch=0.3.16; e3nn=0.4.4; cuEquivariance-torch=0.10.0.

**SMOKE RUN: functional validation only, not a hardware performance measurement.**

Rates use median trial duration. Ranges are observed min–max throughput, not confidence intervals.
GPU examples count original inputs, not paired views; updates include forward, backward and AdamW.

| Benchmark | Workload | Trials | Median throughput | Min–max throughput | ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
| Storage: generation + hash + write + fsync | 0.008000 GiB; float16; 80 points/cloud | 1 | 196.80 MiB/s | 196.80–196.80 MiB/s | — |
| Storage: write calls + fsync (diagnostic) | 0.008000 GiB; float16; 80 points/cloud | 1 | 1,514.18 MiB/s | 1,514.18–1,514.18 MiB/s | — |
| Storage: sequential read, eviction advised | 0.008000 GiB; float16; 80 points/cloud | 1 | 1,279.44 MiB/s | 1,279.44–1,279.44 MiB/s | — |
| Storage: sequential read, warm | 0.008000 GiB; float16; 80 points/cloud | 1 | 10,092.37 MiB/s | 10,092.37–10,092.37 MiB/s | — |
| Storage: random mmap, eviction advised | 0.008000 GiB; float16; 80 points/cloud; batch=4; 4 batches | 1 | 3,749.88 clouds/s | 3,749.88–3,749.88 clouds/s | — |
| Storage: random mmap, warm | 0.008000 GiB; float16; 80 points/cloud; batch=4; 4 batches | 1 | 601,525.50 clouds/s | 601,525.50–601,525.50 clouds/s | — |
| CPU: LAMMPS | 256 atoms; 1 MPI ranks x 1 threads; 3 steps/trial; warmup=2; lj/cut | 1 | 7,007.40 MD steps/s | 7,007.40–7,007.40 MD steps/s | — |
| CPU: LAMMPS | 256 atoms; 2 MPI ranks x 1 threads; 3 steps/trial; warmup=2; lj/cut | 1 | 12,982.24 MD steps/s | 12,982.24–12,982.24 MD steps/s | — |
| GPU: PointNet | batch=2; 80 points; 2 views/example | 1 | 167.70 examples/s | 167.70–167.70 examples/s | 11.93 |
| GPU: MACE | batch=2; 80 points; 2 views/example; channels=8; e3nn | 1 | 7.55 examples/s | 7.55–7.55 examples/s | 264.74 |
| GPU: forecast transformer | batch=2; history/future=3/2; dim=256; width=32 | 1 | 374.29 examples/s | 374.29–374.29 examples/s | 5.34 |

Buffered eviction-advised reads are not guaranteed cold-device reads. Warm reads include caching.
The primary write result includes generation and hashing; write-call-only throughput is diagnostic.
Background load is not controlled. Host load averages at start (1/5/15 min): [3.2177734375, 3.4228515625, 3.8076171875].

Full precision: `tables/summary.csv`. Raw trials and hardware/software details: `technical/results.json`.
Resolved settings: `technical/config.json`. Definitions and implementation hashes: `tables/METRICS.md` and `technical/metric-contract.json`.
