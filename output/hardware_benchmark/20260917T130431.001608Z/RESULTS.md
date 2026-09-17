# Hardware benchmark results

Run: 20260917T130431.001608Z; status: **complete**.
Started: 2026-09-17T13:04:31.075401+00:00; finished: 2026-09-17T13:05:51.808637+00:00.
CPU: AMD EPYC 9275F 24-Core Processor; 12 allowed logical CPUs. Host: `node58`.
Python: 3.12.13; NumPy: 1.26.4.
Storage target: `/home/infres/vmorozov/PointCloudMaterials`; exact file size: 1,073,741,760 bytes.
Filesystem: nfs on ssd.enst.fr:/data/ir800 mounted at /home/infres
CPU r1_t1: LAMMPS (22 Jul 2025 - Update 4); launcher: `[]`.
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch: 2.11.0+cu128; CUDA: 12.8; driver: 615.71.09.
GPU timing: FP32; matmul=highest; eager; AdamW; 30 updates/trial; warmup=10; host threads=4.
MACE software: mace-torch=0.3.16; e3nn=0.4.4; cuEquivariance-torch=0.10.0.

Rates use median trial duration. Ranges are observed min–max throughput, not confidence intervals.
GPU examples count original inputs, not paired views; updates include forward, backward and AdamW.

| Benchmark | Workload | Trials | Median throughput | Min–max throughput | ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
| Storage: generation + hash + write + fsync | 1.000000 GiB; float16; 80 points/cloud | 3 | 293.97 MiB/s | 290.45–297.80 MiB/s | — |
| Storage: write calls + fsync (diagnostic) | 1.000000 GiB; float16; 80 points/cloud | 3 | 1,085.29 MiB/s | 1,083.94–1,085.80 MiB/s | — |
| Storage: sequential read, eviction advised | 1.000000 GiB; float16; 80 points/cloud | 3 | 1,079.69 MiB/s | 1,069.60–1,109.37 MiB/s | — |
| Storage: sequential read, warm | 1.000000 GiB; float16; 80 points/cloud | 3 | 41,727.39 MiB/s | 41,306.16–43,097.45 MiB/s | — |
| Storage: random mmap, eviction advised | 1.000000 GiB; float16; 80 points/cloud; batch=512; 128 batches | 3 | 27,138.29 clouds/s | 26,391.50–27,547.71 clouds/s | — |
| Storage: random mmap, warm | 1.000000 GiB; float16; 80 points/cloud; batch=512; 128 batches | 3 | 2,352,790.06 clouds/s | 2,345,173.33–2,357,025.57 clouds/s | — |
| CPU: LAMMPS | 32,000 atoms; 1 MPI ranks x 1 threads; 500 steps/trial; warmup=100; lj/cut | 3 | 99.97 MD steps/s | 99.85–100.58 MD steps/s | — |
| GPU: PointNet | batch=128; 80 points; 2 views/example | 3 | 21,187.75 examples/s | 21,132.87–21,188.64 examples/s | 6.04 |
| GPU: MACE | batch=16; 80 points; 2 views/example; channels=64; e3nn | 3 | 641.44 examples/s | 641.40–641.47 examples/s | 24.94 |
| GPU: forecast transformer | batch=512; history/future=16/16; dim=256; width=256 | 3 | 92,969.76 examples/s | 92,968.89–93,008.52 examples/s | 5.51 |

Buffered eviction-advised reads are not guaranteed cold-device reads. Warm reads include caching.
The primary write result includes generation and hashing; write-call-only throughput is diagnostic.
Background load is not controlled. Host load averages at start (1/5/15 min): [1.70263671875, 6.1845703125, 8.3720703125].

Full precision: `tables/summary.csv`. Raw trials and hardware/software details: `technical/results.json`.
Resolved settings: `technical/config.json`. Definitions and implementation hashes: `tables/METRICS.md` and `technical/metric-contract.json`.
