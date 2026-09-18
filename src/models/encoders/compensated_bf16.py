"""Fused compensated BF16 GEMM, including deterministic split-K weight gradients.

Inputs and outputs are FP32. Three BF16 tensor-core products approximate the
FP32 product; conversions and residuals live inside the kernel, not temporary
global-memory tensors. This is the same high/low decomposition as the reference
implementation in structural_precision.py, not ordinary low-precision rounding.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _compensated_gemm(A, B, C, M, N, K, SAM, SAK, SBK, SBN,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      SPLIT: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    k = part * BK + tl.arange(0, BK)
    high = tl.zeros((BM, BN), tl.float32)
    low = tl.zeros((BM, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(A + m[:, None] * SAM + k[None, :] * SAK,
                    (m[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * SBK + n[None, :] * SBN,
                    (k[:, None] < K) & (n[None, :] < N), 0)
        ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
        al = (a - ah.to(tl.float32)).to(tl.bfloat16)
        bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
        high = tl.dot(ah, bh, high)
        low = tl.dot(ah, bl, low)
        low = tl.dot(al, bh, low)
        k += BK * SPLIT
    tl.store(C + part * M * N + m[:, None] * N + n[None, :], high + low,
             (m[:, None] < M) & (n[None, :] < N))


@triton_op('pcm::compensated_bf16_mm', mutates_args={})
def compensated_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError('Compensated GEMM requires FP32 inputs')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('Compensated GEMM requires compatible matrices')
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError('Compensated GEMM requires a single CUDA device')
    m, k = a.shape
    n = b.shape[1]
    split = 32 if k > 1024 and m <= 512 and n <= 512 else 1
    partial = torch.empty((split, m, n), device=a.device, dtype=torch.float32)
    wrap_triton(_compensated_gemm)[(triton.cdiv(m, 32), triton.cdiv(n, 64), split)](
        a, b, partial, m, n, k, *a.stride(), *b.stride(),
        BM=32, BN=64, BK=32, SPLIT=split, num_warps=4, num_stages=3)
    return partial[0] if split == 1 else partial.sum(0)
