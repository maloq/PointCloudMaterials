#!/usr/bin/env python3
"""Drive sustained high utilization on a CUDA GPU."""

import argparse
import signal
import time
from dataclasses import dataclass
from typing import List

import torch


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class WorkItem:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


STOP_REQUESTED = False


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    return f"{gib:.2f} GiB"


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _reserve_device_memory(device: torch.device, target_bytes: int) -> tuple[List[torch.Tensor], int]:
    buffers: List[torch.Tensor] = []
    allocated = torch.cuda.memory_allocated(device)
    bytes_needed = max(0, target_bytes - allocated)
    if bytes_needed == 0:
        return buffers, 0

    remaining = bytes_needed
    chunk = min(1 << 30, remaining)  # 1 GiB chunks first
    min_chunk = 8 << 20  # 8 MiB minimum

    while remaining > 0 and chunk >= min_chunk:
        this_chunk = min(chunk, remaining)
        try:
            buffers.append(torch.empty(this_chunk, dtype=torch.uint8, device=device))
            remaining -= this_chunk
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            chunk //= 2

    reserved = bytes_needed - remaining
    return buffers, reserved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Max out CUDA GPU utilization with repeated GEMMs")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Square matrix dimension (N for NxN)")
    parser.add_argument("--streams", type=int, default=4, help="Number of CUDA streams")
    parser.add_argument("--matmuls-per-step", type=int, default=4, help="Matmuls launched per stream per loop")
    parser.add_argument("--dtype", choices=tuple(DTYPE_MAP.keys()), default="fp16", help="Compute dtype")
    parser.add_argument("--memory-fraction", type=float, default=0.90, help="Target fraction of VRAM to hold")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means run until Ctrl+C")
    parser.add_argument("--log-interval", type=float, default=2.0, help="Seconds between status logs")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this system.")
    if args.matrix_size <= 0:
        raise SystemExit("--matrix-size must be > 0")
    if args.streams <= 0:
        raise SystemExit("--streams must be > 0")
    if args.matmuls_per_step <= 0:
        raise SystemExit("--matmuls-per-step must be > 0")
    if args.log_interval <= 0:
        raise SystemExit("--log-interval must be > 0")
    if not (0.0 <= args.memory_fraction <= 0.98):
        raise SystemExit("--memory-fraction must be in [0.0, 0.98]")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    dtype = DTYPE_MAP[args.dtype]

    print(f"GPU: {props.name}")
    print(f"Total VRAM: {_format_bytes(props.total_memory)}")
    print(
        f"Config: device={args.device}, dtype={args.dtype}, matrix_size={args.matrix_size}, "
        f"streams={args.streams}, matmuls_per_step={args.matmuls_per_step}"
    )

    streams = [torch.cuda.Stream(device=device) for _ in range(args.streams)]
    work_items: List[WorkItem] = []

    n = args.matrix_size
    for stream in streams:
        with torch.cuda.stream(stream):
            a = torch.randn((n, n), device=device, dtype=dtype)
            b = torch.randn((n, n), device=device, dtype=dtype)
            c = torch.empty((n, n), device=device, dtype=dtype)
        work_items.append(WorkItem(a=a, b=b, c=c))

    torch.cuda.synchronize(device)

    target_bytes = int(props.total_memory * args.memory_fraction)
    reserve_buffers, reserved_bytes = _reserve_device_memory(device, target_bytes)
    if reserve_buffers:
        print(f"Reserved additional VRAM: {_format_bytes(reserved_bytes)}")
    else:
        print("No extra VRAM reservation performed.")

    start = time.time()
    next_log = start + args.log_interval
    launches = 0

    print("Running load loop. Press Ctrl+C to stop.")
    while not STOP_REQUESTED:
        for idx, stream in enumerate(streams):
            item = work_items[idx]
            with torch.cuda.stream(stream):
                for _ in range(args.matmuls_per_step):
                    torch.matmul(item.a, item.b, out=item.c)
                    item.a, item.b, item.c = item.b, item.c, item.a
            work_items[idx] = item
            launches += args.matmuls_per_step

        now = time.time()
        if now >= next_log:
            torch.cuda.synchronize(device)
            elapsed = max(now - start, 1e-6)
            launches_per_sec = launches / elapsed
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            print(
                f"[{elapsed:7.1f}s] launches/s={launches_per_sec:,.1f}, "
                f"allocated={_format_bytes(allocated)}, reserved={_format_bytes(reserved)}"
            )
            next_log = now + args.log_interval

        if args.duration > 0 and (now - start) >= args.duration:
            break

    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    print(f"Stopped after {elapsed:.1f}s")

    # Keep a reference so reservation is not optimized away early.
    _ = reserve_buffers


if __name__ == "__main__":
    main()
