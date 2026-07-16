"""CLI for restartable jumpy FFS homogeneous-crystallization rates."""

from __future__ import annotations

import argparse
from pathlib import Path

from .atomistic.jumpy_ffs_config import load_jumpy_ffs_config
from .atomistic.jumpy_ffs_runner import generate_jumpy_ffs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run weighted jumpy forward-flux sampling using an integer connected-PTM "
            "largest-cluster coordinate and repository-owned Langevin-NVT shots."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the exact manifest-bound journal. Without this flag, an existing "
            "output directory is an error."
        ),
    )
    args = parser.parse_args()
    config = load_jumpy_ffs_config(args.config)
    result = generate_jumpy_ffs(config, resume=args.resume)
    print(
        "jFFS rate: "
        f"{result.rate_per_A3_ps:.8e} A^-3 ps^-1 "
        f"({result.rate_per_m3_s:.8e} m^-3 s^-1); "
        f"P(A->B)={result.crossing_probability:.8e}, "
        f"basin flux={result.basin_flux_per_ps:.8e} ps^-1"
    )


if __name__ == "__main__":
    main()
