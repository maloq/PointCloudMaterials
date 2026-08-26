#!/usr/bin/env python
"""Staged entry point for the frozen-encoder temporal VAMP experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_vamp.pipeline import main


if __name__ == "__main__":
    main()
