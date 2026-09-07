#!/usr/bin/env python3
"""Command entry point; implementation is in src.training_methods.spatiotemporal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_methods.spatiotemporal import main

if __name__ == "__main__":
    raise SystemExit(main())
