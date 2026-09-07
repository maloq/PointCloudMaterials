#!/usr/bin/env python3
"""Command entry point; implementation is in src.analysis.plots.plot_experiment_summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.plots.plot_experiment_summary import main

if __name__ == "__main__":
    raise SystemExit(main())
