#!/usr/bin/env python3
"""Temporary launcher for already-submitted September 2026 source jobs.

Keep until both independent-source campaigns and their controller chains finish.
New submissions should use the maintained campaign command/experiment recipe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.simulation.campaigns.independent_meam_source import main

if __name__ == "__main__":
    raise SystemExit(main())
