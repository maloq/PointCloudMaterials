#!/usr/bin/env python3
"""Entry point for run predictability map; see src.temporal_vamp.commands.run_predictability_map."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_vamp.commands.run_predictability_map import main

if __name__ == "__main__":
    raise SystemExit(main())
