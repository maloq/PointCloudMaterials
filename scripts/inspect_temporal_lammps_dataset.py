#!/usr/bin/env python3
"""Maintained entry point; implementation in src.data_utils.inspect_temporal_lammps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils.inspect_temporal_lammps import main
if __name__ == '__main__':
    raise SystemExit(main())
