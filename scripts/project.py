#!/usr/bin/env python3
"""Portable machine settings, dataset catalogs, bundles and project snapshots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.project_runtime.cli import main

if __name__ == '__main__':
    main()
