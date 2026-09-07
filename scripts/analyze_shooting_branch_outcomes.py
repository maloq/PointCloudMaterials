#!/usr/bin/env python3
"""Entry point for analyze shooting branch outcomes; see src.temporal_vamp.commands.analyze_shooting_branch_outcomes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_vamp.commands.analyze_shooting_branch_outcomes import main

if __name__ == "__main__":
    raise SystemExit(main())
