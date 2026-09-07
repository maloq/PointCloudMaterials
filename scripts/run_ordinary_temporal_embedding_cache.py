#!/usr/bin/env python3
"""Entry point for run ordinary temporal embedding cache; see src.temporal_vamp.commands.run_ordinary_temporal_embedding_cache."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_vamp.commands.run_ordinary_temporal_embedding_cache import main

if __name__ == "__main__":
    raise SystemExit(main())
