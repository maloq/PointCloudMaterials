#!/usr/bin/env python
"""Evaluate cached temporal embeddings and a fitted linear VAMP model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_vamp.pipeline import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "evaluate"])
    main()
