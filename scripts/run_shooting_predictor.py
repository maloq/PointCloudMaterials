#!/usr/bin/env python
"""Train the shooting-ensemble predictive representation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_vamp.shooting_pipeline import main


if __name__ == "__main__":
    main()
