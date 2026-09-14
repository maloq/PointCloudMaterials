"""Dated recipe for the descriptor-free 12-hour temporal campaign."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.research.temporal_hypotheses_12h.train import main

if __name__ == '__main__':
    main()
