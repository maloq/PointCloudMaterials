"""Temporary forwarding entry point for already-submitted preparation jobs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.research.spatiotemporal.prepare_spatiotemporal_vicreg_views import main
if __name__ == '__main__':
    main()
