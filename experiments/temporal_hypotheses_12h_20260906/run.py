"""Dated recipe for the descriptor-free 12-hour temporal campaign."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.training_methods.temporal_campaign import main

if __name__=='__main__':
    main()
