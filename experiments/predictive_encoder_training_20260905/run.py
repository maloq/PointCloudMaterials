"""Run the task-supervised encoder comparison."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.training_methods.predictive_structure import main

if __name__=='__main__':
    main()
