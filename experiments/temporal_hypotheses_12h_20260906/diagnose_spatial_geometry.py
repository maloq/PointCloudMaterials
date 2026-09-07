"""Compare the user's older static GeoFrame field with predictive density."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.analysis.static_spatial_diagnosis import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(json.loads(parser.parse_args().config.read_text()))
