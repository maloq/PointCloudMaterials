"""Thin entry point for the fixed TDA readout protocol."""
import argparse
from .run import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
