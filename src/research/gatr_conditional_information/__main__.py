import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Information beyond radial structure in frozen GATr')
    p.add_argument('--config',required=True);p.add_argument('--stage',choices=('all','prepare','probe','report'),default='all')
    args=p.parse_args();config=json.loads(Path(args.config).read_text())
    from .data import require_hardware
    require_hardware(config)
    if args.stage in ('all','prepare'):
        from .data import prepare
        prepare(config)
    if args.stage in ('all','probe'):
        from .probe import run
        run(config)
    if args.stage in ('all','report'):
        from .report import report
        report(config)

if __name__=='__main__': main()
