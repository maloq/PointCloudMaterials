"""Detached completion queue for the retained benchmark training sweep."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    root=Path(__file__).resolve().parents[2]
    config=Path(__file__).parent/'config.json';out=root/json.loads(config.read_text())['output']
    status=dict(state='waiting_for_training',pid=os.getpid())
    def save(): (out/'postprocess_status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        while True:
            training=json.loads((out/'training_status.json').read_text())
            if training['state']=='failed':raise RuntimeError(f'Training failed: {training}')
            if training['state']=='complete':break
            time.sleep(5)
        stages=[('choose_checkpoints.py',[]),('evaluate.py',['--stage','all']),
                ('robustness.py',['--stage','evaluate']),('temporal.py',['--stage','evaluate'])]
        for script,args in stages:
            status.update(state='running',stage=script);save()
            with (out/(Path(script).stem+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).parent/script),'--config',str(config),*args],
                    cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(state='complete');save()
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
