"""Cache immutable screen configuration reads around the original frozen queue.

This orchestration release resumes the exact original scientific producer and
config, including optimizer/RNG identity. It does not modify frozen source.
Invoke this file with PCM_PROJECT_ROOT pointing at the original frozen code.
"""
import copy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def cached_copies(loader):
    pinned = lru_cache(maxsize=2)(loader)

    def read(path):
        # screen_config updates the returned task list. Never expose cached state.
        return copy.deepcopy(pinned(path))

    return read


def main():
    producer = Path(os.environ['PCM_PROJECT_ROOT']).resolve()
    manifest = json.loads((producer.parent/'code-files.json').read_text())
    entry = 'src/research/mace_epi/queue.py'
    if hashlib.sha256((producer/entry).read_bytes()).hexdigest() != manifest[entry]:
        raise ValueError('Original frozen scientific queue changed')
    sys.path.insert(0, str(producer))
    from src.research.mace_epi import queue
    if Path(queue.__file__).resolve() != producer/entry:
        raise ValueError('Scientific imports must use the original frozen producer')
    queue.load_config = cached_copies(queue.load_config)
    original_child = queue.child
    wrapper = str(Path(__file__).resolve())

    def child(command, log):
        if command[:2] != ['-m', 'src.research.mace_epi.queue']:
            raise ValueError(f'Unexpected frozen queue child command: {command}')
        return original_child([wrapper, *command[2:]], log)

    queue.child = child
    queue.main()


def run_cli():
    # Completed DataLoader subprocesses can hang in resource-tracker shutdown,
    # retaining their GPU and blocking the next stage after all artifacts exist.
    # Exit only after the synchronous queue stage returns; preserve failures and
    # the trainer's checkpoint/deadline status instead of reporting success.
    code = 0
    try:
        main()
    except SystemExit as exc:
        code = exc.code
    except BaseException:
        traceback.print_exc()
        code = 1
    if code is None:
        code = 0
    elif not isinstance(code, int):
        print(code, file=sys.stderr)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == '__main__':
    run_cli()
