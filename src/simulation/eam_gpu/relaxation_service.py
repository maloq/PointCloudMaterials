"""One persistent CUDA subprocess per relaxation worker, with per-cell timeout."""
import contextlib
import multiprocessing
import os
from pathlib import Path
import traceback


def _serve(connection):
    from .relaxation_fast import execute
    connection.send(dict(state='ready'))
    while True:
        request = connection.recv()
        if request is None:
            break
        directory, settings = request
        os.chdir(directory)
        try:
            command = settings['lammps_command']
            expected = command[command.index('--implementation-sha256')+1]
            potential = command[command.index('--potential')+1]
            with open('stdout.log', 'w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                execute('in.lammps', 'log.lammps', potential, expected_sha256=expected)
            connection.send(dict(state='complete'))
        except Exception:
            connection.send(dict(state='failed', error=traceback.format_exc()))
    connection.close()


class RelaxationService:
    def __init__(self):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child,), daemon=True)
        self.process.start(); child.close()
        if not self.connection.poll(120):
            self.close(); raise TimeoutError('Ni CUDA service did not initialize within 120 seconds')
        if self.connection.recv() != dict(state='ready'):
            self.close(); raise RuntimeError('Ni CUDA service did not become ready')

    def __call__(self, directory, settings):
        if settings['lammps_command'][2] != 'src.simulation.eam_gpu.relaxation_fast':
            raise ValueError('Persistent service requires its declared optimized execution recipe')
        self.connection.send((str(Path(directory).resolve()), settings))
        if not self.connection.poll(settings['frame_timeout_seconds']):
            self.close(); raise TimeoutError(f'Ni relaxation timed out: {directory}')
        try:
            result = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(f'Ni CUDA subprocess exited during {directory}; see callback-failure.json') from error
        if result['state'] != 'complete':
            raise RuntimeError(result['error'])

    def close(self):
        if self.process.is_alive():
            self.process.terminate(); self.process.join(10)
            if self.process.is_alive(): self.process.kill(); self.process.join()
        else: self.process.join()
        self.connection.close()

    def __enter__(self): return self

    def __exit__(self, *_): self.close()
