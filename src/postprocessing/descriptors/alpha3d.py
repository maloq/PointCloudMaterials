"""Content-addressed native alpha persistence, compiled outside the repository."""
from functools import lru_cache
import fcntl
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sysconfig
import time

import numpy as np
import pybind11

from src.project_runtime.paths import machine, resolve_path, storage_path

GUDHI_REVISION = '84974f4ebb9e7c29cfc02b35edb8a9bd5ff4a55e'
DEPENDENCIES = {
    'cgal-cpp': ('5.6.1','hd557e15_2'),
    'libboost-headers': ('1.86.0','ha770c72_5'),
    'gmp': ('6.3.0','hfd2156b_3'),
    'mpfr': ('4.2.2','ha2cb11d_1'),
    'eigen': ('3.4.0','h54a6638_2'),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def extension():
    settings=machine()['execution'].get('rich_alpha')
    if not settings:
        raise RuntimeError('Configure execution.rich_alpha in machine.local.yaml; see docs/target_preparation.md')
    source=Path(__file__).with_suffix('.cpp')
    gudhi=resolve_path(settings['gudhi_source']);prefix=resolve_path(settings['dependencies'])
    revision=subprocess.check_output(['git','-C',str(gudhi),'rev-parse','HEAD'],text=True).strip()
    if revision!=GUDHI_REVISION:raise ValueError('Native alpha requires the pinned GUDHI 3.11.0 headers')
    packages={}
    for package,expected in DEPENDENCIES.items():
        matches=list((prefix/'conda-meta').glob(package+'-*.json'))
        if len(matches)!=1:raise ValueError(f'Missing/ambiguous native alpha dependency: {prefix}/{package}')
        metadata=json.loads(matches[0].read_text())
        if (metadata['version'],metadata['build'])!=expected:
            raise ValueError(f'Changed native alpha dependency {package}; install the pinned build {expected}')
        packages[matches[0].name]=_sha(matches[0])
    compiler=settings.get('cxx','g++')
    flags=['-O3','-DNDEBUG','-std=c++17','-shared','-fPIC','-frounding-math','-ffp-contract=off','-fopenmp','-march=native']
    cpu={}
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if not line:break
        key,_,value=line.partition(':')
        if key.strip() in ('vendor_id','cpu family','model','stepping','model name','flags'):
            cpu[key.strip()]=value.strip()
    binding=dict(source_sha256=_sha(source),loader_sha256=_sha(__file__),gudhi_revision=revision,
        dependency_packages=packages,dependency_root=str(prefix),gudhi_root=str(gudhi),
        compiler=compiler,compiler_version=subprocess.check_output([compiler,'--version'],text=True),
        flags=flags,python_abi=sysconfig.get_config_var('SOABI'),pybind11=pybind11.__version__,
        libc=platform.libc_ver(),architecture=platform.machine(),cpu_target=cpu)
    identity=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    root=storage_path('cache')/'compiled'/'rich-alpha'/identity;root.mkdir(parents=True,exist_ok=True)
    binary=root/('pcm_alpha3d'+sysconfig.get_config_var('EXT_SUFFIX'))
    with (root/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (root/'build.json').exists():
            includes=[sysconfig.get_paths()['include'],pybind11.get_include(),str(prefix/'include'),
                      str(prefix/'include/eigen3')]+[str(p) for p in sorted((gudhi/'src').glob('*/include'))]
            command=[compiler,*flags,str(source),'-o',str(binary)]+['-I'+p for p in includes]
            command+=['-L'+str(prefix/'lib'),'-Wl,-rpath,'+str(prefix/'lib'),'-lgmp','-lmpfr']
            started=time.monotonic()
            with (root/'build.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Native alpha build failed: {root / "build.log"}')
            (root/'build.json').write_text(json.dumps(dict(identity=identity,binding=binding,command=command,
                binary_sha256=_sha(binary),seconds=time.monotonic()-started),indent=2)+'\n')
        receipt=json.loads((root/'build.json').read_text())
        if receipt['identity']!=identity or receipt['binary_sha256']!=_sha(binary):
            raise ValueError(f'Changed native alpha build: {root}')
    spec=importlib.util.spec_from_file_location('pcm_alpha3d',binary)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.build_receipt=str(root/'build.json')
    return module


def topology(points, indices=None):
    coordinates=np.zeros((1,80,3),np.float64)
    coordinates[0,:len(points)]=points
    return extension().packets(coordinates,np.array([len(points)],np.int32),1,indices)[0]
