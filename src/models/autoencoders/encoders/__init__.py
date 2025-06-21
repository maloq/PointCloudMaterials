import importlib, pkgutil

for _m in pkgutil.walk_packages(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")