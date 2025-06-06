import importlib


def get_backend(module="condor.backends.casadi"):
    mod = importlib.import_module(module)
    return mod
