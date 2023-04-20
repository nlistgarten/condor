import casadi
from condor.backends.casadi import implementations
from condor.backends import BackendSymbolData
import numpy as np

name = 'Casadi'
symbol_class = casadi.MX

def shape_to_nm(shape):
    if len(shape) > 2:
        raise ValueError
    n = shape[0]
    if len(shape) == 2:
        m = shape[1]
    else:
        m = 1
    return n,m


def symbol_generator(name, shape=(1,1), symmetric=False, diagonal=False):
    n, m = shape_to_nm(shape)
    print("casadi creating",name, n, m, symmetric, diagonal)
    sym = casadi.MX.sym(name, (n, m))
    #sym = MixedMX.sym(name, (n, m))
    # TODO: symmetric and diagonal, 

    if symmetric:
        assert n == m
        return casadi.tril2symm(casadi.tril(sym))
    if diagonal:
        assert m == 1
        return casadi.diag(sym)
    return sym

def get_symbol_data(symbol):
    if not isinstance(symbol, symbol_class):
        symbol = np.atleast_1d(symbol).reshape(-1)
    shape = symbol.shape
    n, m = shape_to_nm(shape)
    size = n*m
    # TODO: actually check these
    diagonal = False
    symmetric = False
    return BackendSymbolData(
        shape=shape, symmetric=symmetric, diagonal=diagonal, size=size
    )

