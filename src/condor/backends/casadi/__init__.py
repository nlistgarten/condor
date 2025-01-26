import casadi
import numpy as np
from dataclasses import dataclass, field

from condor.backends.element_mixin import BackendSymbolDataMixin
from condor.backends.casadi import utils
from condor.backends.casadi.utils import (flatten, recurse_if_else, substitute,
                                          symbol_class, wrap)
name = "Casadi"


from condor.backends.casadi import operators


def concat(arrs, axis=0):
    """ implement concat from array API for casadi """
    if axis == 0:
        return casadi.vcat(arrs)
    elif axis == 1:
        return casadi.hcat(arrs)
    else:
        raise ValueError("casadi only supports matrices")

def shape_to_nm(shape):
    if len(shape) > 2:
        raise ValueError
    n = shape[0]
    if len(shape) == 2:
        m = shape[1]
    else:
        m = 1
    return n, m

@dataclass
class BackendSymbolData(BackendSymbolDataMixin):
    def __post_init__(self, *args, **kwargs):
        # might potentially want to overwrite for casadi-specific validation, etc.
        super().__post_init__(self, *args, **kwargs)


    def flatten_value(self, value):
        if self.size == 1 and isinstance(value, (float, int)):
            return value
        else:
            if not isinstance(value, (np.ndarray, symbol_class)):
                value = np.array(value)
            return value.reshape((-1,1))


    def wrap_value(self, value):
        if self.size == 1:
            if isinstance(value, (float, int, symbol_class)):
                return value
            elif hasattr(value, "__iter__"):
                return np.array(value).reshape(-1)[0]

        if (
            (isinstance(value, symbol_class) and value.is_constant())
            #or isinstance(value, casadi.DM)
            or not isinstance(value, np.ndarray)
        ):
            value = np.array(value)

        try:
            value = value.reshape(self.shape)
        except ValueError:
            value = value.reshape(self.shape + (-1,))

        return value
        return symbol_class(value).reshape(self.shape)




def symbol_generator(name, shape=(1, 1), symmetric=False, diagonal=False):
    n, m = shape_to_nm(shape)
    sym = symbol_class.sym(name, (n, m))
    # sym = MixedMX.sym(name, (n, m))
    # TODO: symmetric and diagonal,

    if symmetric:
        raise NotImplemented
    if diagonal:
        raise NotImplemented

    if symmetric:
        assert n == m
        return casadi.tril2symm(casadi.tril(sym))
    if diagonal:
        assert m == 1
        sym = casadi.diag(sym)
    return sym


def get_symbol_data(symbol):
    if not isinstance(symbol, (symbol_class, casadi.DM)):
        symbol = np.atleast_1d(symbol)
        # I'm not sure why, but before I reshaped this to a vector always. Only
        # reshaping tensors now...
        # .reshape(-1)
        if symbol.ndim > 2:
            symbol.reshape(-1)
    shape = symbol.shape
    n, m = shape_to_nm(shape)
    size = n * m

    # TODO: actually check these?
    # and automatically reduce size if true? or should these be flags?
    diagonal = False
    symmetric = False

    return BackendSymbolData(
        shape=shape,
        symmetric=symmetric,
        diagonal=diagonal,
    )
