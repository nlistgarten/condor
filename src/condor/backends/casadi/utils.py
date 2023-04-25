import casadi
import condor as co
import numpy as np

class CasadiFunctionCallbackMixin:
    """Base class for wrapping a Function with a Callback"""

    def __init__(self, func, opts={}):
        casadi.Callback.__init__(self)
        self.func = func
        self.construct(func.name(), opts)

    def init(self):
        pass

    def finalize(self):
        pass

    def get_n_in(self):
        return self.func.n_in()

    def get_n_out(self):
        return self.func.n_out()

    def eval(self, args):
        out = self.func(*args)
        return [out] if self.func.n_out() == 1 else out

    def get_sparsity_in(self, i):
        return self.func.sparsity_in(i)

    def get_sparsity_out(self, i):
        return self.func.sparsity_out(i)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        return self.func.jacobian()

def flatten(symbols, complete=True):
    if isinstance(symbols, co.Field):
        symbols = symbols.list_of("backend_repr")
    if isinstance(symbols, casadi.MX):
        symbol = symbols
        return tuple([
            elem
            #for symbol in symbols
            for col in casadi.horzsplit(symbol)
            for elem in casadi.vertsplit(col)
        ])

    if not symbols:
        return symbols

    if isinstance(symbols, float):
        return [symbols]

    if isinstance(symbols[0], casadi.MX):
        return [symbol.reshape((-1,1)) for symbol in symbols]
    else:
        # numeric only?
        if complete:
            return [
                elem
                for symbol in symbols
                for elem in np.atleast_1d(symbol).reshape(-1)
            ]
        return [ np.atleast_1d(symbol).reshape(-1)  for symbol in symbols]

def wrap(field, values):
    if isinstance(values, casadi.MX):
        return [values]
    size_cum_sum = np.cumsum([0] + field.list_of('size'))
    # TODO: need to make sure 

    if isinstance(values, (float, int, casadi.DM,)) or len(field) != len(values):
        values = np.atleast_1d(values).reshape(-1)
    return tuple([
        values[start_idx]
        if symbol.size == 1
        else values[start_idx:end_idx].reshape(symbol.shape)
        for start_idx, end_idx, symbol in zip(size_cum_sum, size_cum_sum[1:], field)
    ])

