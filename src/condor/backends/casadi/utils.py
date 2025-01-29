import casadi
import casadi
import numpy as np

import condor as co
from condor.backends.casadi import symbol_class


# base symbol class is MX because only MX can go through callbacks (e.g., models that
# use AlgebraicSystem or ODESystem)
vertcat = casadi.vertcat

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


def flatten(symbols):
    # TODO: is flatten alwyas getting vert-catted?
    if isinstance(symbols, co.Field):
        # if it's a field, get all of the symbolic representations
        symbols = symbols.list_of("backend_repr")

    if isinstance(symbols, symbol_class):
        # if it's a single
        symbol = symbols
        return tuple(
            [
                elem
                # for symbol in symbols
                for col in casadi.horzsplit(symbol)
                for elem in casadi.vertsplit(col)
            ]
        )

    if isinstance(symbols, (float, int)):
        return [symbols]

    return [
        symbol.reshape((-1, 1))
        if hasattr(symbol, "reshape")
        else np.array(symbol).reshape(-1, 1)
        for symbol in symbols
    ]


def wrap(field, values):
    size_cum_sum = np.cumsum([0] + field.list_of("size"))
    if isinstance(values, symbol_class):
        values = casadi.vertsplit(values)

        return tuple(
            [
                values[start_idx]
                if symbol.size == 1
                else casadi.vcat(values[start_idx:end_idx]).reshape(symbol.shape)
                for start_idx, end_idx, symbol in zip(
                    size_cum_sum, size_cum_sum[1:], field
                )
            ]
        )

    if isinstance(
        values,
        (
            float,
            int,
            casadi.DM,
        ),
    ):  # or len(field) != len(values):
        if not isinstance(
            values,
            (
                float,
                int,
                casadi.DM,
            ),
        ) and len(field) != len(values):
            breakpoint()
        values = np.atleast_1d(values).reshape(-1)

    values = np.array(values)  # .reshape(-1)
    return tuple(
        [
            values[start_idx]  # .reshape((1,-1))[0,0]
            if symbol.size == 1
            else values[start_idx:end_idx].reshape(symbol.shape)
            if values[start_idx:end_idx].size == symbol.size
            else values[start_idx:end_idx].reshape(symbol.shape + (-1,))
            for start_idx, end_idx, symbol in zip(size_cum_sum, size_cum_sum[1:], field)
        ]
    )
