import casadi
import numpy as np

import condor as co


# base symbol class is MX because only MX can go through callbacks (e.g., models that
# use AlgebraicSystem or ODESystem)
symbol_class = casadi.MX


vertcat = casadi.vertcat


def substitute(expr, subs):
    for key, val in subs.items():
        expr = casadi.substitute(expr, key, val)
    return expr


def recurse_if_else(conditions_actions):
    if len(conditions_actions) == 1:
        return conditions_actions[0][0]
    condition, action = conditions_actions[-1]
    remainder = recurse_if_else(conditions_actions[:-1])
    return casadi.if_else(condition, action, remainder)


def symbol_is(a, b):
    return (a.shape == b.shape) and (a == b).is_one()


def evalf(expr, backend_repr2value):
    if not isinstance(expr, list):
        expr = [expr]
    func = casadi.Function(
        "temp_func",
        list(backend_repr2value.keys()),
        expr,
    )
    return func(*backend_repr2value.values())


def reshape(backend_repr, shape):
    if isinstance(
        backend_repr,
        (
            list,
            tuple,
        ),
    ):
        return casadi.vertcat(*backend_repr).reshape(shape)
    elif hasattr(backend_repr, "reshape"):
        return backend_repr.reshape(shape)
    elif np.prod(shape) == 1:
        return backend_repr
    raise ValueError


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


class CasadiFunctionCallback(casadi.Callback):
    """Base class for wrapping a Function with a Callback"""

    def __new__(cls, *args, **kwargs):
        """ check if this is actually something that needs to get wrapped or if it's a
        native op...
        """
        obj = super().__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj


    def __init__(
        self, placeholder_func, wrapper_func, implementation, jacobian_of=None, opts={}
    ):
        casadi.Callback.__init__(self)
        self.wrapper_func = wrapper_func
        self.placeholder_func = placeholder_func
        self.jacobian = None
        self.jacobian_of = jacobian_of
        self.implementation = implementation
        self.opts = opts

    def construct(self):
        super().construct(self.placeholder_func.name(), self.opts)


    def init(self):
        pass

    def finalize(self):
        pass

    def get_n_in(self):
        return self.placeholder_func.n_in()

    def get_n_out(self):
        return self.placeholder_func.n_out()

    def eval(self, args):
        try:
            out = self.wrapper_func(
                *wrap(self.implementation.model.input, args[0])
                # *wrap(self.implementation.model.input, args)
            )
        except Exception as e:
            breakpoint()
            pass
        if self.jacobian_of:
            if hasattr(out, "shape") and out.shape == self.get_sparsity_out(0).shape:
                return (out,)
            jac_out = (
                np.concatenate(flatten(out))
                .reshape(self.get_sparsity_out(0).shape[::-1])
                .T
            )
            if self.jacobian_of.jacobian_of:
                # for hessian, need to provide dependence of jacobian on the original
                # function's output. is it fair to assume always 0?
                return (jac_out, np.zeros(self.get_sparsity_out(1).shape))

            return (jac_out,)
        return [casadi.vertcat(*flatten(out))] if self.get_n_out() == 1 else out
        return [out] if self.get_n_out() == 1 else out
        # return out,
        return (casadi.vertcat(*flatten(out)),)
        return [out] if self.get_n_out() == 1 else out

    def get_sparsity_in(self, i):
        if self.jacobian_of is None or i < self.jacobian_of.get_n_in():
            return self.placeholder_func.sparsity_in(i)
        elif i < self.jacobian_of.get_n_in() + self.jacobian_of.get_n_out():
            # nominal outputs are 0
            return casadi.Sparsity(
                *self.jacobian_of.get_sparsity_out(
                    i - self.jacobian_of.get_n_in()
                ).shape
            )
        else:
            raise ValueError

    def get_sparsity_out(self, i):
        #if self.jacobian_of is None or i < self.jacobian_of.get_n_out():
        if i < 1:
            return casadi.Sparsity.dense(*self.placeholder_func.sparsity_out(i).shape)
        #elif i < self.jacobian_of.get_n_in() + self.jacobian_of.get_n_out():
        else:
            return self.placeholder_func.sparsity_out(i)
            return casadi.Sparsity(
                *self.jacobian_of.get_sparsity_out(
                    i - self.jacobian_of.get_n_in()
                ).shape
            )

    def has_forward(self, nfwd):
        return False
        return True

    def has_reverse(self, nrev):
        return False
        return True

    def get_forward(self, nin, name, inames, onames, opts):
        breakpoint()

    def get_reverse(self, nout, name, inames, onames, opts):
        breakpoint()
        # return casadi.Function(

    def has_jacobian(self):
        return self.jacobian is not None

    def get_jacobian(self, name, inames, onames, opts):
        # breakpoint()
        return self.jacobian


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
