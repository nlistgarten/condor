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



    def __init__(
        self, wrapper_funcs, implementation, jacobian_of=None, input_symbol=None,
        output_symbol=None, opts={}
    ):
        """
        wrapper_funcs -- list of callables to wrap, in order of ascending derivatives

        jacobian_of -- used internally to recursively create references of related
        callbacks, as needed.

        input_symbol - single MX representing all input
        output_symbol - single MX representing all output

        input/output symbol used to identify sparsity and other metadata by creating a
        placeholder casadi.Function. Not used if jacobian_of is provided, because
        jacobian operator is used on jacobian_of's placeholder Function.

        """
        casadi.Callback.__init__(self)

        self.input_symbol = input_symbol
        self.output_symbol = output_symbol

        if jacobian_of is None and input_symbol is not None and output_symbol is not None:
            self.placeholder_func = casadi.Function(
                f"{implementation.model.__name__}_placeholder",
                [self.input_symbol],
                [self.output_symbol],
                # self.input,
                # self.output,
                dict(
                    allow_free=True,
                ),
            )
        else:
            self.placeholder_func = jacobian_of.placeholder_func.jacobian()

        self.wrapper_func = wrapper_funcs[0]
        if len(wrapper_funcs) == 1:
            self.jacobian = None
        else:
            self.jacobian = CasadiFunctionCallback(
                wrapper_funcs = wrapper_funcs[1:],
                implementation = implementation,
                jacobian_of = self,
                opts=opts,
            )

        self.jacobian_of = jacobian_of
        self.implementation = implementation
        self.opts = opts

    def construct(self):
        if self.jacobian is not None:
            self.jacobian.construct()
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
                args[0],
                # this lets 
                #*wrap(self.implementation.model.input, args[0])
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
        return out,
        #breakpoint()
        return [casadi.vertcat(*flatten(out))] if self.get_n_out() == 1 else (out,)
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

def callables_to_operator(wrapper_funcs, *args, **kwargs):
    """ check if this is actually something that needs to get wrapped or if it's a
    native op... if latter, return directly; if former, create callback.

    this function ultimately contains the generic API, CasadiFunctionCallback can be
    casadi specific interface


    to think about: default casadi jacobian operator introduces additional inputs
    (and therefore, after second derivative also introduces additional output) for
    anti-derivative output. This is required for most "solver" type outputs.
    However, probably don't want to use casadi machinery for retaining that solver
    -- would be done externally. Is there ever a time we would want to use casadi
    machinery for passing in previous solution? if so, what would the API be for
    specifying?

    y = fun(x)
    jac = jac_fun(x,y)
    d_jac_x, d_jac_y = hess_fun(x, y, jac)

    if we don't use casadi's mechanisms, we are technically not functional
    but maybe when it gets wrapped by the callbacks, it effectively becomes
    functional because new instances are always created when needed to prevent race
    conditions, etc?

    similarly, how to specify alternate types/details of derivative, like
    forward/reverse mode, matrix-vector product form, etc.

    by definition, since we are consuming functions and cannot arbitrarily take
    derivatives (could look at how casadi does trajectory but even if useful for our
    trajectory, may not generalize to other external solvers) so need to implement
    and declare manually. Assuming this take is right, seems somewhat reasonable to
    assume we get functions, list of callables (at least 1) that evaluates 0th+
    functions (dense jacobians, hessians, etc). Then a totally different spec track for
    jacobian-vector or vector-jacobian products. Could cap out at 2nd derivative, not
    sure what the semantics for hessian-vector vs vector-hessian products, tensor etc.



    """
    if isinstance(wrapper_funcs[0], casadi.Function):
        return wrapper_funcs[0]
    return CasadiFunctionCallback(wrapper_funcs, *args, **kwargs)

def expression_to_operator(input_symbols, output_expressions, name="", **kwargs):
    """ take a symbolic expression and create an operator -- callable that can be used
    with jacobian, etc. operators """
    if not name:
        name = "function_from_expression"
    return casadi.Function(
        name,
        [input_symbols],
        [output_expressions],
    )

def jacobian(of, wrt):
    """ create a callable that computes dense jacobian """
    return casadi.jacobian(of, wrt)

def jac_prod(of, wrt, rev=True):
    """ create directional derivative """
    return casadi.jtimes(of, wrt, not rev)


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
