import casadi
import numpy as np
from dataclasses import dataclass, field

from condor.backends.element_mixin import BackendSymbolDataMixin


from condor.backends.casadi import operators

vertcat = casadi.vertcat
symbol_class = casadi.MX

def shape_to_nm(shape):
    if isinstance(shape, (int, np.int64)):
        shape = (shape,1)
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
        if self.symmetric:
            unique_values = symmetric_to_unique(
                value,
                symbolic=isinstance(value, symbol_class)
            )
            if isinstance(value, symbol_class):
                syms = casadi.symvar(value)
                if len(syms) == 1 and symbol_is(unique_to_symmetric(syms[0]), value):
                    return syms[0]
            return unique_values
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
                ret_val = np.array(value).reshape(-1)[0]
                if isinstance(ret_val, symbol_class):
                    return ret_val
                return np.array(value).reshape(-1)

        if self.symmetric:
            value = unique_to_symmetric(value, symbolic=isinstance(value, symbol_class))


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

def symmetric_to_unique(value, symbolic=True):
    n = value.shape[0]
    unique_shape = (int(n*(n+1)/2), 1)
    indices = np.tril_indices(n)
    if symbolic:
        unique_values = symbol_class(*unique_shape)
    else:
        unique_values = np.empty(unique_shape)
    for kk, (i,j) in enumerate(zip(*indices)):
        unique_values[kk] = value[i,j]
    return unique_values

def unique_to_symmetric(unique, symbolic=True):
    count = unique.shape[0]
    n = m = int((np.sqrt(1+8*count)-1)/2)
    if symbolic:
        # using an empty MX is unverifiable by casadi.is_equal, but concatenating the
        # iniddividual elements from a list works
        matrix_symbols = np.empty((n, m), dtype=np.object_)
    else:
        if unique.shape[1] > 1:
            use_shape = (n,m,unique.shape[1])
        else:
            use_shape = (n,m)
        matrix_symbols = np.empty(use_shape)
    indices = np.tril_indices(n)
    for kk, (i,j) in enumerate(zip(*indices)):
        matrix_symbols[i,j] = unique[kk]
        if i != j:
            matrix_symbols[j,i] = unique[kk]
    if symbolic:
        symbol_list = matrix_symbols.tolist()
        matrix_symbols = casadi.vcat([
            casadi.hcat(row) for row in symbol_list
        ])
    return matrix_symbols

def zeros(shape=(1,1)):
    return symbol_class(*shape)


def symbol_generator(name, shape=(1, 1), symmetric=False, diagonal=False):
    n, m = shape_to_nm(shape)
    if diagonal:
        #assert m == 1
        sym = casadi.diag(sym)
        raise NotImplemented
    elif symmetric:
        assert n == m
        unique_shape = (int(n*(n+1)/2), 1)
        unique_symbols = symbol_class.sym(name, unique_shape)
        matrix_symbols = unique_to_symmetric(unique_symbols)
        return matrix_symbols
    else:
        sym = symbol_class.sym(name, (n, m))
        return sym


def get_symbol_data(symbol, symmetric=None):
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
    if symmetric is None:
        # if unprovided, try to determine if symmetric
        if isinstance( symbol, (symbol_class, casadi.DM)):
            symmetric = symbol.sparsity().is_symmetric() and size > 1
        else:
            symmetric = np.isclose(0, symbol - symbol.T).all() and len(shape) == 2 and size > 1

    return BackendSymbolData(
        shape=shape,
        symmetric=symmetric,
        diagonal=diagonal,
    )


def symbol_is(a, b):
    return (a.shape == b.shape) and (
        (a == b).is_one() or
        casadi.is_equal(a, b, int(1E10))
    )


def evalf(expr, backend_repr2value):
    if not isinstance(expr, list):
        expr = [expr]
    func = casadi.Function(
        "temp_func",
        list(backend_repr2value.keys()),
        expr,
    )
    return func(*backend_repr2value.values())


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
            # using callables_to_operator SHOULD mean that casadi backend for one-layer
            # callable can re-enter native casadi -- infinite differentiable, etc.
            self.jacobian = callables_to_operator(
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
                #np.concatenate(flatten(out))
                out
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
    with jacobian, etc. operators 

    assume "MISO" -- but slightly misleading since output can be arbitrary size
    """
    if not name:
        name = "function_from_expression"
    return casadi.Function(
        name,
        input_symbols,
        [output_expressions],
    )
