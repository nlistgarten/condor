import numpy as np
import sympy as sym
import casadi
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.pycode import PythonCodePrinter
from sympy.utilities.lambdify import lambdify
from sympy.core.expr import Expr

BACKEND_SYMBOL = sym.Symbol
class _null_name:
    pass


class _base_condor_descriptor:
    def __set_name__(self, owner, name):
        print("setting", name, "for", owner, "as", self)
        self._name = name
        self._owner = owner
        self._resolve_name = ".".join([self._owner.__class__.__name__, self._name])

    def __get__(self, obj, objtype=None):
        print("getting", self._resolve_name, "as", self,  "for", obj)
        pass
        return 

    def __set__(self, obj, value):
        raise AttributeError(self._resolve_name + " cannot be set as " + self.__class__)

class CodePrinter(PythonCodePrinter):
    def doprint(self, *args, **kwargs):
        return super().doprint(*args, **kwargs)

    def _print(self, expr, **kwargs):
        # _print_Function did a recursive call for handling the call syntax (args)
        # and it was being cast as an NDimArray which was nice. I guess re-create that
        # instead of going straight to the type of function? went ahead and did it

        # so far, we're really doing condor in-memory codeprinting... TableLookup
        # assumes registry is passed to lambdify modules to add to "namespace"
        # could be nice to have TableLookup class own the registry, then an
        # accessor that can be generically passed to lambdify or process other types of
        # code printing (e.g., one that could be serialized/doesn't need the reference
        # to the table generated during sympify_problem). Similar behavior may be needed
        # for the other Ops (optimizer/rootfinder, SGM)
        if isinstance(expr, AppliedFunction):
            cls = type(type(expr))
            funcprintmethodname = '_print_' + cls.__name__
            funcprintmethod = getattr(self, funcprintmethodname, None)
            if funcprintmethod is None:
               funcprintmethod = PythonCodePrinter._print

            return "%s(%s)" % (
                funcprintmethod(expr.func), self.stringify(expr.args, ", ")
            )
        else:
            return super()._print(expr, **kwargs)


    def _print_NDimArray(self, expr, **kwargs):
        return repr(expr)

    def _print_TableLookup(self, expr, **kwargs):
        return repr(expr)

    def _print_Solver(self, expr, **kwargs):
        return repr(expr)

    def _print_Piecewise(self, expr):
        eargs = expr.args

        def recurse(eargs):
            val, cond = eargs[0]
            if len(eargs) == 1:
                return str(val)
            else:
                rest = eargs[1:]
                return f"if_else({cond}, {val}, {recurse(rest)})"

        return recurse(eargs)

from sympy.core.function import (
    UndefinedFunction as symUndefined,
    AppliedUndef as symApplied
)

class AppliedFunction(symApplied):
    ApplicationType = {}

    def __new__(cls, *args, **options):
        newcls = super().__new__(cls, *args, **options)

        return newcls


class UndefinedFunction(symUndefined):
    def __new__(mcl, name, 
                bases=(AppliedFunction,),
                __dict__=None, **kwargs):
        # called when an argument is applied, e.g., tab_h(...)
        # returns the called function expression as subclass of condor.AppliedFunction
        # applied tab_h is what codeprinter sees, without catching it gets processed as
        # a Function which would be  fine, since don't want to repeat the  print 
        # function and args logic, but Printer._print has some special Function filter
        # logic and the CodePrinter._print_Function uses string value of known_functions
        # dict or delegates if "allow_unknown_functions". I guess it would be reasonable
        # for _print_TableLookup to just add to known_functions dict?
        # also, the logic is just 
        # return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        # so I could just do that in the condor.AppliedFunc print logic...


        newcls = super().__new__(mcl, name, bases, __dict__=__dict__, **kwargs)
        #newcls.__mro__ = (newcls.__mro__[0], mcl,) + newcls.__mro__[1:]
        return newcls

class TableLookup(UndefinedFunction):
    registry = {}


class Solver(UndefinedFunction):
    registry = {}

class assignment_cse:
    def __init__(self, assignment_dict, return_assignments=True):
        self.assignment_dict = assignment_dict
        self.return_assignments = return_assignments
        self.reverse_assignment_dict = {
            v: k for k, v in self.assignment_dict.items()
        }

    def __call__(self, expr):
        if hasattr(expr, 'subs'):
            expr_ = expr.subs(self.reverse_assignment_dict)
        else:
            expr_ = [
                expr__.subs(self.reverse_assignment_dict)
                for expr__ in expr
            ]
        if self.return_assignments:
            if hasattr(expr_, '__len__'):
                expr_ = list(expr)
            else:
                expr_ = [expr]
            expr.extend(sym.flatten(self.assignment_dict.keys()))
        return self.assignment_dict.items(), expr




def sympy2casadi(
        sympy_expr,
        sympy_vars, # really, arguments/signature
        #cse_generator=assignment_cse,
        #cse_generator_args=dict(
            extra_assignments={}, return_assignments=True
        #)
):
    ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in sympy_vars])

    ca_vars_split = casadi.vertsplit(ca_vars)

    mapping = {
        "ImmutableDenseMatrix": casadi.blockcat,
        "MutableDenseMatrix": casadi.blockcat,
        "Abs": casadi.fabs,
        "abs": casadi.fabs,
        "Array": casadi.MX,
    }
    # can set allow_unknown_functions=False and pass user_functions dict of
    # {func_expr: func_str} instead of allow_unknown_function but  it's the
    # presence in "modules" that adds them to namespace on `exec` line of lambdify so
    # it's a little redundant. sticking with this version for now
    # user_functions gets added to known_functions
    printer = CodePrinter(dict(
        fully_qualified_modules=False,
    ))
    print("lambdifying...")
    f = lambdify(
        sympy_vars,
        sympy_expr,
        modules=[
            TableLookup.registry,
            Solver.registry,
            mapping,
            casadi
        ],
        printer=printer,
        dummify=False,
        #cse=cse_generator(**cse_generator_args),
        cse=(
            assignment_cse(extra_assignments, return_assignments)
        ),
    )

    print("casadifying...")
    out = f(*ca_vars_split)

    return out, ca_vars, f


class CasadiFunctionCallback(casadi.Callback):
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


def array_ufunc(self, ufunc, method, *inputs, out=None, **kwargs):
    sym_ufunc = getattr(sym, ufunc.__name__, None)
    if sym_ufunc is not None:
        return sym_ufunc(inputs[0])
    else:
        other = inputs[1]
        prefix = ""
        if inputs[1] is self:
            other = inputs[0]
            prefix = "r"

        ufunc_map = dict(
            subtract="sub",
            add="add",
            true_divide="truediv",
            divide="truediv",
            multiply="mul",
        )

        name = ufunc_map[ufunc.__name__]
        op = getattr(self, f"__{prefix}{name}__")

        if isinstance(other, np.ndarray):
            return np.array([op(x) for x in other]).view(other.__class__)
        else:
            return op(other)


Expr.__array_ufunc__ = array_ufunc

class SymbolicArray(np.ndarray):
    """ndarray containing sympy symbols, compatible with numpy methods like np.exp"""

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Replace calls like np.exp with sympy equivalent"""
        sym_ufunc = getattr(sym, ufunc.__name__, None)
        if sym_ufunc is not None:
            result = np.array([sym_ufunc(inp) for inp in inputs[0]]).view(SymbolicArray)
        else:
            # convert inputs that are type SymbolicArray to ndarray
            args = []
            for inp in inputs:
                args.append(
                    inp if not isinstance(inp, SymbolicArray) else inp.view(np.ndarray)
                )

            outputs = out
            out_no = []
            if outputs:
                out_args = []
                for j, output in enumerate(outputs):
                    if isinstance(output, SymbolicArray):
                        out_no.append(j)
                        out_args.append(output.view(np.ndarray))
                    else:
                        out_args.append(output)
                kwargs["out"] = tuple(out_args)
            else:
                outputs = (None,) * ufunc.nout

            result = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            if result is NotImplemented:
                return NotImplemented

        if not isinstance(result, np.ndarray):
            result = np.atleast_1d(result)

        return result.view(SymbolicArray)

def construct_explicit_matrix(name, n, m=1, symmetric=False, diagonal=0,
                              dynamic=False, **kwass):
    """
    construct a matrix of symbolic elements
    Parameters
    ----------
    name : string
        Base name for variables; each variable is name_ij, which
        admitedly only works clearly for n,m < 10
    n : int
        Number of rows
    m : int
        Number of columns
    symmetric : bool, optional
        Use to enforce a symmetric matrix (repeat symbols above/below diagonal)
    diagonal : bool, optional
        Zeros out off diagonals. Takes precedence over symmetry.
    dynamic : bool, optional
        Whether to use sympy.physics.mechanics dynamicsymbol. If False, use
        sp.symbols
    kwargs : dict
        remaining kwargs passed to symbol function
    Returns
    -------
    matrix : sympy Matrix
        The Matrix containing explicit symbolic elements
    """
    if dynamic:
        symbol_func = vector.dynamicsymbols
    else:
        symbol_func = sym.symbols

    if n != m and (diagonal or symmetric):
        raise ValueError("Cannot make symmetric or diagonal if n != m")

    if diagonal:
        return sp.diag(
            *[symbol_func(
                name+'_{}{}'.format(i+1, i+1), **kwass) for i in range(m)])
    else:
        matrix = sym.Matrix([
            [symbol_func(name+'_{}_{}'.format(j+1, i+1), **kwass)
             for i in range(m)] for j in range(n)
        ])

        if symmetric:
            for i in range(1, m):
                for j in range(i):
                    matrix[j, i] = matrix[i, j]

        return matrix


class _condor_symbol_generator(_base_condor_descriptor):
    def __init__(self, func=construct_explicit_matrix, **fixed_kwargs):
        print('init', self)
        self.func = func
        self.fixed_kwargs = fixed_kwargs
        self.count = 0
        self.children = []

    def __call__(self, n, **kwargs):
        print("calling for", self)
        pass_kwargs = dict(
            name="%s%d_" % (self._resolve_name, self.count),
            n=n,
            **self.fixed_kwargs,
            **kwargs
        )
        return self.func(**pass_kwargs)



class _condor_computation(_base_condor_descriptor):
    def __init__(self, matched_to=None):
        print('init', self)
        self._matched_to = matched_to
        pass

    def __setitem__(self, key, value):
        print("setting item for", self, ":", key, "=", value)
        pass

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print("setting item for", self, ".", name, "=", value)
        pass


# not really necessary, place holder in case it is
class CondorDict(dict):
    def __getitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        #breakpoint()
        return super().__setitem__(*args, **kwargs)

from enum import _is_dunder

class CondorModelType(type):
    """
    Metaclass for Condor  model
    """

    @classmethod
    def __prepare__(cls, name, bases, **kwds):
        print("preparing for", name)
        sup_dict = super().__prepare__(cls, name, bases, **kwds)
        cls_dict = CondorDict()

        dicts = [sup_dict] + [base.__dict__ for base in bases]
        for _dict in dicts:
            cls_dict.update({
                k: v for k, v in  _dict.items()
                if (isinstance(v, _base_condor_descriptor) or (_dict is sup_dict))
            })
        return cls_dict

    def __new__(cls, name, bases, attrs, **kwargs):
        print("__new__ for", name)
        new_cls = super().__new__(cls, name, bases, attrs, **kwargs)
        return new_cls

"""
do parent systems always write subsystems by getting all their output? I think so
Groups definitely need some type of naming scheme because all outputs get lifted up as
outputs (not disciplined/strict system encapsolation). Although my implementation of a
blockdiagram also does this by default?

get_val is nice for masking only what's needed, I guess we could use masks
what about multiple calls where some values don't change? need to reference that

in general symbolic and numeric references,

yeah, might need something
    fuel_flow_rate, thrust = propulsion(flight_condition, throttle)[
        propulsion.output.fuel_flow_rate,
        propulsion.output.thrust
    ]

which is ~ how the output df indexing should work? calls should genreally return the df,
indexable by number, symbolic object, etc.


"""
