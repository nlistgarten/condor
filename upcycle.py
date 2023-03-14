import os

import casadi
import numpy as np
import sympy as sym
from openmdao.components.balance_comp import BalanceComp
from openmdao.components.meta_model_structured_comp import \
    MetaModelStructuredComp
from openmdao.core.problem import Problem
from openmdao.vectors.default_vector import DefaultVector
from pycycle.elements.US1976 import USatm1976Comp, USatm1976Data
from sympy.core.expr import Expr
from sympy.printing.pycode import PythonCodePrinter
from sympy.utilities.lambdify import lambdify

os.environ["OPENMDAO_REPORTS"] = "none"


class UpcycleCodePrinter(PythonCodePrinter):
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
        if isinstance(expr, UpcycleAppliedFunction):
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

from sympy.core.function import UndefinedFunction, AppliedUndef

class UpcycleAppliedFunction(AppliedUndef):
    ApplicationType = {}

    def __new__(cls, *args, **options):
        #breakpoint()
        newcls = super().__new__(cls, *args, **options)

        return newcls


class UpcycleUndefinedFunction(UndefinedFunction):
    def __new__(mcl, name, 
                bases=(UpcycleAppliedFunction,),
                __dict__=None, **kwargs):
        # called when an argument is applied, e.g., tab_h(...)
        # returns the called function expression as subclass of UpcycleAppliedFunction
        # applied tab_h is what codeprinter sees, without catching it gets processed as
        # a Function which would be  fine, since don't want to repeat the  print 
        # function and args logic, but Printer._print has some special Function filter
        # logic and the CodePrinter._print_Function uses string value of known_functions
        # dict or delegates if "allow_unknown_functions". I guess it would be reasonable
        # for _print_TableLookup to just add to known_functions dict?
        # also, the logic is just 
        # return "%s(%s)" % (func, self.stringify(expr.args, ", "))
        # so I could just do that in the UpcycleAppliedFunc print logic...


        newcls = super().__new__(mcl, name, bases, __dict__=__dict__, **kwargs)
        #newcls.__mro__ = (newcls.__mro__[0], mcl,) + newcls.__mro__[1:]
        return newcls

class TableLookup(UpcycleUndefinedFunction):
    pass

def sympify_problem(prob):
    """Set up and run `prob` with symbolic inputs

    `prob` should *not* have had ``final_setup`` called.
    """
    prob.setup(local_vector_class=SymbolicVector, derivatives=False)
    prob.final_setup()
    # run_apply_nonlinear gives better matching to om but makes 1.0*sym everywhere
    prob.model._apply_nonlinear()

    root_vecs = prob.model._get_root_vectors()

    out_syms = root_vecs["output"]["nonlinear"].syms

    res_exprs = [
        rexpr
        for rarray in prob.model._residuals.values()
        for rexpr in sym.flatten(rarray)
        if not isinstance(rexpr, sym.core.numbers.Zero)
    ]
    res_mat = sym.Matrix(res_exprs)

    return prob, res_mat, out_syms


def extract_problem_data(prob):
    out_meta = prob.model._var_abs2meta["output"]
    cons_meta = prob.driver._cons
    obj_meta = prob.driver._objs
    dv_meta = prob.driver._designvars

    n = len(prob.model._outputs)
    x0 = np.zeros(n)
    lbx = np.full(n, -np.inf)
    ubx = np.full(n, np.inf)
    iobj = None
    count = 0
    explicit_idxs = []
    non_indep_idxs = []
    constant_idxs = []
    for name, meta in out_meta.items():
        size = meta["size"]

        # x0[count : count + size] = meta["val"].flatten()
        x0[count : count + size] = prob.model._outputs[name].flatten()

        lb = meta["lower"]
        ub = meta["upper"]
        if lb is not None:
            lbx[count : count + size] = lb
        if ub is not None:
            ubx[count : count + size] = ub

        if name in cons_meta:
            lbx[count : count + size] = cons_meta[name]["lower"]
            ubx[count : count + size] = cons_meta[name]["upper"]

        # TODO make sure explicit indepvarcomp sets this tag
        if "openmdao:indep_var" in meta["tags"]:
            found = False
            for dv_name, dv_meta_loc in dv_meta.items():
                if name == dv_meta_loc["source"]:
                    found = True
                    lbx[count : count + size] = dv_meta_loc["lower"]
                    ubx[count : count + size] = dv_meta_loc["upper"]
            if not found:
                constant_idxs.append(count)

        else:
            non_indep_idxs += range(count, count + size)

        if "openmdao:allow_desvar" not in meta["tags"]:
            explicit_idxs += range(count, count + size)

        if name in obj_meta:
            iobj = count

        count += size

    return x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs, constant_idxs

class assignment_cse:
    def __init__(self, assignment_dict, return_assignments=True):
        self.assignment_dict = assignment_dict
        self.original_arg_to_dummy = {}
        self.dummified_assignment_list = {}
        for arg, expr in assignment_dict.items():
            # sorry Kenny
            new_arg = self.original_arg_to_dummy[arg] = sym.Dummy()
            self.dummified_assignment_list[new_arg] = expr.subs(
                self.original_arg_to_dummy
            )
        self.return_assignments = return_assignments

    def __call__(self, expr):
        rev_dummified_assignments = {
            v: k for k, v in self.dummified_assignment_list.items()
        }
        if hasattr(expr, 'subs'):
            expr_ = expr.subs(rev_dummified_assignments)
        else:
            expr_ = [
                expr__.subs(rev_dummified_assignments)
                for expr__ in expr
            ]
        if self.return_assignments:
            if hasattr(expr_, '__len__'):
                expr_ = list(expr_)
            else:
                expr_ = [expr_]
            expr_.extend(self.dummified_assignment_list.keys())
        return self.dummified_assignment_list.items(), expr_


def sympy2casadi(sympy_expr, sympy_vars, extra_assignments={}):
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
    printer = UpcycleCodePrinter(dict(
        fully_qualified_modules=False,
    ))
    print("lambdifying...")
    f = lambdify(
        sympy_vars,
        sympy_expr,
        modules=[
            interp_registry,
            mapping,
            casadi
        ],
        printer=printer,
        cse=assignment_cse(extra_assignments),
    )

    print("casadifying...")
    out = f(*ca_vars_split)

    return out, ca_vars, f


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


orig_add_balance = BalanceComp.add_balance


def add_balance(self, *args, **kwargs):
    kwargs["normalize"] = False
    # kwargs["use_mult"] = False
    orig_add_balance(self, *args, **kwargs)


BalanceComp.add_balance = add_balance


interp_registry = {}
orig_mm_compute = MetaModelStructuredComp.compute


def make_interp_wrapper(interp):
    def wrapper(*args):
        return interp(casadi.vertcat(*args[0]))

    return wrapper


def mmsc_compute(self, inputs, outputs):
    if isinstance(inputs, SymbolicVector):
        for output_name in outputs:
            name = f"{self.name}_{output_name}"

            if name in interp_registry:
                ca_interp_wrapped = interp_registry[name]
            else:
                ca_interp = casadi.interpolant(
                    name,
                    "linear",
                    self.inputs,
                    self.training_outputs[output_name].ravel(order="F"),
                )
                ca_interp_wrapped = make_interp_wrapper(ca_interp)

            interp_registry[name] = ca_interp_wrapped

            # TODO flatten inputs?
            f = TableLookup(name)(sym.Array(sym.flatten(inputs.values())))
            outputs[output_name] = f
    else:
        orig_mm_compute(self, inputs, outputs)


MetaModelStructuredComp.compute = mmsc_compute

orig_usatm1976_compute = USatm1976Comp.compute


def usatm1976_compute(self, inputs, outputs):
    if isinstance(inputs, SymbolicVector):
        for output_name in outputs:
            fname = f"{self.name}_{output_name}"
            attr_name = output_name[:-1]
            ca_interp = casadi.interpolant(
                fname,
                "linear",
                [USatm1976Data.alt],
                getattr(USatm1976Data, attr_name),
            )
            ca_interp_wrapped = make_interp_wrapper(ca_interp)
            interp_registry[fname] = ca_interp_wrapped
            f = TableLookup(fname)(sym.Array(sym.flatten(inputs.values())))
            outputs[output_name] = f
    else:
        orig_usatm1976_compute(self, inputs, outputs)


USatm1976Comp.compute = usatm1976_compute


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


class SymbolicVector(DefaultVector):
    """OpenMDAO Vector implementation backed by SymbolicArray"""

    def __getitem__(self, name):
        """Ensure accessing items always gives an array (as opposed to a scalar)"""
        return np.atleast_1d(super().__getitem__(name)).view(SymbolicArray)

    # override
    def _create_data(self):
        """Replace root Vector's ndarray allocation with SymbolicArray"""
        system = self._system()
        names = []
        # om uses this and relies on ordering when building views, should be ok
        for abs_name, meta in system._var_abs2meta[self._typ].items():
            sz = meta["size"]
            if sz == 1:
                names.append(abs_name)
            else:
                names.extend([f"{abs_name}[{i}]" for i in range(sz)])
        syms = np.array([sym.Symbol(name) for name in names]).view(SymbolicArray)
        self.syms = syms
        return syms

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        """Disable setting values"""
        pass
