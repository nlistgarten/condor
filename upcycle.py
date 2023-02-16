import casadi
import numpy as np
import sympy as sym
from openmdao.components.balance_comp import BalanceComp
from openmdao.components.meta_model_structured_comp import \
    MetaModelStructuredComp
from openmdao.core.problem import Problem
from openmdao.vectors.default_vector import DefaultVector
from sympy.core.expr import Expr
from sympy.utilities.lambdify import lambdify


def upcycle_problem(num_prob):
    # run with a symbolic vector to get symbolic residual expressions
    prob = Problem()
    prob.model = num_prob.model
    prob.setup(local_vector_class=SymbolicVector, derivatives=False)
    prob.final_setup()
    prob.model._apply_nonlinear()

    # compute dR/do
    root_vecs = prob.model._get_root_vectors()
    out_syms = root_vecs["output"]["nonlinear"].syms

    out_meta = prob.model._var_abs2meta["output"]

    res_exprs = [
        rexpr
        for rarray in prob.model._residuals.values()
        for rexpr in sym.flatten(rarray)
        if not isinstance(rexpr, sym.core.numbers.Zero)
    ]
    res_mat = sym.Matrix(res_exprs)
    res_mat.shape

    count = 0
    arrays = 0
    zeros = 0
    for rname, rarray in prob.model._residuals.items():
        for rexpr in sym.flatten(rarray):
            if isinstance(rexpr, np.ndarray):
                arrays += 1
            elif isinstance(rexpr, sym.core.numbers.Zero):
                meta = out_meta[rname]
                if "openmdao:indep_var" not in meta["tags"]:
                    raise ValueError(f"Output has no dependence on inputs: {rname}")
                zeros += 1
            else:
                count += 1
            # TODO check if zero/all zeros
            # TODO check if array

    assert res_mat.shape[0] == count
    assert arrays == 0

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
            for dv_name, dv_meta_loc in dv_meta.items():
                if name == dv_meta_loc["source"]:
                    lbx[count : count + size] = dv_meta_loc["lower"]
                    ubx[count : count + size] = dv_meta_loc["upper"]
        else:
            non_indep_idxs += range(count, count + size)

        if "openmdao:allow_desvar" not in meta["tags"]:
            # breakpoint()
            explicit_idxs += range(count, count + size)

        if name in obj_meta:
            iobj = count

        count += size

    return x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs


def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    # TODO more/better defensive checks
    assert casadi_var.is_vector()
    if casadi_var.shape[1] > 1:
        casadi_var = casadi_var.T
    casadi_var = casadi.vertsplit(casadi_var)

    mapping = {
        "ImmutableDenseMatrix": casadi.blockcat,
        "MutableDenseMatrix": casadi.blockcat,
        "Abs": casadi.fabs,
        "Array": casadi.MX,
    }

    f = lambdify(sympy_var, sympy_expr, modules=[interp_registry, mapping, casadi])

    return f(*casadi_var), casadi_var


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
    kwargs["use_mult"] = False
    orig_add_balance(self, *args, **kwargs)


BalanceComp.add_balance = add_balance


orig_mm_compute = MetaModelStructuredComp.compute

interp_registry = {}


def mmsc_compute(self, inputs, outputs):
    global interp_registry

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
                # ca_interp_wrapped = ca_interp #lambda *x: ca_interp(casadi.vertcat(x))
                def ca_interp_wrapped(*args):
                    return ca_interp(casadi.vertcat(*args[0]))

            interp_registry[name] = ca_interp_wrapped

            # breakpoint()
            # TODO flatten inputs?
            f = sym.Function(name)(sym.Array(sym.flatten(inputs.values())))
            # f._imp_ = ca_interp_wrapped
            outputs[output_name] = f
    else:
        orig_mm_compute(self, inputs, outputs)


MetaModelStructuredComp.compute = mmsc_compute


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
                names.extend([f"{abs_name}_{i}" for i in range(sz)])
        syms = np.array([sym.Symbol(name) for name in names]).view(SymbolicArray)
        self.syms = syms
        return syms

    def set_var(self, name, val, idxs=None, flat=False, var_name=None):
        """Disable setting values"""
        pass


if __name__ == "__main__":
    xnd = np.array(sym.symbols("x:3"))
    xs = xnd.view(SymbolicArray)
    print(xs[0])
