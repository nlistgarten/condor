import upcycle  # isort: skip
import casadi
import sympy as sym
import numpy as np
from sympy.printing.pycode import PythonCodePrinter


class CustomPrinter(PythonCodePrinter):
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



# sym_prob, res_mat, out_syms = upcycle.sympify_problem(up_prob)
x1, x2 = sympy_var = sym.symbols("x1:3")
sympy_expr = sym.Matrix(
    [
        sym.sqrt(sym.Abs(x1 * x2)),
        # sym.Piecewise((x1, x1 > 1), (x2, x1 < -1), (x1 + x2, True)),
    ],
)

casadi_var_in = casadi.vertcat(*[casadi.MX.sym(s.name) for s in sympy_var])
# ca_res, ca_vars_out = upcycle.sympy2casadi(exprs, syms, ca_vars)

if casadi_var_in.shape[1] > 1:
    casadi_var_in = casadi_var_in.T
casadi_var = casadi.vertsplit(casadi_var_in)

mapping = {
    "ImmutableDenseMatrix": casadi.blockcat,
    "MutableDenseMatrix": casadi.blockcat,
    "Abs": casadi.fabs,
    # "abs": casadi.fabs,
    "Array": casadi.MX,
}

f = sym.lambdify(
    sympy_var,
    sympy_expr,
    modules=[mapping, casadi],
    # printer=CustomPrinter(
    #     {
    #         "fully_qualified_modules": False,
    #     }
    # ),
)

out = f(*casadi_var)
cf = casadi.Function("f", casadi.vertsplit(casadi_var_in), casadi.vertsplit(out))
