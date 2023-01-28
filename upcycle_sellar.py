import os

import numpy as np
import openmdao.api as om
import sympy as sym
import casadi
from sympy.utilities.lambdify import lambdify

from om_sellar_implicit import make_sellar_problem
from upcycle import SymbolicVector

# avoid some errors from reports expecting float data
os.environ["OPENMDAO_REPORTS"] = "none"

print("checking problem optimization")
print(40 * "=")
prob = make_sellar_problem()

# check the problem still works
prob.run_driver()

print("x = ", prob.model.get_val("x"))
print("z = ", prob.model.get_val("z"))
print("obj = ", prob.model.get_val("obj"))


prob.setup(local_vector_class=SymbolicVector)

prob.final_setup()


print("apply nonlinear")
print(40 * "=")

prob.model._apply_nonlinear()

print(*list(prob.model._residuals.items()), sep="\n")

print("eval jacobians")
print(40 * "=")
out_syms = prob.model._get_root_vectors()["output"]["nonlinear"].syms
out_mat = sym.Matrix(out_syms)

in_syms = prob.model._get_root_vectors()["input"]["nonlinear"].syms
in_mat = sym.Matrix(in_syms)

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
    for rexpr in sym.flatten(rarray):  # .flatten():
        if isinstance(rexpr, np.ndarray):
            print("array")
            # print(rexpr.shape, rexpr.size, rexpr)
            arrays += 1

        elif isinstance(rexpr, sym.core.numbers.Zero):
            print("zero")
            zeros += 1
        else:
            count += 1
            print(rexpr, "\n" * 3)
        # TODO check if zero/all zeros
        # TODO check if array
        # rexpr.diff(out_mat)
print(res_mat.shape[0] == count)

res_mat.jacobian(out_syms)

# end of upcycle
# ---------------


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
    }

    f = lambdify(sympy_var, sympy_expr, modules=[mapping, casadi])

    return f(*casadi_var), casadi_var


ca_vars = casadi.vertcat(*[casadi.MX.sym(lamda_arg.name) for lamda_arg in out_syms])
f, ca_vars_out = sympy2casadi(res_mat, out_syms, ca_vars)

# condor component construction (with casadi)

# condor optimization problem construction

nlp = {"x": ca_vars, "f": ca_vars[-3], "g": f}
S = casadi.nlpsol("S", "ipopt", nlp)

out = S(
    x0=[1, 5, 2, 1, 1, 3, 0, -10],
    lbg=0,
    ubg=0,
    lbx=[0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    ubx=[10, 10, 10, np.inf, np.inf, np.inf, 0, 0],
)
import pprint
pprint.pprint(out)
