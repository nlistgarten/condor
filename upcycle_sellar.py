import upcycle  # isort: skip

import os
import pprint

import casadi
import numpy as np

from om_sellar_implicit import make_sellar_problem

# avoid some errors from reports expecting float data
os.environ["OPENMDAO_REPORTS"] = "none"

prob = make_sellar_problem()
# prob.setup()
# prob.final_setup()
# 
# pprint.pprint(list(prob.model._residuals.items()))
# pprint.pprint(list(prob.model._outputs.items()))
# pprint.pprint(list(prob.model._inputs.items()))
# 
# print(40*"-")
# prob.model._apply_nonlinear()
# 
# pprint.pprint(list(prob.model._residuals.items()))
# pprint.pprint(list(prob.model._outputs.items()))
# pprint.pprint(list(prob.model._inputs.items()))

prob.final_setup()
x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs = upcycle.extract_problem_data(prob)

prob.model._apply_nonlinear()
prob.model.list_outputs()

prob.run_driver()
prob.list_problem_vars(print_arrays=True)

sym_prob, res_mat, out_syms = upcycle.upcycle_problem(prob)


ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = upcycle.sympy2casadi(res_mat, out_syms, ca_vars)

nlp = {"x": ca_vars, "f": ca_vars[iobj], "g": ca_res}
S = casadi.nlpsol("S", "ipopt", nlp)

# update explicit output guesses
res = casadi.Function("res", casadi.vertsplit(ca_vars), casadi.vertsplit(ca_res))
xz = np.zeros_like(x0)
xz[non_indep_idxs] = np.array(res(*x0)).squeeze()
x0[explicit_idxs] += xz[explicit_idxs]

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
pprint.pprint(out)
