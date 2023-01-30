from upcycle import upcycle_problem, extract_problem_data, sympy2casadi  # isort: skip

import os
import pprint

import casadi
import numpy as np

from om_sellar_implicit import make_sellar_problem

# avoid some errors from reports expecting float data
os.environ["OPENMDAO_REPORTS"] = "none"

prob = make_sellar_problem()
prob.run_driver()
prob.list_problem_vars(print_arrays=True)

res_mat, out_syms = upcycle_problem(prob)
x0, lbx, ubx, iobj = extract_problem_data(prob)

ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = sympy2casadi(res_mat, out_syms, ca_vars)

nlp = {"x": ca_vars, "f": ca_vars[iobj], "g": ca_res}
S = casadi.nlpsol("S", "ipopt", nlp)

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
pprint.pprint(out)
