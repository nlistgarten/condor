from upcycle import upcycle_problem, sympy2casadi  # isort: skip

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

out_meta = prob.model._var_abs2meta["output"]
cons_meta = prob.driver._cons
obj_meta = prob.driver._objs
dv_meta = prob.driver._designvars

n = len(out_syms)
x0 = np.zeros(n)
lbx = np.full(n, -np.inf)
ubx = np.full(n, np.inf)
count = 0
for name, meta in out_meta.items():
    size = meta["size"]

    x0[count : count + size] = meta["val"].flatten()

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

    if name in obj_meta:
        iobj = count

    count += size

ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = sympy2casadi(res_mat, out_syms, ca_vars)

nlp = {"x": ca_vars, "f": ca_vars[iobj], "g": ca_res}
S = casadi.nlpsol("S", "ipopt", nlp)

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
pprint.pprint(out)
