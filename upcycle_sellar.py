import upcycle  # isort: skip

import os
import pprint
import re

import casadi
import numpy as np
import pandas as pd

from om_sellar_implicit import make_sellar_problem

os.environ["OPENMDAO_REPORTS"] = "none"

prob = make_sellar_problem()
prob.set_solver_print(-1)
up_prob = make_sellar_problem()


prob.final_setup()
x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs = upcycle.extract_problem_data(prob)

prob.model._apply_nonlinear()
om_res = prob.model._residuals.asarray().copy()

prob.run_driver()


sym_prob, res_mat, out_syms = upcycle.upcycle_problem(up_prob)

ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = upcycle.sympy2casadi(res_mat, out_syms, ca_vars)


nlp = {"x": ca_vars, "f": ca_vars[iobj], "g": ca_res}
S = casadi.nlpsol("S", "ipopt", nlp, {"ipopt": {"print_level": 5}, "print_time": False})

res = casadi.Function("res", casadi.vertsplit(ca_vars), casadi.vertsplit(ca_res))

df_res = pd.DataFrame(
    {
        "casadi": casadi.vertcat(*res(*x0)).toarray().squeeze(),
        "om": om_res[non_indep_idxs],
    },
    index=out_syms[non_indep_idxs],
)
print(df_res)

xz = np.zeros_like(x0)
xz[non_indep_idxs] = np.array(res(*x0)).squeeze()
x0[explicit_idxs] += xz[explicit_idxs]

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

df = pd.DataFrame(
    {
        "initial_value": x0,
        "upper_bound": ubx,
        "lower_bound": lbx,
        "solution_value": out["x"].toarray().squeeze(),
    },
    index=out_syms,
)

pattern = re.compile(r"(.*)\[(\d)+\]$")

om_vals = []
for idx, row in df.iterrows():
    name = idx.name
    try:
        val = prob.get_val(name)[0]
    except KeyError:
        match = pattern.match(name)
        base_name = match[1]
        i = int(match[2])
        val = prob.get_val(base_name)[i]

    om_vals.append(val)
df["om_vals"] = om_vals

print(df)
