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
x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs, const_idxs = upcycle.extract_problem_data(prob)

prob.model._apply_nonlinear()
om_res = prob.model._residuals.asarray().copy()

prob.run_driver()


sym_prob, res_mat, out_syms = upcycle.sympify_problem(up_prob)

datfp = "sellar.dat"
load = os.path.exists(datfp)
if False:
    print("loading casadi data")
    ser = casadi.FileDeserializer(datfp)
    res = ser.unpack()
    ca_vars = ser.unpack()
    ca_res = ser.unpack()
    ser = None
else:
    print("casadifying...")
    ca_res, ca_vars = upcycle.sympy2casadi(res_mat, out_syms)
    res = casadi.Function("res", casadi.vertsplit(ca_vars), casadi.vertsplit(ca_res))

    ser = casadi.FileSerializer(datfp)
    ser.pack(res)
    ser.pack(ca_vars)
    ser.pack(ca_res)
    ser = None


def idx_mx(mx, idx):
    return casadi.vertcat(*[casadi.vertsplit(mx)[i] for i in idx])


indep_idxs = np.setdiff1d(np.arange(len(x0)), non_indep_idxs)

nlp_opts = {
    "ipopt": {
        "print_level": 0,
        "warm_start_init_point": "yes",
    },
    "print_time": False,
}
subsys_idxs = [0, 1]  # indices in ca_res
inv_subsys_res_idxs = np.setdiff1d(np.arange(ca_res.size()[0]), subsys_idxs)
inv_subsys_var_idxs = np.r_[indep_idxs, inv_subsys_res_idxs + len(indep_idxs)]
subsys_y_prev = casadi.MX.sym("subsys_y_prev", len(subsys_idxs))
subsys_y_curr = casadi.MX.sym("subsys_y_curr", len(subsys_idxs))
subsys_x = idx_mx(idx_mx(ca_vars, non_indep_idxs), subsys_idxs)
subsys_nlp = {
    "x": subsys_x,
    "p": idx_mx(ca_vars, indep_idxs),
    "f": 0,
    "g": idx_mx(ca_res, subsys_idxs),
}
subsys_S = casadi.nlpsol("subsys_S", "ipopt", subsys_nlp, nlp_opts)

subsys_y_out = subsys_S(x0=subsys_y_curr, lbg=0, ubg=0)

# remove subsystem residuals from ca_res (parent)
# dummy symbol for previous solution (subsys_y_prev)
# call subsys_S with x0=subsys_y_prev
# output will be subsys_y_curr
# substitute guess for y in ca_res with subsys_y_curr
# delete guess for y from ca_vars
# (done?) parent problem gets a callback to shift subsys_y_prev => subsys_y_curr

# similar thing for explicit but substituting a Function?

new_ca_vars = idx_mx(ca_vars,  inv_subsys_var_idxs)
new_ca_res = idx_mx(ca_res, inv_subsys_res_idxs)

new_ca_res_subs = casadi.substitute(new_ca_res, subsys_x, subsys_y_out["x"])

nlp = {"x": new_ca_vars, "f": ca_vars[iobj], "p": subsys_y_curr, "g": new_ca_res_subs}
S = casadi.nlpsol("S", "ipopt", nlp, nlp_opts)

# res = casadi.Function("res", casadi.vertsplit(new_ca_vars), casadi.vertsplit(new_ca_res_subs))

S(
    x0=x0[inv_subsys_var_idxs],
    p=x0[[i+len(indep_idxs) for i in subsys_idxs]],
    ubx=ubx[inv_subsys_var_idxs],
    lbx=lbx[inv_subsys_var_idxs],
    ubg=0,
    lbg=0,
)


import sys
sys.exit()

df_res = pd.DataFrame(
    {
        "casadi": casadi.vertcat(*res(*x0)).toarray().squeeze(),
        "om": om_res[non_indep_idxs],
    },
    index=out_syms[non_indep_idxs],
)
print(df_res)

xz = np.zeros_like(x0)
print(xz[explicit_idxs])
for i in range(4):
    xz[non_indep_idxs] = np.array(res(*x0)).squeeze()
    x0[explicit_idxs] += xz[explicit_idxs]
    print(xz[explicit_idxs])

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

# print(df)
