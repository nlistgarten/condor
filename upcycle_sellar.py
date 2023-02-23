import upcycle  # isort: skip

import pprint
import re

import casadi
import numpy as np
import pandas as pd
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data

from om_sellar_implicit import make_sellar_problem

prob = make_sellar_problem()
up_prob = make_sellar_problem()

prob.final_setup()
x0, lbx, ubx, iobj, non_indep_idxs, explicit_idxs = upcycle.extract_problem_data(prob)

# prob.run_driver()
# prob.list_problem_vars(print_arrays=True)

# symbol name, symbol
# upper bound, lower bound


#
sym_prob, res_mat, out_syms = upcycle.upcycle_problem(up_prob)
viwer_data = _get_viewer_data(up_prob)


ca_vars = casadi.vertcat(*[casadi.MX.sym(s.name) for s in out_syms])
ca_res, ca_vars_out = upcycle.sympy2casadi(res_mat, out_syms, ca_vars)
#


nlp = {"x": ca_vars, "f": ca_vars[iobj], "g": ca_res}
S = casadi.nlpsol("S", "ipopt", nlp)

# update explicit output guesses
res = casadi.Function("res", casadi.vertsplit(ca_vars), casadi.vertsplit(ca_res))
xz = np.zeros_like(x0)
xz[non_indep_idxs] = np.array(res(*x0)).squeeze()
x0[explicit_idxs] += xz[explicit_idxs]

df = pd.DataFrame(
    {
        "initial_value": x0,
        "upper_bound": ubx,
        "lower_bound": lbx,
    },
    index=out_syms,
)

out = S(x0=x0, lbx=lbx, ubx=ubx, lbg=0, ubg=0)

df["solution_value"] = out["x"].toarray()

for idx, row in df.iterrows():
    name = idx.name
    try:
        print(name)
        print(prob.get_val(name))
    except KeyError:
        match = re.match(r"(*)\[(\d)+\]$", name)
        base_name = match[1]
        i = int(match[2])
        print(base_name)
        print(prob.get_val(base_name)[i])

