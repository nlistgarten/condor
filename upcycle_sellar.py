import upcycle  # isort: skip

import os
import pprint
import re

import numpy as np
import pandas as pd

from om_sellar_implicit import make_sellar_problem

os.environ["OPENMDAO_REPORTS"] = "none"

upsolver, prob = upcycle.upcycle_problem(make_sellar_problem)

inputs = np.hstack([upcycle.get_val(prob, absname) for absname in upsolver.inputs])
out = upsolver.run(*inputs)
print(out)

prob.set_solver_print(-1)
prob.run_model()

cols = ("name", "om_val", "ca_val")
vals = []
for name, ca_val in zip(upsolver.outputs, out):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=0., atol=1e-13)])
print(df[~np.isclose(df["om_val"], df["ca_val"], rtol=1e-13, atol=0.)])


