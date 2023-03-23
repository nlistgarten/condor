import upcycle  # isort: skip

import os
import pprint
import re

import numpy as np
import pandas as pd

from om_sellar_implicit import make_sellar_problem

os.environ["OPENMDAO_REPORTS"] = "none"

updriver, prob = upcycle.upcycle_problem(make_sellar_problem)

### Check run_model
upsolver = updriver.children[0]
inputs = np.hstack([upcycle.get_val(prob, absname) for absname in upsolver.inputs])
out = upsolver(inputs)[0]


prob.set_solver_print(-1)
prob.run_model()

cols = ("name", "om_val", "ca_val")
vals = []
for name, ca_val in zip(upsolver.outputs, out):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df_root = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
assert df_root[~np.isclose(df_root["om_val"], df_root["ca_val"], rtol=0., atol=1e-13)].size == 0
assert df_root[~np.isclose(df_root["om_val"], df_root["ca_val"], rtol=1e-13, atol=0.)].size == 0

## try optimizer
inputs = np.hstack([upcycle.get_val(prob, absname) for absname in updriver.inputs])
out = updriver(inputs)
prob.run_driver()

vals = []
for name, ca_val in zip(updriver.outputs, out["g"].toarray().squeeze()):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])
for name, ca_val in zip(updriver.inputs, out["x"].toarray().squeeze()):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df_opt = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
assert df_opt[~np.isclose(df_opt["om_val"], df_opt["ca_val"], rtol=0., atol=1e-7)].size == 0
