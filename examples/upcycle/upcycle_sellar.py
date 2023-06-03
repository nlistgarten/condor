import condor.upcycle as upcycle # isort: skip


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

# TODO: future, general serialization should print to file with import condor and numpy
# or backend ufuncs -- generally like using numpy and expect the backend to work. casadi
# and sympy array with numpy dispatching both do

from numpy import exp, array
import condor as co
exec(upsolver.model_string)
exec(updriver.model_string)

solver_inputs = [
    #prob.get_val(upcycle.contaminate_variable_name(absname))
    upcycle.get_val(prob, absname)
    for absname in upsolver.inputs
]
root_out = root(*solver_inputs)


prob.set_solver_print(-1)
prob.run_model()

cols = ("name", "om_val", "co_val")
vals = []
#for name, co_val in zip(upsolver.outputs, out):
for name in upsolver.outputs:
    om_val = upcycle.get_val(prob, name).reshape(-1)[0]
    co_val = getattr(root_out, name).toarray().reshape(-1)[0]
    vals.append([name, om_val, co_val])
df_root = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
assert df_root[~np.isclose(df_root["om_val"], df_root["co_val"], rtol=0., atol=1e-12)].size == 0
assert df_root[~np.isclose(df_root["om_val"], df_root["co_val"], rtol=1e-12, atol=0.)].size == 0

## try optimizer
#inputs = np.hstack([upcycle.get_val(prob, absname) for absname in updriver.inputs])
driver_inputs = [
    #prob.get_val(upcycle.contaminate_variable_name(absname))
    upcycle.get_val(prob, absname)
    for absname in updriver.inputs
]
driver_out = optimizer(*driver_inputs)
prob.run_driver()

vals = []
#for name, co_val in zip(upsolver.outputs, out):
for name in updriver.inputs:
    om_val = upcycle.get_val(prob, name).reshape(-1)
    co_val = getattr(driver_out, name).reshape(-1)
    vals.append([name, om_val, co_val])

df_opt = pd.DataFrame(vals, columns=cols)
assert np.sum(np.abs(np.concatenate((df_opt['om_val'] - df_opt['co_val']).values.tolist())) > 1E-6) == 0
# df_opt[~np.isclose(df_opt["om_val"], df_opt["co_val"], rtol=0., atol=1e-7)].size == 0

import sys
sys.exit()



for name, ca_val in zip(updriver.outputs, out["g"].toarray().squeeze()):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])
for name, ca_val in zip(updriver.inputs, out["x"].toarray().squeeze()):
    om_val = upcycle.get_val(prob, name)
    vals.append([name, om_val, ca_val])

df_opt = pd.DataFrame(vals, columns=cols)

# passes with or without warm start
assert df_opt[~np.isclose(df_opt["om_val"], df_opt["ca_val"], rtol=0., atol=1e-7)].size == 0
