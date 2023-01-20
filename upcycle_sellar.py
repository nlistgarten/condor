import os

import openmdao.api as om

from om_sellar_implicit import make_sellar_problem
from upcycle import SymbolicVector

# avoid some errors from reports expecting float data
os.environ["OPENMDAO_REPORTS"] = "none"

print("checking problem optimization")
print(40 * "=")
prob = make_sellar_problem()

# check the problem still works
prob.run_driver()


prob.setup(local_vector_class=SymbolicVector)

prob.final_setup()

print("apply nonlinear")
print(40 * "=")

prob.model._apply_nonlinear()

print(*list(prob.model._residuals.items()), sep="\n")
