import numpy as np
import condor as co
import matplotlib.pyplot as plt
from sgm_test_util import LTI_plot
from scipy import linalg


class DblInt(co.ODESystem):
    A = np.array([
        [0, 1],
        [0, 0],
    ])
    B = np.array([[0,1]]).T
    x = state(shape=A.shape[0])
    t1 = parameter()
    t2 = parameter()
    # TODO: fix once mode and control work?
    dot[x] = A@x + B*(
        -1*(t < t1)
        +1*(t >= t1)*(t<(t1+t2))
    )

class Switch1(DblInt.Event):
    function = t - t1

class Switch2(DblInt.Event):
    function = t - t2 - t1
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [10., 0.]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x)/2)
    tf = 10.

    class Casadi(co.Options):
        nsteps = 5000
        atol = 1e-15
        max_step = 1.


sim = Transfer(t1=1., t2= 4.,)
LTI_plot(sim)
plt.show()

import sys
sys.exit()

from condor.backends.casadi.implementations import OptimizationProblem
class CtOptLQR(co.OptimizationProblem):
    K = variable(shape=DblIntLQR.K.shape)
    objective = DblIntLQR(K).cost

    class Casadi(co.Options):
        method = OptimizationProblem.Method.scipy_cg

lqr_sol = CtOptLQR()

S = linalg.solve_continuous_are(dblintA,dblintB, DblIntLQR.Q, DblIntLQR.R)
K = linalg.solve(DblIntLQR.R, dblintB.T@S)

lqr_are = DblIntLQR(K)
jac_callback = lqr_are.implementation.callback.jac_callback
jac_callback(K, [0])

