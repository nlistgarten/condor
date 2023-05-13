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
        1*(t < t1)
        -1*(t >= t1)*(t<(t1+t2))
    )

class Switch1(DblInt.Event):
    function = t - t1

class Switch2(DblInt.Event):
    function = t - t2 - t1
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x)/2)
    tf = 10.

    class Casadi(co.Options):
        nsteps = 5000
        atol = 1e-15
        max_step = 1.


from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    t1 = variable(lower_bound=0)
    t2 = variable(lower_bound=0)
    objective = Transfer(t1, t2).cost
    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_cg

sim = Transfer(t1=1., t2= 4.,)
LTI_plot(sim)
#plt.show()
0.0169526, 3.76823
broke_ts = [0.01695263, 3.76822885]

#sim = Transfer(*broke_ts)
#jac_callback = sim.implementation.callback.jac_callback
#jac_callback(broke_ts, [0])
MinimumTime.implementation.set_initial(t1=2.163165480675697, t2=4.361971866705403)
opt = MinimumTime()

