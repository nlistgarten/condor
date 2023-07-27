import numpy as np
import condor as co
import matplotlib.pyplot as plt
from sgm_test_util import LTI_plot
from scipy import linalg

with_time_state = False
# either include time as state or increase tolerances to ensure sufficient ODE solver
# accuracy

class DblInt(co.ODESystem):
    A = np.array([
        [0, 1],
        [0, 0],
    ])
    B = np.array([[0,1]]).T

    x = state(shape=A.shape[0])
    mode = state()


    t1 = parameter()
    t2 = parameter()
    u = modal()
    dot[x] = A@x + B*u

    if with_time_state:
        tt = state()
        dot[tt] = 1.

class Accel(DblInt.Mode):
    condition = mode == 0.
    action[u] = 1.

class Switch1(DblInt.Event):
    #function = t - t1
    at_time = t1,
    update[mode] = 1.

class Decel(DblInt.Mode):
    condition = mode == 1.
    action[u] = -1.

class Switch2(DblInt.Event):
    #function = t - t2 - t1
    #mode == 2.

    at_time = t2 + t1,
    #update[mode] = 2.
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x)/2)

    if not with_time_state:
        class Casadi(co.Options):
            state_adaptive_max_step_size = True
            state_max_step_size = 4


from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    t1 = variable(lower_bound=0)
    t2 = variable(lower_bound=0)
    objective = Transfer(t1, t2).cost

    class Casadi(co.Options):
        exact_hessian = False
        method = OptimizationProblem.Method.scipy_cg

sim = Transfer(t1=1., t2= 4.,)
jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])

"""
old:
eval jacobian for jac_Transfer
args [DM([1, 4]), DM(00)]
p=[1, 4]
[[-56.9839, 0]]



"""

LTI_plot(sim)

#plt.show()
MinimumTime.implementation.set_initial(t1=2.163165480675697, t2=4.361971866705403)

from time import perf_counter

t_start = perf_counter()
opt = MinimumTime()
t_stop = perf_counter()

print("time to run:", t_stop - t_start)
print(opt.t1, opt.t2)
print(jac)
print(opt._stats)
