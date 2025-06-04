from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sgm_test_util import LTI_plot

import condor as co

with_time_state = False
# either include time as state or increase tolerances to ensure sufficient ODE solver
# accuracy


class DblInt(co.ODESystem):
    A = np.array(
        [
            [0, 1],
            [0, 0],
        ]
    )
    B = np.array([[0, 1]]).T

    x = state(shape=A.shape[0])
    mode = state()
    pos_at_switch = state()

    t1 = parameter()
    t2 = parameter()
    u = modal()
    dot[x] = A @ x + B * u

    if with_time_state:
        tt = state()
        dot[tt] = 1.0


class Accel(DblInt.Mode):
    condition = mode == 0.0
    action[u] = 1.0


class Switch1(DblInt.Event):
    # function = t - t1
    at_time = t1
    update[mode] = 1.0

    update[pos_at_switch] = x[0]


class Decel(DblInt.Mode):
    condition = mode == 1.0
    action[u] = -1.0


class Switch2(DblInt.Event):
    # function = t - t2 - t1
    # mode == 2.

    at_time = t2 + t1
    # update[mode] = 2.
    terminate = True


class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9.0, 0.0]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x) / 2)

    if not with_time_state:

        class Casadi(co.Options):
            state_adaptive_max_step_size = 4


class AccelerateTransfer(DblInt.TrajectoryAnalysis, exclude_events=[Switch1]):
    initial[x] = [-9.0, 0.0]
    Q = np.eye(2)
    cost = trajectory_output((x.T @ Q @ x) / 2)

    if not with_time_state:

        class Casadi(co.Options):
            state_adaptive_max_step_size = 4


sim = Transfer(
    t1=1.0,
    t2=4.0,
)
print(sim.pos_at_switch)
# jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])


class MinimumTime(co.OptimizationProblem):
    t1 = variable(lower_bound=0)
    t2 = variable(lower_bound=0)
    transfer = Transfer(t1, t2)
    objective = transfer.cost

    # class Casadi(co.Options):
    class Options:
        exact_hessian = False
        __implementation__ = co.implementations.ScipyCG


"""
old:
eval jacobian for jac_Transfer
args [DM([1, 4]), DM(00)]
p=[1, 4]
[[-56.9839, 0]]



"""

MinimumTime.set_initial(t1=2.163165480675697, t2=4.361971866705403)

t_start = perf_counter()
opt = MinimumTime()
t_stop = perf_counter()

print("time to run:", t_stop - t_start)
print(opt.t1, opt.t2)
# print(jac)
print(opt._stats)
LTI_plot(opt.transfer)

sim_accel = AccelerateTransfer(**opt.transfer.parameter.asdict())
assert sim_accel._res.e[0].rootsfound.size == opt.transfer._res.e[0].rootsfound.size - 1

plt.show()
