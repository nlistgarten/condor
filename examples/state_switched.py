import numpy as np
import condor as co
import matplotlib.pyplot as plt
from sgm_test_util import LTI_plot
from scipy import linalg
import casadi as ca


class DblInt(co.ODESystem):
    A = np.array([
        [0, 1],
        [0, 0],
    ])
    B = np.array([[0,1]]).T
    x = state(shape=A.shape[0])
    mode = state()
    p1 = parameter()
    p2 = parameter()
    u = modal()
    dot[x] = A@x + B*u

class Accel(DblInt.Mode):
    condition = mode == 0.
    action[u] = 1.

class Switch1(DblInt.Event):
    function = x[0] - p1
    update[mode] = 1.

class Decel(DblInt.Mode):
    condition = mode == 1.
    action[u] = -1.

class Switch2(DblInt.Event):
    function = x[0] - p2
    #update[mode] = 2.
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    xd = [1., 2.]
    Q = np.eye(2)
    cost = trajectory_output(
        ((x-xd).T @ (x-xd))/2
    )
    tf = 20.

    class Casadi(co.Options):
        state_max_step_size = 0.25
        state_atol = 1E-15
        state_rtol = 1E-12
        adjoint_atol = 1E-15
        adjoint_rtol = 1E-12
        #state_solver = co.backend.implementations.TrajectoryAnalysis.Solver.CVODE
        #adjoint_solver = co.backend.implementations.TrajectoryAnalysis.Solver.CVODE

from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    p1 = variable()
    p2 = variable()
    sim = Transfer(p1, p2)
    objective = sim.cost

    class Casadi(co.Options):
        exact_hessian = False
        #method = OptimizationProblem.Method.scipy_cg

p0 = -4., -1.
sim = Transfer(*p0)
sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
LTI_plot(sim)
#plt.show()

MinimumTime.implementation.set_initial(p1=p0[0], p2=p0[1])

from time import perf_counter

t_start = perf_counter()
opt = MinimumTime()
t_stop = perf_counter()
print("time to run:", t_stop - t_start)
print(opt.p1, opt.p2)
print(opt._stats)

