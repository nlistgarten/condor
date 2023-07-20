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
    dot[x] = A@x*(mode!=2.) + B*(
        1*(mode==0.)
        -1*(mode==1.)
    )

class Switch1(DblInt.Event):
    function = x[0] - p1
    update[mode] = 1.

class Switch2(DblInt.Event):
    function = x[0] - p2
    update[mode] = 2.
    terminate = True

class Transfer(DblInt.TrajectoryAnalysis):
    initial[x] = [-9., 0.]
    xd = [1., 2.]
    Q = np.eye(2)
    cost = trajectory_output(
        ((x-xd).T @ (x-xd))/2
    )
    tf = 100.

    class Casadi(co.Options):
        integrator_options = dict(
            max_step = 1.,
            atol = 1E-15,
            rtol = 1E-12,
        )

from condor.backends.casadi.implementations import OptimizationProblem
class MinimumTime(co.OptimizationProblem):
    p1 = variable()
    p2 = variable()
    objective = Transfer(p1, p2).cost

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

