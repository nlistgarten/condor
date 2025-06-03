import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sgm_test_util import LTI_plot

import condor as co

dblintA = np.array(
    [
        [0, 1],
        [0, 0],
    ]
)
dblintB = np.array([[0, 1]]).T

DblInt = co.LTI(A=dblintA, B=dblintB, name="DblInt")
# class Terminate(DblInt.Event):
#    at_time = 32.,
#    terminate = True


class DblIntLQR(DblInt.TrajectoryAnalysis):
    initial[x] = [1.0, 0.1]
    Q = np.eye(2)
    R = np.eye(1)
    # tf = None
    tf = 32.0
    u = dynamic_output.u
    cost = trajectory_output(integrand=(x.T @ Q @ x + u.T @ R @ u) / 2)

    class Options:
        state_rtol = 1e-8
        adjoint_rtol = 1e-8
        pass


ct_sim = DblIntLQR([1, 0.1])
LTI_plot(ct_sim)
# ct_sim = DblIntLQR([0., 0.,])
# LTI_plot(ct_sim)


class CtOptLQR(co.OptimizationProblem):
    K = variable(shape=DblIntLQR.K.shape)
    objective = DblIntLQR(K).cost

    class Options:
        exact_hessian = False
        __implementation__ = co.implementations.ScipyCG


from time import perf_counter

t_start = perf_counter()
lqr_sol = CtOptLQR()
t_stop = perf_counter()

S = linalg.solve_continuous_are(dblintA, dblintB, DblIntLQR.Q, DblIntLQR.R)
K = linalg.solve(DblIntLQR.R, dblintB.T @ S)

lqr_are = DblIntLQR(K)

print(lqr_sol._stats)
print(lqr_are.cost, lqr_sol.objective)
print(lqr_are.cost > lqr_sol.objective)
print("      ARE sol:", K, "\niterative sol:", lqr_sol.K)
print("time to run:", t_stop - t_start)

plt.show()
import sys

sys.exit()
