"""
Continuous Time LQR
===================
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from _sgm_test_util import LTI_plot
from scipy import linalg

import condor as co

dblintA = np.array([[0, 1], [0, 0]])
dblintB = np.array([[0, 1]]).T

DblInt = co.LTI(a=dblintA, b=dblintB, name="DblInt")


class DblIntLQR(DblInt.TrajectoryAnalysis):
    initial[x] = [1.0, 0.1]

    Q = np.eye(2)
    R = np.eye(1)

    tf = 32.0

    u = dynamic_output.u
    cost = trajectory_output(integrand=(x.T @ Q @ x + u.T @ R @ u) / 2)

    class Options:
        state_rtol = 1e-8
        adjoint_rtol = 1e-8


ct_sim = DblIntLQR(k=[1.0, 0.1])
LTI_plot(ct_sim)


# %%


class CtOptLQR(co.OptimizationProblem):
    k = variable(shape=DblIntLQR.k.shape)
    objective = DblIntLQR(k).cost

    class Options:
        # exact_hessian = False
        __implementation__ = co.implementations.ScipyCG


t_start = perf_counter()
lqr_sol = CtOptLQR()
t_stop = perf_counter()


S = linalg.solve_continuous_are(dblintA, dblintB, DblIntLQR.Q, DblIntLQR.R)
k = linalg.solve(DblIntLQR.R, dblintB.T @ S)

lqr_are = DblIntLQR(k)

print(lqr_sol._stats)
print(lqr_are.cost, lqr_sol.objective)
print(lqr_are.cost > lqr_sol.objective)
print("      ARE sol:", k, "\niterative sol:", lqr_sol.k)
print("time to run:", t_stop - t_start)

plt.show()
