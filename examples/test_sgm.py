import sys

import matplotlib.pyplot as plt
import numpy as np

import condor as co

dblintA = np.array(
    [
        [0, 1],
        [0, 0],
    ]
)
dblintB = np.array([[0, 1]]).T

DblInt = co.LTI(A=dblintA, B=dblintB, name="DblInt")


class DblIntLQR(DblInt.TrajectoryAnalysis):
    initial[x] = [1.0, 0.1]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 32.0
    cost = trajectory_output(integrand=(x.T @ Q @ x + (K @ x).T @ R @ (K @ x)) / 2)
    # cost = trajectory_output(integrand= x.T@Q@x + u.T @ R @ u)

    class Casadi(co.Options):
        nsteps = 5000
        atol = 1e-15
        # max_step = 1.


# ct_sim = DblIntLQR([1, .1])
# LTI_plot(ct_sim)

ct_sim = DblIntLQR(
    [
        0.0,
        0.0,
    ]
)
LTI_plot(ct_sim)
# plt.show()


sys.exit()

sampled_sim = DblIntSampledLQR([0.7, 0.2])
LTI_plot(sampled_sim)
plt.show()

lqr_sol_samp = SampledOptLQR()

# -------

DblIntSampled = co.LTI(A=dblintA, B=dblintB, name="DblIntSampled", dt=5.0)


class DblIntSampledLQR(DblIntSampled.TrajectoryAnalysis):
    initial[x] = [1.0, 0.0]
    initial[u] = -K @ [1.0, 0.0]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 100.0
    cost = trajectory_output(integrand=x.T @ Q @ x + u.T @ R @ u)

    class Casadi(co.Options):
        max_step = 1.0


sampled_sim = DblIntSampledLQR([5e-3, 0.1])
LTI_plot(sampled_sim)
plt.show()


DblIntDt = co.LTI(A=dblintA, B=dblintB, name="DblIntDt", dt=5.0, dt_plant=True)


class DblIntDtLQR(DblIntDt.TrajectoryAnalysis):
    initial[x] = [1.0, 0.0]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 32.0
    # cost = trajectory_output(integrand= x.T@Q@x + u.T @ R @ u)
    cost = trajectory_output(integrand=x.T @ Q @ x + (K @ x).T @ R @ (K @ x))

    class Casadi(co.Options):
        max_step = 1.0


dt_sim = DblIntDtLQR([0.005, 0.1])
LTI_plot(dt_sim)
plt.show()
