import numpy as np
import condor as co
import matplotlib.pyplot as plt
from sgm_test_util import LTI_plot
from scipy import linalg

dblintA = np.array([
    [0, 1],
    [0, 0],
])
dblintB = np.array([[0,1]]).T

DblInt = co.LTI(A=dblintA, B=dblintB, name="DblInt")
class DblIntLQR(DblInt.TrajectoryAnalysis):
    initial[x] = [1., 0.1]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 32.
    u = output.u
    cost = trajectory_output(integrand= (x.T@Q@x + u.T @ R @ u)/2)

    class Casadi(co.Options):
        nsteps = 5000
        atol = 1e-15
        #max_step = 1.



ct_sim = DblIntLQR([1, .1])
LTI_plot(ct_sim)

ct_sim = DblIntLQR([0., 0.,])
LTI_plot(ct_sim)


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

print(lqr_sol._stats)

