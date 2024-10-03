import numpy as np
import condor as co
import matplotlib.pyplot as plt
from scipy import linalg, signal
from sgm_test_util import LTI_plot

dblintA = np.array([
    [0, 1],
    [0, 0],
])
dblintB = np.array([[0,1]]).T
dt = 0.5


DblIntSampled = co.LTI(A=dblintA, B=dblintB, name="DblIntSampled", dt=dt)
class DblIntSampledLQR(DblIntSampled.TrajectoryAnalysis):
    initial[x] = [1., 0.1]
    #initial[u] = -K@initial[x]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 32. # 12 iters, 21 calls 1E-8 jac
    #tf = 16. # 9 iters, 20 calls, 1E-7
    cost = trajectory_output(integrand= (x.T@Q@x + u.T @ R @ u)/2)
    class Casadi(co.Options):
        adjoint_adaptive_max_step_size=False
        state_max_step_size=0.5/8
        adjoint_max_step_size=0.5/8

from condor.backends.casadi.implementations import OptimizationProblem
class SampledOptLQR(co.OptimizationProblem):
    K = variable(shape=DblIntSampledLQR.K.shape)
    objective = DblIntSampledLQR(K).cost
    class Casadi(co.Options):
        exact_hessian = False
        #method = OptimizationProblem.Method.scipy_cg
        #method = OptimizationProblem.Method.scipy_trust_constr


sim = DblIntSampledLQR([1.00842737, 0.05634044])

sim = DblIntSampledLQR([0., 0.])
sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])


from time import perf_counter

t_start = perf_counter()
lqr_sol_samp = SampledOptLQR()
t_stop = perf_counter()


#sampled_sim = DblIntSampledLQR([0., 0.])
#sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])

Q = DblIntSampledLQR.Q
R = DblIntSampledLQR.R
A = dblintA
B = dblintB

Ad, Bd = signal.cont2discrete((A, B, None, None), dt)[:2]
S = linalg.solve_discrete_are(Ad, Bd, Q, R,)
K = linalg.solve(Bd.T @ S @ Bd + R, Bd.T @ S @ Ad)

#sim = DblIntSampledLQR([1.00842737, 0.05634044])
sim = DblIntSampledLQR(K)

jac = sim.implementation.callback.jac_callback(sim.implementation.callback.p, [])
LTI_plot(sim)
plt.show()

#sim = DblIntSampledLQR([0., 0.])



#sampled_sim = DblIntSampledLQR([0., 0.])
#sampled_sim.implementation.callback.jac_callback([0., 0.,], [0.])



sampled_sim = DblIntSampledLQR(K)
jac_cb= sampled_sim.implementation.callback.jac_callback
jac_cb(K, [0.])

print(lqr_sol_samp._stats)
print(lqr_sol_samp.objective < sampled_sim.cost)
print(lqr_sol_samp.objective, sampled_sim.cost)
print("      ARE sol:", K,
    "\niterative sol:", lqr_sol_samp.K)
print("time to run:", t_stop - t_start)
