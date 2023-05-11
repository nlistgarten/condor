import numpy as np
import condor as co
import matplotlib.pyplot as plt

from dataclasses import asdict

def LTI_plot(sim,t_slice=slice(None,None)):
    for field in [sim.state, sim.output]:
        for sym_name, symbol in asdict(field).items():
            if symbol.ndim > 1:
                n = symbol.shape[0]
            else:
                n = 1
            fig, axes = plt.subplots(n,1, constrained_layout=True, sharex=True)
            plt.suptitle(f"{sim.__class__.__name__} {field.__class__.__name__}.{sym_name}")
            if n > 1:
                for ax, x in zip(axes, symbol):
                    ax.plot(sim.t[t_slice], x.squeeze()[t_slice])
                    ax.grid(True)
            else:
                plt.plot(sim.t[t_slice], symbol.squeeze()[t_slice])
                plt.grid(True)


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
    cost = trajectory_output(integrand= (x.T@Q@x + (K@x).T @ R @ (K@x))/2)
    #cost = trajectory_output(integrand= x.T@Q@x + u.T @ R @ u)

    class Casadi(co.Options):
        nsteps = 5000
        atol = 1e-15
        #max_step = 1.



#ct_sim = DblIntLQR([1, .1])
#LTI_plot(ct_sim)

ct_sim = DblIntLQR([0., 0.,])
LTI_plot(ct_sim)
#plt.show()




from condor.backends.casadi.implementations import OptimizationProblem
class CtOptLQR(co.OptimizationProblem):
    K = variable(shape=DblIntLQR.K.shape)
    objective = DblIntLQR(K).cost

    class Casadi(co.Options):
        method = OptimizationProblem.Method.scipy_cg

lqr_sol = CtOptLQR()


import sys
sys.exit()

DblIntSampled = co.LTI(A=dblintA, B=dblintB, name="DblIntSampled", dt=5.)
class DblIntSampledLQR(DblIntSampled.TrajectoryAnalysis):
    initial[x] = [1., 0.]
    initial[u] = -K@[1., 0.]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 100.
    cost = trajectory_output(integrand= x.T@Q@x + u.T @ R @ u)

    class Casadi(co.Options):
        max_step = 1.

sampled_sim = DblIntSampledLQR([5E-3, 0.1])
LTI_plot(sampled_sim)


DblIntDt = co.LTI(A=dblintA, B=dblintB, name="DblIntDt", dt=5., dt_plant=True)
class DblIntDtLQR(DblIntDt.TrajectoryAnalysis):
    initial[x] = [1., 0.]
    Q = np.eye(2)
    R = np.eye(1)
    tf = 100.
    #cost = trajectory_output(integrand= x.T@Q@x + u.T @ R @ u)
    cost = trajectory_output(integrand= x.T@Q@x + (K@x).T @ R @ (K@x))

    class Casadi(co.Options):
        max_step = 1.

dt_sim = DblIntDtLQR([0.005, 0.1])
LTI_plot(dt_sim)
plt.show()

