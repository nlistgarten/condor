import numpy as np
import condor as co
import matplotlib.pyplot as plt


class MySystem(co.ODESystem):
    print("starting user code for LTI class")
    ts = list()
    for i in range(3):
        ts.append(parameter(name=f"t_{i}"))

    n = 2
    m = 1
    x = state(shape=n)
    C = state(shape=(n,n))# symmetric=True)
    A = np.array([
        [0, 1],
        [0, 0],
    ])


    # A = parameter(n,n)
    # B = parameter(n,m)
    # K = parameter(m,n)

    W = C.T @ C

    # indexing an output/computation by state
    dot[x] = A@x #(A - B @ K) @ x
    dot[C] = A@C + C@A.T
    # dot naming, although have to do more work to minimize other setattr's 
    output.y = C.T @ C

    print("ending user code for LTI class")


class MyEvent(MySystem.Event):
    function = MySystem.t - 10.
    update[x] = x**2 + C @ x

class MyEvent(MySystem.Event):
    function = MySystem.t - 100.
    update[x] = x -2

assert MyEvent.inner_to is MySystem
assert co.Event in MyEvent.__bases__
assert MySystem.Event not in MyEvent.__bases__
assert MySystem.Event in MySystem.inner_models
assert MyEvent in MySystem.Event
assert MyEvent.update._symbols[0].match in MySystem.state

class MySim(MySystem.TrajectoryAnalysis):
    # this over-writes which is actually really nice
    initial[x] = 1.
    initial[C] = np.eye(2)*parameter()

    out1 = trajectory_output(integrand=x.T@x)
    out2 = trajectory_output(x.T@x)
    out3 = trajectory_output(C[0,0], C[1,1])

    # correctly fails
    #out4 = trajectory_output(C[0,0], x)


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




class CtOptLQR(co.OptimizationProblem):
    K = variable(shape=DblIntLQR.K.shape)
    objective = DblIntLQR(K).cost

    class Casadi(co.Options):
        exact_hessian = False

lqr_sol = CtOptLQR()


from scipy.optimize import minimize

minimize(
    lambda p: DblIntLQR.implementation.callback(p).toarray().squeeze(),
    [0., 0.],
    jac=lambda p: DblIntLQR.implementation.callback.jac_callback(p, [0]).toarray().squeeze(),
    method='CG',
    #options=dict(maxiter=1),
)

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


class Sys1out(co.ExplicitSystem):
    x = input()
    y = input()
    output.z = x**2 + y

class Sys2out(co.ExplicitSystem):
    """
    A test explicit system with two outputs
    """
    x = input()
    y = input()
    output.w = x**2 + y**2
    output.z = x**2 + y

class MySolver(co.AlgebraicSystem):
    x = parameter()
    z = parameter()
    y2 = implicit_output(lower_bound=0., initializer=1.)
    y1 = implicit_output(lower_bound=0.)

    residual.y1 = y2 + x**2
    residual.y2 = y1 - x+z

    class Casadi(co.Options):
        warm_start = False

mysolution = MySolver(10, 1)
assert mysolution.y2 == -100
assert mysolution.y1 == 9

class MyProblem(co.OptimizationProblem):
    x = variable()
    y = variable()
    constraint(x**2 + y**2, lower_bound = 1., upper_bound = 1.)
    constraint(x**2 + y**2 + 5, eq=6.)

"""
class Solve(co.AlgebraicSystem):

class FailedSolver1(co.ExplicitSystem):
    x = input()
    y = input()
    output.x = x**2 + y**2

class FailedSolver2(co.ExplicitSystem):
    x = input()
    y = input()
    output = x**2 + y**2

class FailedSolver3(co.ExplicitSystem):
    x = input()
    y = input()
    output.input = x**2 + y**2

"""
