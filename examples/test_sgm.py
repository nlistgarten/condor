import numpy as np
import condor as co


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
    function = MySystem.independent_variable - 10.
    update[x] = x**2 + C @ x

class MyEvent(MySystem.Event):
    function = MySystem.independent_variable - 100.
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
