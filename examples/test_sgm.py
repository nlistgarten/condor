import numpy as np
import condor as co

class MySystem(co.ODESystem):
    print("starting user code for LTI class")
    n = 2
    m = 1
    x = state(n)
    C = state((n,n),)# symmetric=True)
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
    update[x] = 1.0

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
