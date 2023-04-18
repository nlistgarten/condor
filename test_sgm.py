import numpy as np
import condor as co

class MySystem(co.DynamicsModel):
    print("starting user code for LTI class")
    n = 2
    m = 1
    x = state(n)
    C = state(n=n,m=n, symmetric=True)
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

class Sys2(co.ExplicitSystem):
    x = input()
    y = input()
    output.z = x**2 + y

class MySolver(co.AlgebraicSystem):
    x = parameter()
    z = parameter(n=2)
    y2 = implicit_output()
    y1 = implicit_output()

    residual.y1 = y2 + x**2
    residual.y2 = y1 - x+z

    initializer[y2] = 1.
