import condor as co
import numpy as np

class MatSys(co.ExplicitSystem):
    A = input(shape=(3,4))
    B = input(shape=(4,2))
    output.C = A@B

ms = MatSys(np.random.rand(3,4), np.random.rand(4,2))

class Sys(co.ExplicitSystem):
    x = input()
    y = input()
    v = y**2
    output.w = x**2 + y**2
    output.z = x**2 + y

sys = Sys(1.2, 3.4)
print(sys, sys.output)

class Opt(co.OptimizationProblem):
    x = variable()
    y = variable()

    sys = Sys(x=x, y=y)

    objective = (sys.w - 1)**2 - sys.z

Opt.set_initial(x=3., y=4.)
opt = Opt()
