import time
tic = time.perf_counter()

import condor as co
from casadi import exp

class Coupling(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = implicit_output(initializer=1.)
    y2 = implicit_output(initializer=1.)

    residual.y1 = y1 - z[0] ** 2 - z[1] - x + 0.2 * y2
    residual.y2 = y2 - y1**0.5 - z[0] - z[1]


coupling = Coupling(1, [5., 2.])

class Sellar(co.OptimizationProblem):
    x = variable(lower_bound=0, upper_bound=10)
    z = variable(shape=2, lower_bound=0, upper_bound=10)
    coupling = Coupling(x, z)
    y1, y2 = coupling

    objective = x**2 + z[1] + y1 + exp(-y2)
    constraint(y1, lower_bound=3.16, name="con1")
    constraint(y2, upper_bound=24.)
    constraint(y1+y2)

    #constraint(x, lower_bound=0, upper_bound=10)
    #constraint(z, lower_bound=0, upper_bound=10)

    class Casadi(co.Options):
        if True:
            method = (
                co.backends.casadi.implementations.OptimizationProblem.Method.scipy_slsqp
                #co.backends.casadi.implementations.OptimizationProblem.Method.scipy_trust_constr
            )
            #disp = True
            #iprint = 3
            #tol = 1E-9
            #maxiter = 2

Sellar.implementation.set_initial(x=1., z=[5., 2.,])
Sellar._meta.bind_submodels = False
sellar_opt2 = Sellar()

toc = time.perf_counter()
print("total time:", toc - tic)
import sys
sys.exit()

for con in Sellar.implementation.con:
    print(con["fun"]([1,5,2], []))
    print(con["jac"]([1,5,2], []))

