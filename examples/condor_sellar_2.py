import condor as co
from casadi import exp

class Resid(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = implicit_output(initializer=1.)
    y2 = implicit_output(initializer=1.)

    residual.y1 = y1 - z[0] ** 2 - z[1] - x + 0.2 * y2
    residual.y2 = y2 - y1**0.5 - z[0] - z[1]

class Sellar(co.OptimizationProblem):
    x = variable(lower_bound=0, upper_bound=10)
    z = variable(shape=2, lower_bound=0, upper_bound=10)

    y1, y2 = Resid(x, z)

    objective = x**2 + z[1] + y1 + exp(-y2)
    constraint(y1, lower_bound=3.16, name="con1")
    constraint(y2, upper_bound=24.)

Sellar.implementation.set_initial(x=1., z=[5., 2.,])
sellar_opt2 = Sellar()

