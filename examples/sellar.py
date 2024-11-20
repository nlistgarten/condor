import time
tic = time.perf_counter()

import condor as co
from casadi import exp

class Coupling(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = implicit_output(initializer=1.)
    y2 = implicit_output(initializer=1.)

    y1_agreement = residual(y1 == z[0] ** 2 + z[1] + x - 0.2 * y2)
    residual(y2 == y1**0.5 + z[0] + z[1])


coupling = Coupling(1, [5., 2.])


class Sellar(co.OptimizationProblem):
    x = variable(lower_bound=0, upper_bound=10)
    z = variable(shape=2, lower_bound=0, upper_bound=10)
    coupling = Coupling(x, z)
    y1, y2 = coupling

    objective = x**2 + z[1] + y1 + exp(-y2)
    constraint(3.16 < y1, name="y1_bound")
    y2_bound = constraint(y2 < 24)

    #constraint(x, lower_bound=0, upper_bound=10)
    #constraint(z, lower_bound=0, upper_bound=10)

    class Options:
        @staticmethod
        def iteration_callback(i, variable, objective, constraints):
            print(f"iter {i}: x = {variable.x}, z = {variable.z.squeeze()}")
            print(f"objective = {objective}")
            print(constraints)
            sellar_init = Sellar.from_values(**variable.asdict())
            print(sellar_init.coupling.implicit_output)

        print_level = 0


#Sellar.implementation.set_initial(x=1., z=[5., 2.,])
Sellar.x.initializer = 1.
Sellar.z.initializer = [5., 2.,]
#Sellar._meta.bind_embedded_models = False
sellar_opt = Sellar()


toc = time.perf_counter()
print("total time:", toc - tic)

print(sellar_opt.coupling, sellar_opt.coupling.implicit_output)
