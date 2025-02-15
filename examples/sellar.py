import condor as co
warm_start = False

class Coupling(co.AlgebraicSystem):
    x = parameter(shape=3)
    y1 = variable(initializer=1., warm_start=warm_start)
    y2 = variable(initializer=1., warm_start=warm_start)

    residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
    residual(y2 == y1**0.5 + x[0] + x[1])

coupling = Coupling([5., 2., 1]) # evaluate the model numerically
print(coupling.y1, coupling.y2) # individual elements are bound numerically
print(coupling.variable) # fields are bound as a dataclass

from condor.backend import operators as ops

class Sellar(co.OptimizationProblem):
    x = variable(shape=3, lower_bound=0, upper_bound=10, warm_start=False)
    coupling = Coupling(x)
    y1, y2 = coupling

    objective = x[2]**2 + x[1] + y1 + ops.exp(-y2)
    constraint(y1 >= 3.16)
    constraint(24. >= y2)

    class Options:
        __implementation__ = co.implementations.OptimizationProblem
        print_level = 0

        @staticmethod
        def iter_callback(
            idx, variable, objective, constraints,
            instance=None
        ):
            print()
            #print(f"iter {idx}: {variable}")
            for k, v in constraints.asdict().items():
                elem = Sellar.constraint.get(name=k)
                #print(" "*4, f"{k}: {elem.lower_bound} < {v} < {elem.upper_bound}")
                #if instance is not None:
                #    print(" ", f"{instance.coupling.variable}")


Sellar.set_initial(x=[5,2,1])
import numpy as np
for idx in range(2):
    sellar = Sellar()
    print()
    print("objective value:", sellar.objective) # scalar value
    print(sellar.constraint) # field
    print(sellar.coupling.y1) # embedded-model element
    print(
        "embedded solver iter count:", Sellar.coupling.implementation.callback.total_iters
    )
    self = Sellar.coupling.implementation.callback
    self.update_warmstart( self.warm_start + True)
    Sellar.coupling.implementation.callback.total_iters = 0


Sellar.Options.__implementation__ = co.implementations.OptimizationProblem
Sellar.set_initial(x=[5,2,1])
#ipopt_s = Sellar()

