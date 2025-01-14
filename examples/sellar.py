import condor as co

class Coupling(co.AlgebraicSystem):
    x = parameter(shape=3)
    y1 = variable(initializer=1.)
    y2 = variable(initializer=1.)

    residual(y1 == x[0] ** 2 + x[1] + x[2] - 0.2 * y2)
    residual(y2 == y1**0.5 + x[0] + x[1])

coupling = Coupling([5., 2., 1]) # evaluate the model numerically
print(coupling.y1, coupling.y2) # individual elements are bound numerically
print(coupling.variable) # fields are bound as a dataclass

from condor import operators as ops

class Sellar(co.OptimizationProblem):
    x = variable(shape=3, lower_bound=0, upper_bound=10)
    coupling = Coupling(x)
    y1, y2 = coupling

    objective = x[2]**2 + x[1] + y1 + ops.exp(-y2)
    constraint(y1 > 3.16)
    constraint(24. > y2)

    class Options:
        pass

        method = (
            co.implementations.OptimizationProblem.Method.scipy_slsqp
            #co.implementations.OptimizationProblem.Method.scipy_trust_constr
        )
        #disp = True
        #verbose=3
        #iprint = 2
#        tol = 1E-8
#        maxiter = 0

        @staticmethod
        def iter_callback(
            idx, variable, objective, constraints,
            instance=None
        ):
            print()
            print(f"iter {idx}: {variable}")
            for k, v in constraints.asdict().items():
                elem = Sellar.constraint.get(name=k)
                print(" "*4, f"{k}: {elem.lower_bound} < {v} < {elem.upper_bound}")
                print(" ", f"{instance.coupling.variable}")


Sellar.set_initial(x=[5,2,1])
sellar = Sellar()
print()
print("objective value:", sellar.objective) # scalar value
print(sellar.constraint) # field
print(sellar.coupling.y1) # embedded-model element

Sellar.Options.method = co.implementations.OptimizationProblem.Method.ipopt
ipopt_s = Sellar()

