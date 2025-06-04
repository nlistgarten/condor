from casadi import exp

import condor as co


class Resid(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = variable(initializer=1.0)
    y2 = variable(initializer=1.0)

    residual(-y1 == -(z[0] ** 2) - z[1] - x + 0.2 * y2)
    residual(-y2 == -(y1**0.5) - z[0] - z[1])


class Obj(co.ExplicitSystem):
    x = input()
    z = input(shape=2)

    y1, y2 = Resid(x, z)

    output.obj = x**2 + z[1] + y1 + exp(-y2)


class Constr(co.ExplicitSystem):
    x = input()
    z = input(shape=2)

    y1, y2 = Resid(x, z)

    output.con1 = 3.16 - y1
    output.con2 = y2 - 24.0


class Sellar(co.OptimizationProblem):
    x = variable(lower_bound=0, upper_bound=10)
    z = variable(shape=2, lower_bound=0, upper_bound=10)

    objective = Obj(x, z).obj
    constrs = Constr(x, z)
    constraint(constrs.con1, upper_bound=0.0)
    constraint(constrs.con2, upper_bound=0.0)

    class Options:
        # method = (
        #    co.backends.casadi.implementations.OptimizationProblem.Method.scipy_slsqp
        # )
        # disp = True
        iprint = 2
        tol = 1e-8
        maxiter = 0

        # @staticmethod
        # def iter_callback(idx, variable, objective, constraints):
        #    print()
        #    print(f"iter {idx}: {variable}")
        #    for k, v in constraints.asdict().items():
        #        elem = Design.constraint.get(name=k)
        #        print(" "*4, f"{k}: {elem.lower_bound} < {v} < {elem.upper_bound}")


# resid_sol = Resid(1, [5., 2.])
Sellar.set_initial(
    x=1.0,
    z=[
        5.0,
        2.0,
    ],
)
# Sellar.implementation.set_initial(x=0, z=[3.15, 0.,])
sellar_opt = Sellar()
resid_at_opt = Resid(*sellar_opt)
obj_at_opt = Obj(*sellar_opt)
