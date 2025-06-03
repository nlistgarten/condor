from casadi import exp

import condor as co


class Resid(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = implicit_output(initializer=1.0)
    y2 = implicit_output(initializer=1.0)

    residual.y1 = y1 - z[0] ** 2 - z[1] - x + 0.2 * y2
    residual.y2 = y2 - y1**0.5 - z[0] - z[1]


class Obj(co.ExplicitSystem):
    x = input()
    z = input(shape=2)

    y1, y2 = Resid(x, z)

    output.obj = x**2 + z[1] + y1 + exp(-y2)


class Constr(co.ExplicitSystem):
    x = input()
    z = input(shape=2)

    resid = Resid(x, z)

    output.con1 = 3.16 - resid.y1
    output.con2 = resid.y2 - 24.0


do_bind = True


class Sellar(co.OptimizationProblem, bind_submodels=do_bind):
    x = variable(lower_bound=0, upper_bound=10)
    z = variable(shape=2, lower_bound=0, upper_bound=10)

    objective = Obj(x, z).obj
    constrs = Constr(x, z)
    constraint(constrs.con1, upper_bound=0.0)
    constraint(constrs.con2, upper_bound=0.0)


resid_sol = Resid(1, [5.0, 2.0])
Sellar.implementation.set_initial(
    x=1.0,
    z=[
        5.0,
        2.0,
    ],
)
sellar_opt = Sellar()
resid_at_opt = Resid(*sellar_opt)
obj_at_opt = Obj(*sellar_opt)

if do_bind:
    print("should get real numbers")
else:
    print("should get symbolic result from model definition")

print(sellar_opt.constrs.resid.y1, sellar_opt.constrs.resid.y2)
