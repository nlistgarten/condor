import condor as co
import numpy as np


class ComponentRaw(co.models.ModelTemplate):
    """ Raw Component base """
    input = co.FreeField(co.Direction.input)
    output = co.AssignedField(co.Direction.output)

    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y


class ComponentImplementation(co.implementations.ExplicitSystem):
    pass


co.implementations.ComponentRaw = ComponentImplementation


class ComponentAT(co.ExplicitSystem, as_template=True):
    """ AT component base """
    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y

class MyComponentR(ComponentRaw):
    """ my component R """
    u = input()
    output.w = z+u

class MyComponentA(ComponentAT):
    """ my component A """
    u = input()
    output.w = z+u

MyComponentR(u=1.23).z == MyComponentA(u=1.23).z

comp = MyComponent(u=1., z=5.)

class MyComponent1(Component):
    pass

comp1 = MyComponent1()


class MyComponent2(Component):
    u = input()
    output.x = z+u

comp2 = MyComponent2(u=1.)


class MatSys(co.ExplicitSystem):
    A = input(shape=(3,4))
    B = input(shape=(4,2))
    output.C = A@B

ms = MatSys(np.random.rand(3,4), np.random.rand(4,2))


class SymMatSys(co.ExplicitSystem):
    A = input(shape=(3,3), symmetric=True)
    B = input(shape=(3,3))
    output.C = A@B + B.T @ A

a = np.random.rand(3,3)
sms = SymMatSys(a + a.T, np.random.rand(3,3))

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
