import condor as co

class Component(co.models.ModelTemplate):
    input = co.FreeField(co.Direction.input)
    output = co.AssignedField(co.Direction.output)

    x = placeholder(default=2.)
    y = placeholder(default=1.)

    output.z = x**2 + y

class ComponentImplementation(co.backend.implementations.TrajectoryAnalysis):
    pass

co.backend.implementations.Component = ComponentImplementation

Component() # error
# similarly, prevent user models from adding fields

class MyComp0(Component):
    pass

MyComp0() --> z=5

class MyComp5(Component):
    u = input()
    output.v = u**2 + 2*u + 1

MyComp0(3.) --> z= 5, v=13

class MyComp1(Component):
    x = input()
    y = input()

MyComp1(x=2., y=3.) -> z=7

class MyComp2(Component):
    x = input()
    y = 3.

MyComp2(1.0) -> z=4
MyComp2(3., 4.) # errors

class MyComp3(Component):
    x = 3.
    y = 4.

MyComp3() -> z= 13

class MyComp4(Component):
    u = input()
    x = u**0.5
    y = 0

    output.v = x + 5

MyComp4(4.) -> v= 7, z = 4



"""
"""



