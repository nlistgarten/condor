import condor as co

if True:
    class Component(co.models.ModelTemplate):
        input = co.FreeField(co.Direction.input)
        output = co.AssignedField(co.Direction.output)

        x = placeholder(default=2.)
        y = placeholder(default=1.)

        output.z = x**2 + y

    class ComponentImplementation(co.backend.implementations.ExplicitSystem):
        pass

    co.backend.implementations.Component = ComponentImplementation

else:

    class Component(co.ExplicitSystem, as_template=True):
        x = placeholder(default=2.)
        y = placeholder(default=1.)

        output.z = x**2 + y

##########

import pytest
with pytest.raises(Exception):
    Component()
    raise Exception()
# try:
#     Component() # error
# except:
#     pass
# else:
#     #assert False
#     pass
# # similarly, prevent user models from adding fields

class MyComp0(Component):
    pass

assert MyComp0().z == 5

class MyComp5(Component):
    u = input()
    output.v = u**2 + 2*u + 1

out = MyComp0(3.)
assert out.z == 5
assert out.v == 13

class MyComp1(Component):
    x = input()
    y = input()

out = MyComp1(x=2., y=3.)
assert out.z==7

class MyComp2(Component):
    x = input()
    y = 3.


assert MyComp2(1.0).z==4
#MyComp2(3., 4.)

class MyComp3(Component):
    x = 3.
    y = 4.

assert MyComp3().z==13

class MyComp4(Component):
    u = input()
    x = u**0.5
    y = 0

    output.v = x + 5

out = MyComp4(4.)
assert out.v== 7
assert out.z== 4



"""
"""



