import condor as co
import pytest

# TODO test as_template


class ComponentRaw(co.models.ModelTemplate):
    input = co.FreeField(co.Direction.input)
    output = co.AssignedField(co.Direction.output)

    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y


class ComponentImplementation(co.implementations.ExplicitSystem):
    pass


co.implementations.ComponentRaw = ComponentImplementation



class ComponentAsTemplate(co.ExplicitSystem, as_template=True):
    x = placeholder(default=2.0)
    y = placeholder(default=1.0)

    output.z = x**2 + y


@pytest.mark.parametrize('Component', [ComponentRaw, ComponentAsTemplate])
class TestPlaceholders:
    def test_default_impl(self, Component):
        class MyComp0(Component):
            pass

        assert MyComp0().z == 5


    def test_new_io(self, Component):
        class MyComp5(Component):
            u = input()
            output.v = u**2 + 2 * u + 1

        out = MyComp5(3.0)
        assert out.z == 5
        assert out.v == 16


    def test_use_placeholders(self, Component):
        class MyComp1(Component):
            x = input()
            y = input()

        out = MyComp1(x=2.0, y=3.0)
        assert out.z == 7


    def test_partial_placeholder(self, Component):
        class MyComp2(Component):
            x = input()
            y = 3.0

        assert MyComp2(x=1.0).z == 1**2 + MyComp2.y

        # TODO currently this works, but probably shouldn't
        # keyword args x=..., y=... does fail
        assert MyComp2(3.0, 4.0).z == 3**2 + 3


    def test_override_placeholders(self, Component):
        class MyComp3(Component):
            x = 3.0
            y = 4.0

        assert MyComp3().z == 3**2 + 4


    def test_computed_placeholder(self, Component):
        class MyComp4(Component):
            u = input()
            x = u**0.5
            y = 0

            output.v = x + 5

        out = MyComp4(4.0)
        assert out.v == 4**0.5 + 5
        assert out.z == (4**0.5)**2 + 0
