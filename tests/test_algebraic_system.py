import condor as co
import pytest

# TODO test with actual bounds


class Coupling(co.AlgebraicSystem):
    x = parameter()
    z = parameter(shape=2)
    y1 = variable(initializer=1.0)
    y2 = variable(initializer=1.0)

    y1_agreement = residual(y1 == z[0] ** 2 + z[1] + x - 0.2 * y2)
    residual(y2 == y1**0.5 + z[0] + z[1])


@pytest.mark.parametrize("enforce_bounds", [True, False])
def test_sellar_solvers(enforce_bounds):
    Coupling.Options.enforce_bounds = enforce_bounds
    Coupling(x=1, z=[5, 2])


def test_set_initial():
    Coupling.set_initial(y1=1.1, y2=1.5)
    Coupling(x=1, z=[5.0, 2.0])


def test_set_initial_typo():
    with pytest.raises(ValueError, match="variables"):
        Coupling.set_initial(x=1)
