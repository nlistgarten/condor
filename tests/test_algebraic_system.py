import pytest
import numpy as np

import condor as co

# TODO test with actual bounds


@pytest.fixture
def sellar_system():
    class Coupling(co.AlgebraicSystem):
        x = parameter()
        z = parameter(shape=2)
        y1 = variable(initializer=1.0)
        y2 = variable(initializer=1.0)

        y1_agreement = residual(y1 == z[0] ** 2 + z[1] + x - 0.2 * y2)
        residual(y2 == y1**0.5 + z[0] + z[1])

    return Coupling


@pytest.mark.parametrize("enforce_bounds", [True, False])
def test_sellar_solvers(sellar_system, enforce_bounds):
    sellar_system.Options.enforce_bounds = enforce_bounds
    out = sellar_system(x=1, z=[5, 2])

    # residuals bound, get by name
    assert np.isclose(out.residual.y1_agreement, 0, atol=1e-6)

    assert np.isclose(out.variable.y1, 25.5883, rtol=1e-6)
    assert np.isclose(out.variable.y2, 12.0585, rtol=1e-6)


def test_set_initial(sellar_system):
    sellar_system.set_initial(y1=1.1, y2=1.5)
    out = sellar_system(x=1, z=[5.0, 2.0])
    for _, resid in out.residual.asdict().items():
        assert np.isclose(resid, 0, atol=1e-6)


def test_set_initial_typo(sellar_system):
    with pytest.raises(ValueError, match="variables"):
        sellar_system.set_initial(x=1)
