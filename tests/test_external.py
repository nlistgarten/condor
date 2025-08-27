import numpy as np
import pytest

import condor
from condor.backend import operators as ops


class Numeric(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="a")
        self.input(name="b", shape=3)
        self.output(name="x", shape=3)
        self.output(name="y")

    def function(self, inputs):
        a, b = inputs.asdict().values()
        x = a**2 + 2 * b**2
        y = ops.sin(a)

        if self.output_mode == 0:
            return np.concat([x.squeeze(), np.atleast_1d(y)])
            out = np.array((4, 1))
            out[:3, 0] = x.squeeze()
            out[3, 0] = y
        elif self.output_mode == 1:
            return dict(x=x, y=y)
        elif self.output_mode == 2:
            return x, y

    def jacobian(self, input):
        dxda = 2 * input.a * np.ones_like(input.b)
        dxdb = 4 * np.diag(input.b.squeeze())
        dyda = ops.cos(input.a)
        # out of order on purpose
        jac = np.zeros((4, 4))
        jac[:3, 0] = dxda.squeeze()
        jac[:3, 1:] = dxdb
        jac[3, 0] = dyda
        return jac


class Condoric(condor.ExplicitSystem):
    a = input()
    b = input(shape=3)
    output.x = a**2 + 2 * b**2
    output.y = ops.sin(a)


rng = np.random.default_rng(12345)


@pytest.mark.parametrize("output_mode", range(3))
def test_external_output(output_mode):
    kwargs = dict(a=rng.random(1), b=rng.random(3))
    nsys = Numeric(output_mode)
    nout = nsys(**kwargs)
    cout = Condoric(**kwargs)

    for output in Condoric.output:
        assert np.all(getattr(nout, output.name) == getattr(cout, output.name))


@pytest.mark.skip(reason="Casadi backend doesn't support matrix/matrix jacobian yet")
def test_external_solver_dict_io():
    class Jac(condor.ExplicitSystem):
        inp = input.create_from(dsys.input)
        dsys_out = dsys(**inp)
        nsys_out = nsys(**inp)
        c_out = Condoric(**inp)

        output.dsys_dxda = ops.jacobian(dsys_out.x, input.a)
        output.dsys_dxdb = ops.jacobian(dsys_out.x, input.b)
        output.dsys_dyda = ops.jacobian(dsys_out.y, input.a)
        output.dsys_dydb = ops.jacobian(dsys_out.y, input.b)

        output.nsys_dxda = ops.jacobian(nsys_out.x, input.a)
        output.nsys_dxdb = ops.jacobian(nsys_out.x, input.b)
        output.nsys_dyda = ops.jacobian(nsys_out.y, input.a)
        output.nsys_dydb = ops.jacobian(nsys_out.y, input.b)

        output.c_dxda = ops.jacobian(c_out.x, input.a)
        output.c_dxdb = ops.jacobian(c_out.x, input.b)
        output.c_dyda = ops.jacobian(c_out.y, input.a)
        output.c_dydb = ops.jacobian(c_out.y, input.b)

    dsys_out = dsys(a=2, b=np.array([[1, 2, 3]]))
    nsys_out = nsys(**dsys_out.input)

    assert np.all(
        dsys_out.x.squeeze() == (dsys_out.a**2 + 2 * dsys_out.b**2).squeeze()
    )  # [1, 0]
    assert np.all(dsys_out.y.squeeze() == np.sin(dsys_out.a).squeeze())  # [1, 0]

    out_jac = Jac(**dsys_out.input)
    assert np.all(out_jac.c_dxda == out_jac.dsys_dxda)
    assert np.all(out_jac.c_dxdb == out_jac.dsys_dxdb)
    assert np.all(out_jac.c_dyda == out_jac.dsys_dyda)
    assert np.all(out_jac.c_dydb == out_jac.dsys_dydb)

    assert np.all(out_jac.c_dxda == out_jac.nsys_dxda)
    assert np.all(out_jac.c_dxdb == out_jac.nsys_dxdb)
    assert np.all(out_jac.c_dyda == out_jac.nsys_dyda)
    assert np.all(out_jac.c_dydb == out_jac.nsys_dydb)


if __name__ == "__main__":
    test_external_output(2)
