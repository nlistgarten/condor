import numpy as np

import condor
from condor.backend import operators as ops


def test_external_solver_dict_io():
    class DictLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=3)
            self.output(name="x", shape=3)
            self.output(name="y", shape=1)

        def function(self, inputs):
            x = inputs["a"] ** 2 + 2 * inputs["b"] ** 2
            y = ops.sin(inputs["a"])
            return dict(x=x, y=y)

        def jacobian(self, input):
            dyda = ops.cos(input.a)
            dxda = 2 * input.a * np.ones_like(input.b)
            dxdb = 4 * np.diag(input.b.squeeze())
            # out of order on purpose
            return {
                ("y", "a"): dyda,
                # should you have to provide this?
                # ("y", "b"): np.zeros((1,3)),
                ("x", "b"): dxdb,
                ("x", "a"): dxda,
            }

    class NumericLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=3)
            self.output(name="x", shape=3)
            self.output(name="y")

        def function(self, inputs):
            a, b = inputs.asdict().values()
            x = a**2 + 2 * b**2
            y = ops.sin(a)
            return np.concat([x.squeeze(), np.atleast_1d(y)])

            return x, y
            out = np.array((4, 1))
            out[:3, 0] = x.squeeze()
            out[3, 0] = y
            return out

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

    dsys = DictLike()
    nsys = NumericLike()

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


def test_external_solver_array_io():
    class ArrayLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=(2, 3))
            self.output(name="x")


if __name__ == "__main__":
    test_external_solver_dict_io()
