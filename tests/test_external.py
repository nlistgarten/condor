import numpy as np

import condor
from condor.backend import operators as ops


def test_external_solver_dict_io():
    class DictLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=3)
            self.output(name="x", shape=3)

        def function(self, inputs):
            x = inputs["a"] ** 2 + 2 * inputs["b"] ** 2
            return {"x": x}

        def jacobian(self, input):
            dxda = 2 * input.a * np.ones_like(input.b)
            dxdb = 4 * np.diag(input.b.squeeze())
            # out of order on purpose
            return {("x", "b"): dxdb, ("x", "a"): dxda}

    class Analytic(condor.ExplicitSystem):
        a = input()
        b = input(shape=3)
        output.x = a**2 + 2 * b**2

    class Jac(condor.ExplicitSystem):
        sys = DictLike()
        inp = input.create_from(sys.input)
        num_out = sys(**inp)

        anal_out = Analytic(**num_out.input)

        output.num_dxda = ops.jacobian(num_out.x, input.a)
        output.num_dxdb = ops.jacobian(num_out.x, input.b)

        output.anal_dxda = ops.jacobian(anal_out.x, input.a)
        output.anal_dxdb = ops.jacobian(anal_out.x, input.b)

    sys = DictLike()
    # out = sys(a=2, b=np.array([[1, 2, 3], [4, 5, 6]]))
    out = sys(a=2, b=np.array([[1, 2, 3]]))
    assert np.all(out.x.squeeze() == (out.a**2 + 2 * out.b**2).squeeze())  # [1, 0]

    out_jac = Jac(**out.input)
    assert np.all(out_jac.anal_dxda == out_jac.num_dxda)
    assert np.all(out_jac.anal_dxdb == out_jac.num_dxdb)


def test_external_solver_array_io():
    class ArrayLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=(2, 3))
            self.output(name="x")
