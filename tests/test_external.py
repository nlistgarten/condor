import numpy as np

import condor
from condor.backend import operators as ops


def test_external_solver_dict_io():
    class DictLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=3)
            self.output(name="x")

        def function(self, inputs):
            x = inputs["a"] ** 2 + 2 * inputs["b"] ** 2
            return {"x": x}

        def jacobian(self, input):
            dxda = 2 * input.a
            dxdb = 4 * input.b
            # out of order on purpose
            return {("x", "b"): dxdb, ("x", "a"): dxda}

    class Jac(condor.ExplicitSystem):
        sys = DictLike()
        inp = input.create_from(sys.input)
        out = sys(**inp)

        output.dxda = ops.jacobian(out.x, input.a)
        output.dxdb = ops.jacobian(out.x, input.b)

    sys = DictLike()
    out = sys(a=2, b=np.array([[1, 2, 3], [4, 5, 6]]))
    assert out.x == out.a**2 + 2 * out.b[1, 0]

    out_jac = Jac(**out.input)
    jac = out_jac.dxda
    assert jac.dxda == 2 * jac.a


def test_external_solver_array_io():
    class ArrayLike(condor.ExternalSolverWrapper):
        def __init__(self):
            self.input(name="a")
            self.input(name="b", shape=(2, 3))
            self.output(name="x")
