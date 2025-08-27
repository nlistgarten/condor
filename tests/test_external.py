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

        if self.output_mode == 0:
            jac = np.zeros((4, 4))
            jac[:3, 0] = dxda.squeeze()
            jac[:3, 1:] = dxdb
            jac[3, 0] = dyda
            return jac
        if self.output_mode == 1:
            return {
                ("y", "a"): dyda,
                # should you have to provide this?
                # ("y", "b"): np.zeros((1,3)),
                ("x", "b"): dxdb,
                ("x", "a"): dxda,
            }
        if self.output_mode == 2:
            return dict(
                y__a=dyda,
                x__b=dxdb,
                x__a=dxda,
            )


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


@pytest.mark.parametrize("output_mode", range(3))
def test_external_jacobian(output_mode):
    kwargs = dict(a=rng.random(1), b=rng.random(3))
    nsys = Numeric(output_mode)

    class Jac(condor.ExplicitSystem):
        inp = input.create_from(nsys.input)

        nout = nsys(**inp)
        cout = Condoric(**inp)

        for output_ in Condoric.output:
            for input_ in input:
                setattr(
                    output,
                    f"nsys_d{output_.name}_d{input_.name}",
                    ops.jacobian(
                        getattr(nout, output_.name),
                        getattr(input, input_.name),
                    ),
                )
                setattr(
                    output,
                    f"csys_d{output_.name}_d{input_.name}",
                    ops.jacobian(
                        getattr(cout, output_.name),
                        getattr(input, input_.name),
                    ),
                )

    out_jac = Jac(**kwargs)
    for output_ in Condoric.output:
        for input_ in Jac.input:
            assert np.all(
                getattr(out_jac, f"nsys_d{output_.name}_d{input_.name}")
                == getattr(out_jac, f"csys_d{output_.name}_d{input_.name}")
            )


if __name__ == "__main__":
    test_external_jacobian(2)
