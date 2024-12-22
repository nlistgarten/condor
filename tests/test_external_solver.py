import casadi
import ndsplines
import numpy as np
import pytest

import condor as co
from condor.backends.casadi.utils import wrap


class TableLookup(co.ExternalSolverWrapper):
    def __init__(self, xx, yy, degrees):
        input_data = []
        for k, v in xx.items():
            self.input(name=k)
            input_data.append(v)
        output_data = []
        for k, v in yy.items():
            self.output(name=k)
            output_data.append(v)
        output_data = np.stack(output_data, axis=-1)
        self.interpolant = ndsplines.make_interp_spline(
            input_data, output_data, degrees
        )
        self.jac_interps = [
            self.interpolant.derivative(idx) for idx in range(self.interpolant.xdim)
        ]

    def function(self, *xx):
        raw_out = self.interpolant(np.array(xx).reshape(-1))
        wrapped_out = wrap(self.output, raw_out.squeeze())
        return wrapped_out

    def jacobian(self, *xx):
        array_vals = [
            interp(np.array(xx).reshape(-1))[0, :] for interp in self.jac_interps
        ]
        # TODO -- original implementation did not have transpose, but generic version
        # needs it
        return_val = np.stack(array_vals, axis=1)
        return casadi.DM(return_val)


data_yy = dict(
    sigma=np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1833, 0.1621, 0.1429, 0.1256, 0.1101, 0.0966],
            [0.4, 0.3600, 0.3186, 0.2801, 0.2454, 0.2147, 0.1879],
            [0.6, 0.5319, 0.4654, 0.4053, 0.3526, 0.3070, 0.2681],
            [0.8, 0.6896, 0.5900, 0.5063, 0.4368, 0.3791, 0.3309],
            [1.0, 0.7857, 0.6575, 0.5613, 0.4850, 0.4228, 0.3712],
        ]
    ),
    sigstr=np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.04, 0.7971, 0.9314, 0.9722, 0.9874, 0.9939, 0.9969],
            [0.16, 0.7040, 0.8681, 0.9373, 0.9688, 0.9839, 0.9914],
            [0.36, 0.7476, 0.8767, 0.9363, 0.9659, 0.9812, 0.9893],
            [0.64, 0.8709, 0.9338, 0.9625, 0.9778, 0.9865, 0.9917],
            [1.0, 0.9852, 0.9852, 0.9880, 0.9910, 0.9935, 0.9954],
        ]
    ),
)
data_xx = dict(
    xbbar=np.linspace(0, 1.0, data_yy["sigma"].shape[0]),
    xhbar=np.linspace(0, 0.3, data_yy["sigma"].shape[1]),
)


@pytest.fixture
def table():
    return TableLookup(data_xx, data_yy, 1)


def test_table_lookup(table):
    table(xbbar=0.5, xhbar=0.5)
    tt = table(xbbar=0, xhbar=0)
    assert tt.sigma == 0


def test_opt_on_table(table):
    class Opt(co.OptimizationProblem):
        xx = variable()
        yy = variable()
        interp = table(0.5, yy)
        objective = (interp.sigma - 0.2) ** 2 + (interp.sigstr - 0.7) ** 2

        class Options:
            print_level = 0
            exact_hessian = False

    opt3 = Opt()
