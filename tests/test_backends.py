import numpy as np

import condor as co

backend = co.backend
ops = backend.operators


def test_reshaping():
    class Reshape(co.ExplicitSystem):
        """
        generates trapezoidal wing
        """

        x = input(shape=3)
        y = input(shape=4)

        output.mat = x @ y.T

    x = np.arange(3) * 1.0
    y = np.arange(4) * 1.0

    assert np.all(Reshape(x, y).mat == (x[:, None] @ y[None, :]))
