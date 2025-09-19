import numpy as np

import condor
from condor.backend import operators as ops


def simple_rot(th, axis):
    non_axis = [i for i in range(3) if i != axis]
    dcm = np.zeros((3, 3))
    dcm[axis, axis] = 1
    dcm[non_axis[0], non_axis[0]] = np.cos(th)
    dcm[non_axis[0], non_axis[1]] = -np.sin(th)
    dcm[non_axis[1], non_axis[1]] = np.cos(th)
    dcm[non_axis[1], non_axis[0]] = np.sin(th)
    return dcm


class Numeric(condor.ExternalSolverWrapper):
    def __init__(self, output_mode):
        self.output_mode = output_mode

        self.input(name="x")
        self.input(name="y")
        self.output(name="DCM", shape=(3, 3))

    def function(self, inputs):
        dcm = simple_rot()

        if self.output_mode == 0:
            return np.concat([x.squeeze(), np.atleast_1d(y)])
            out = np.array((4, 1))
            out[:3, 0] = x.squeeze()
            out[3, 0] = y
        elif self.output_mode == 1:
            return dict(x=x, y=y)
        elif self.output_mode == 2:
            return x, y


class Condoric(condor.ExplicitSystem):
    a = input()
    b = input(shape=3)
    output.x = a**2 + 2 * b**2
    output.y = ops.sin(a)


rng = np.random.default_rng(12345)
