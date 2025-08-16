import numpy as np

import condor

# pass get_settings in a list of configurable variables with defaults
# it returns a dictionary with the the configured values
conf = condor.settings.get_settings(A=np.array([1.0]), B=None)
A, B = conf.values()


class LTI(condor.ODESystem):
    x = state(shape=A.shape[0])

    xdot = A @ x

    if B is not None:
        K = parameter(shape=B.T.shape)

        u = -K @ x
        dynamic_output.u = u

        xdot += B @ u

    dot[x] = xdot
