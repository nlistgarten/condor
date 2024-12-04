import condor as co
import numpy as np
from scipy.signal import cont2discrete

settings = co.settings.get_settings(
    A=None,
    B=None,
    dt=0.0,
    dt_plant=False,
)


class LTI(co.ODESystem):
    A = settings["A"]
    B = settings["B"]

    x = state(shape=A.shape[0])
    xdot = A @ x

    if settings["dt"] <= 0.0 and settings["dt_plant"]:
        raise ValueError

    if B is not None:
        K = parameter(shape=B.T.shape)

        if settings["dt"] and not settings["dt_plant"]:
            u = state(shape=B.shape[1])

        else:
            # feedback control matching system
            u = -K @ x
            dynamic_output.u = u

        xdot += B @ u

    if not (settings["dt_plant"] and settings["dt"]):
        dot[x] = xdot


if settings["dt"]:

    class DT(LTI.Event):
        function = np.sin(t * np.pi / settings["dt"])
        if settings["dt_plant"]:
            if B is None:
                B = np.zeros((A.shape[0], 1))
            Ad, Bd, *_ = cont2discrete((A, B, None, None), dt=settings["dt"])
            update[x] = (Ad - Bd @ K) @ x
        elif B is not None:
            update[u] = -K @ x
