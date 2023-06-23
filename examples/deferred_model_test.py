import condor as co
from condor.settings import get_settings
import numpy as np

default_settings = dict(
    A=None,
    B=None,
    dt=0.,
    dt_plant=False,
)

use_settings = get_settings(default_settings)

dt = use_settings['dt']
dt_plant = use_settings['dt_plant']

class LTI(co.ODESystem):
    A = use_settings['A']
    B = use_settings['B']

    x = state(shape=A.shape[0])
    xdot = A@x

    if dt <= 0. and dt_plant:
        raise ValueError

    if B is not None:
        K = parameter(shape=B.T.shape)

        if dt and not dt_plant:
            u = state(shape=B.shape[1])

        else:
            # feedback control matching system
            u = -K@x
            output.u = u

        xdot += B@u

    if not (dt_plant and dt):
        dot[x] = xdot

if dt:
    class DT(LTI.Event):
        function = np.sin(t*np.pi/dt)
        if dt_plant:
            from scipy.signal import cont2discrete
            if B is None:
                B = np.zeros((A.shape[0], 1))
            Ad,Bd,*_ = cont2discrete((A,B,None,None), dt=dt)
            update[x] = (Ad - Bd@K) @ x
        elif B is not None:
            update[u] = -K@x




