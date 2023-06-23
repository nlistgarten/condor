import condor as co
import numpy as np
from condor.settings import set_settings
import importlib

def get_LTI(**kwargs):
    set_settings(kwargs)
    import deferred_model_test as module
    importlib.reload(module)
    return module.LTI

dbl_int = get_LTI(
    A = np.array([
        [0, 1],
        [0, 0],
    ]),
    B = np.array([[0,1]]).T
)

sp_dbl_int = get_LTI(A=dbl_int.A, B=dbl_int.B, dt=.5)

print("\n"*10)
print("double int events (expected empty):")
print(dbl_int.Event.subclasses)
print("sampled double int events (expected DT update event)):")
print(sp_dbl_int.Event.subclasses)
print("same object in memory?")
print(dbl_int is sp_dbl_int)





