import condor as co
import numpy as np
from condor.settings import settings


def get_LTI(**kwargs):
    mod = settings.get_module("deferred_model_test", **kwargs)
    return mod.LTI

A = np.array([
    [0, 1],
    [0, 0],
])
B = np.array([[0,1]]).T

sp_dbl_int = get_LTI(A=A, B=B, dt=.5)

dbl_int = get_LTI(A=A, B=B)


print("\n"*10)
print("double int events (expected empty):")
print(dbl_int.Event.subclasses)
print("sampled double int events (expected DT update event)):")
print(sp_dbl_int.Event.subclasses)
print("same object in memory?")
print(dbl_int is sp_dbl_int)





