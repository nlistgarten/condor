import condor as co
import numpy as np



A = np.array([
    [0, 1],
    [0, 0],
])
B = np.array([[0,1]]).T

sp_mod = co.settings.get_module("deferred_model_test", A=A, B=B, dt=.5)
sp_dbl_int = sp_mod.LTI


ct_mod = co.settings.get_module("deferred_model_test", A=A, B=B)
dbl_int = ct_mod.LTI


print("\n"*10)
print("double int events (expected empty):")
print(dbl_int.Event.subclasses)
print("sampled double int events (expected DT update event)):")
print(sp_dbl_int.Event.subclasses)
print("same object in memory?")
print(dbl_int is sp_dbl_int)





