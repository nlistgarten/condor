import condor as co
import numpy as np


class MyInterp(co.Tablelookup):
    x = input()
    y = output()

    input_data[x] = np.arange(10)*0.1
    output_data[y] = (input_data[x]-0.8)**2


class MyOpt(co.OptimizationProblem):
    xx = variable()
    objective = MyInterp(xx).y

    class Casadi(co.Options):
        exact_hessian = False

mysol = MyOpt().xx


class VDEL3_interp(co.Tablelookup):

    flap_span_ratio = input()
    taper_ratio = input()
    VDEL3 = output()

    input_data[flap_span_ratio] = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    input_data[taper_ratio] = [0.0, 0.33, 1.0]
    output_data[VDEL3] = np.array([
            [0.0, 0.0, 0.0],
            [0.4, 0.28, 0.2],
            [0.67, 0.52, 0.4],
            [0.86, 0.72, 0.6],
            [0.92, 0.81, 0.7],
            [0.96, 0.88, 0.8],
            [0.99, 0.95, 0.9],
            [1.0, 1.0, 1.0],
    ])



    class Casadi(co.Options):
        degrees = 1

VDEL3_interp_obj = VDEL3_interp(flap_span_ratio=0.3, taper_ratio=0.2)


class MyOpt2(co.OptimizationProblem):
    xx = variable()
    yy = variable()
    objective = (VDEL3_interp(flap_span_ratio=xx, taper_ratio=yy).VDEL3 - 0.3)**2

    class Casadi(co.Options):
        exact_hessian = False
