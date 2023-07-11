import condor as co
import numpy as np

adelfd = np.array(
    [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 38.0, 40.0, 42.0, 44.0, 50.0, 55.0, 60.0]
)
# flap angle correction of oswald efficiency factor
adel6 = np.array(
    [1.0, 0.995, 0.99, 0.98, 0.97, 0.955, 0.935, 0.90, 0.875, 0.855, 0.83, 0.80, 0.70, 0.54, 0.30]
)
# induced drag correction factors
asigma = np.array(
    [0.0, 0.16, 0.285, 0.375, 0.435, 0.48, 0.52, 0.55, 0.575, 0.58, 0.59, 0.60, 0.62, 0.635, 0.65]
)

class CDIFlapInterp(co.Tablelookup):
    flap_defl = input()
    dCL_flaps_coef = output()
    CDI_factor = output()

    input_data[flap_defl] = adelfd
    output_data[dCL_flaps_coef] = asigma
    output_data[CDI_factor] = adel6

    class Casadi(co.Options):
        degrees = 1

flap_defl = 7.
cdi_flap_interp = CDIFlapInterp(flap_defl=flap_defl)

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

mysol2 = MyOpt2()


