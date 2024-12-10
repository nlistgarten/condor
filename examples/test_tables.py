import condor as co
import numpy as np

sig1 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.1833, 0.1621, 0.1429, 0.1256, 0.1101, 0.0966],
        [0.4, 0.3600, 0.3186, 0.2801, 0.2454, 0.2147, 0.1879],
        [0.6, 0.5319, 0.4654, 0.4053, 0.3526, 0.3070, 0.2681],
        [0.8, 0.6896, 0.5900, 0.5063, 0.4368, 0.3791, 0.3309],
        [1.0, 0.7857, 0.6575, 0.5613, 0.4850, 0.4228, 0.3712],
    ]
)
sig2 = np.array(
    [
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.04, 0.7971, 0.9314, 0.9722, 0.9874, 0.9939, 0.9969],
        [0.16, 0.7040, 0.8681, 0.9373, 0.9688, 0.9839, 0.9914],
        [0.36, 0.7476, 0.8767, 0.9363, 0.9659, 0.9812, 0.9893],
        [0.64, 0.8709, 0.9338, 0.9625, 0.9778, 0.9865, 0.9917],
        [1.0, 0.9852, 0.9852, 0.9880, 0.9910, 0.9935, 0.9954],
    ]
)
xbbar = np.linspace(0, 1.0, sig1.shape[0])
xhbar = np.linspace(0, 0.3, sig1.shape[1])

class Interp(co.TableLookup):
    bbar = input()
    hbar = input()
    sigma = output()
    sigstr = output()

    input_data[bbar] = xbbar
    input_data[hbar] = xhbar
    output_data[sigma] = sig1
    output_data[sigstr] = sig2

    class Casadi(co.Options):
        degrees = 1
interp = Interp(0.5, 0.5)


class MyOpt3(co.OptimizationProblem):
    xx = variable()
    yy = variable()
    interp = Interp(xx, yy)
    objective = (interp.sigma - 0.2)**2 + (interp.sigstr - .7)**2

    class Casadi(co.Options):
        exact_hessian = False

opt3 = MyOpt3()


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

class CDIFlapInterp(co.TableLookup):
    flap_defl = input()
    dCL_flaps_coef = output()
    CDI_factor = output()

    input_data[flap_defl] = adelfd
    output_data[dCL_flaps_coef] = asigma
    output_data[CDI_factor] = adel6

    class Options:
        degrees = 1

flap_defl = 7.
cdi_flap_interp = CDIFlapInterp(flap_defl=flap_defl)

class MyInterp(co.TableLookup):
    x = input()
    y = output()

    input_data[x] = np.arange(10)*0.1
    output_data[y] = (input_data[x]-0.8)**2


class MyOpt(co.OptimizationProblem):
    xx = variable()
    objective = MyInterp(xx).y

    class Options:
        exact_hessian = False

mysol = MyOpt().xx

class VDEL3_interp(co.TableLookup):

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

    class Options:
        degrees = 1

VDEL3_interp_obj = VDEL3_interp(flap_span_ratio=0.3, taper_ratio=0.2)


class MyOpt2(co.OptimizationProblem):
    xx = variable()
    yy = variable()
    objective = (VDEL3_interp(flap_span_ratio=xx, taper_ratio=yy).VDEL3 - 0.3)**2

    class Casadi(co.Options):
        exact_hessian = False

mysol2 = MyOpt2()


